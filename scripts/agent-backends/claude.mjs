// Official Claude Agent SDK bridge. stdout is exclusively Akh's NDJSON
// protocol; provider credentials stay in the SDK's normal authentication path.
import { createInterface } from "node:readline";
import { pathToFileURL } from "node:url";

export async function run(start, sdk, z, send, request, abortController) {
  const definitions = start.tools.map(({ function: spec }) => {
    const schema = z.fromJSONSchema(spec.parameters);
    return sdk.tool(spec.name, spec.description, schema.shape, async (arguments_) => {
      const result = await request("tool", { name: spec.name, arguments: arguments_ });
      const text = result.data == null ? result.text : JSON.stringify(result);
      return { content: [{ type: "text", text }], isError: !result.success };
    });
  });
  const mcpServer = sdk.createSdkMcpServer({ name: "akh", version: "1.0.0", tools: definitions });
  const options = {
    cwd: start.cwd,
    abortController,
    includePartialMessages: true,
    // Credential/profile scope comes from the runtime's configured environment.
    // Brokered turns exclude ambient settings; no OAuth cache is copied here.
    settingSources: start.brokered ? [] : ["user", "project", "local"],
    systemPrompt: { type: "preset", preset: "claude_code", append: start.instructions },
    permissionMode: start.plan ? "plan" : "default",
    mcpServers: { akh: mcpServer },
    ...(start.brokered ? {
      tools: ["AskUserQuestion"],
      strictMcpConfig: true,
      settings: { disableAllHooks: true },
      disallowedTools: ["Bash", "Read", "Write", "Edit", "Glob", "Grep", "Agent", "Task", "Skill", "NotebookEdit", "WebFetch", "WebSearch"],
      ...(Object.keys(start.aliases ?? {}).length ? { toolAliases: start.aliases } : {}),
    } : {}),
    // This namespaced server executes through Akh's own permission gate.
    allowedTools: start.tools.map(({ function: spec }) => `mcp__akh__${spec.name}`),
    canUseTool: async (name, input) => {
      if (name === "AskUserQuestion") {
        const result = await request("question", { questions: input });
        if (result.cancelled === true) return { behavior: "deny", message: "The user skipped these questions." };
        return { behavior: "allow", updatedInput: { ...input, answers: result.answers } };
      }
      if (start.brokered) return { behavior: "deny", message: `Native ${name} is disabled. Use the provided mcp__akh__ tools in the bound workspace. Nothing ran.` };
      if (start.plan) return { behavior: "deny", message: "Akh is in plan mode" };
      const { allow } = await request("approval", { name, arguments: input });
      return allow
        ? { behavior: "allow", updatedInput: input }
        : { behavior: "deny", message: "The Akh client declined this action" };
    },
    ...(start.binary ? { pathToClaudeCodeExecutable: start.binary } : {}),
    ...(start.model ? { model: start.model } : {}),
    ...(start.effort ? { effort: start.effort } : {}),
    ...(start.resume ? { resume: start.resume } : {}),
  };
  let hadDeltas = false;
  let hadReasoningDeltas = false;
  let completed = false;
  // Streaming input is required for in-process MCP tools in the Agent SDK.
  let releaseInput;
  const inputDone = new Promise((resolve) => { releaseInput = resolve; });
  async function* input() {
    yield { type: "user", message: { role: "user", content: start.prompt }, parent_tool_use_id: null, session_id: start.resume ?? "" };
    await inputDone;
  }
  const query = sdk.query({ prompt: input(), options });
  try {
    for await (const message of query) {
      if (message.type === "system" && message.subtype === "init") {
        send({ type: "session", id: message.session_id, model: message.model });
      } else if (message.type === "stream_event" && message.event?.delta?.type === "text_delta") {
        send({ type: "text", text: message.event.delta.text });
        hadDeltas = true;
      } else if (message.type === "stream_event" && message.event?.delta?.type === "thinking_delta") {
        send({ type: "reasoning", text: message.event.delta.thinking });
        hadReasoningDeltas = true;
      } else if (message.type === "assistant") {
        for (const block of message.message.content) {
          if (block.type === "text" && !hadDeltas) send({ type: "text", text: block.text });
          if (block.type === "thinking" && !hadReasoningDeltas) send({ type: "reasoning", text: block.thinking });
          if (block.type === "tool_use") send({ type: "tool_started", id: block.id, name: block.name });
        }
      } else if (message.type === "user" && Array.isArray(message.message.content)) {
        for (const block of message.message.content) {
          if (block.type === "tool_result") send({ type: "tool_completed", id: block.tool_use_id, success: !block.is_error, details: block });
        }
      } else if (message.type === "result") {
        completed = true;
        if (message.usage) send({ type: "usage", usage: message.usage });
        send({ type: "done", success: message.subtype === "success" && !message.is_error,
          text: message.result ?? "", error: message.errors?.join("; ") ?? message.subtype });
        break;
      }
    }
    if (!completed) throw new Error("Claude SDK ended without a terminal result");
  } finally {
    releaseInput();
    query.close();
  }
}

async function main() {
  const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
  const pending = new Map();
  const abort = new AbortController();
  let sequence = 0;
  let started = false;
  const send = (message) => process.stdout.write(`${JSON.stringify(message)}\n`);
  const request = (type, fields) => new Promise((resolve, reject) => {
    const id = ++sequence;
    pending.set(id, { resolve, reject });
    send({ type, id, ...fields });
  });
  const stop = () => {
    abort.abort();
    for (const { reject } of pending.values()) reject(new Error("Akh disconnected"));
    pending.clear();
  };
  lines.on("close", stop);
  process.on("SIGTERM", stop);
  for await (const line of lines) {
    const message = JSON.parse(line);
    if (message.type === "response") {
      pending.get(message.id)?.resolve(message.result);
      pending.delete(message.id);
    } else if (message.type === "cancel") {
      stop();
    } else if (message.type === "start" && !started) {
      started = true;
      const sdk = await import("@anthropic-ai/claude-agent-sdk");
      const { z } = await import("zod");
      // Keep stdin flowing while SDK tools wait for replies from Akh.
      void run(message, sdk, z, send, request, abort)
        .catch((error) => send({ type: "error", message: error.message }))
        .finally(() => { lines.close(); process.stdin.destroy(); });
    }
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    process.stdout.write(`${JSON.stringify({ type: "error", message: error.message })}\n`);
    process.exitCode = 1;
  });
}
