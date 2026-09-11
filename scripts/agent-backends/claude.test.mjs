import test from "node:test";
import assert from "node:assert/strict";
import { z } from "zod";
import { run } from "./claude.mjs";

const tool = { function: { name: "lookup", description: "Lookup", parameters: {
  type: "object", properties: { ids: { type: "array", items: { type: "integer" } } }, required: ["ids"],
} } };
const start = { cwd: "/tmp/project", prompt: "Find it", instructions: "You are Thoth",
  model: "example-model", binary: "/usr/bin/claude", resume: "session-a", tools: [tool] };

test("SDK receives resume, persona, tools and approval callbacks; streamed text is not duplicated", async () => {
  const emitted = [];
  let closed = false;
  const requests = [];
  const sdk = {
    tool: (name, description, shape, handler) => ({ name, shape, handler }),
    createSdkMcpServer: (config) => config,
    query: ({ prompt, options }) => {
      assert.equal(options.resume, "session-a");
      assert.equal(options.pathToClaudeCodeExecutable, "/usr/bin/claude");
      assert.equal(options.systemPrompt.append, "You are Thoth");
      assert.equal(options.permissionMode, "default");
      assert.equal(options.allowDangerouslySkipPermissions, undefined);
      return {
        close() { closed = true; },
        async *[Symbol.asyncIterator]() {
          assert.equal((await prompt.next()).value.message.content, "Find it");
          const definition = options.mcpServers.akh.tools[0];
          assert.deepEqual(z.object(definition.shape).parse({ ids: [7] }), { ids: [7] });
          const result = await definition.handler({ ids: [7] });
          assert.equal(result.content[0].text, "found");
          assert.equal(result.isError, false);
          assert.equal((await options.canUseTool("Bash", { command: "touch file" })).behavior, "deny");
          yield { type: "system", subtype: "init", session_id: "session-a" };
          yield { type: "stream_event", event: { delta: { type: "text_delta", text: "hello" } } };
          yield { type: "assistant", message: { content: [{ type: "text", text: "hello" }] } };
          yield { type: "result", subtype: "success", is_error: false, result: "hello" };
        },
      };
    },
  };
  await run(start, sdk, z, (m) => emitted.push(m), async (type, fields) => {
    requests.push({ type, ...fields });
    return type === "tool" ? { success: true, text: "found" } : { allow: false };
  }, new AbortController());
  assert.equal(closed, true);
  assert.deepEqual(emitted.filter((m) => m.type === "text"), [{ type: "text", text: "hello" }]);
  assert.equal(emitted.at(-1).success, true);
  assert.deepEqual(requests[0], { type: "tool", name: "lookup", arguments: { ids: [7] } });
});

test("SDK exhaustion is an error and closes the query", async () => {
  let closed = false;
  const sdk = { tool: () => ({}), createSdkMcpServer: () => ({}), query: () => ({
    close() { closed = true; }, async *[Symbol.asyncIterator]() {},
  }) };
  await assert.rejects(run(start, sdk, z, () => {}, () => {}, new AbortController()), /without a terminal result/);
  assert.equal(closed, true);
});

test("plan mode denies a provider approval without asking the client", async () => {
  const sdk = { tool: () => ({}), createSdkMcpServer: () => ({}), query: ({ options }) => ({
    close() {}, async *[Symbol.asyncIterator]() {
      assert.equal(options.permissionMode, "plan");
      assert.equal((await options.canUseTool("Bash", {})).behavior, "deny");
      yield { type: "result", subtype: "error_max_turns", errors: ["budget exhausted"] };
    },
  }) };
  const emitted = [];
  await run({ ...start, plan: true }, sdk, z, (m) => emitted.push(m), () => assert.fail("approval bypass"), new AbortController());
  assert.equal(emitted.at(-1).success, false);
});

test("brokered profile removes native tools and settings while supplied MCP tools still execute", async () => {
  const sdk = { tool: (name, description, shape, handler) => ({ name, handler }), createSdkMcpServer: (config) => config,
    query: ({ options }) => ({ close() {}, async *[Symbol.asyncIterator]() {
      assert.deepEqual(options.tools, []);
      assert.deepEqual(options.settingSources, []);
      assert.equal(options.strictMcpConfig, true);
      assert.equal(options.settings.disableAllHooks, true);
      assert.ok(options.disallowedTools.includes("Bash"));
      const denial = await options.canUseTool("Bash", { command: "touch escaped" });
      assert.equal(denial.behavior, "deny");
      assert.match(denial.message, /provided mcp__akh__/);
      const toolResult = await options.mcpServers.akh.tools[0].handler({ ids: [7] });
      assert.equal(toolResult.isError, false);
      assert.equal(toolResult.content[0].text, "executor result");
      yield { type: "result", subtype: "success", result: "done" };
    } }),
  };
  await run({ ...start, brokered: true }, sdk, z, () => {}, async (type) => {
    assert.equal(type, "tool");
    return { success: true, text: "executor result" };
  }, new AbortController());
});
