# AgentRun: a harness for repetitive knowledge work

Author: Miguel Ríos Berríos (@MiguelriosEN), Grep.ai. Article id 2100840456200581120, created 2026-09-18T19:23:18.000Z, modified 2026-09-18T19:34:10.000Z.
Retrieved 2026-09-26 via FxTwitter mirror (direct X access returns 403). Verbatim transcription of block text; formatting approximated.

We built AgentRun for jobs our customers do thousands of times a day. Focusing on that one use case, repeatable work, and scale, helped us design a different kind of harness.

At Grep.ai, we've been building AI agents for enterprise companies since 2023. We've gone from brittle agents that made good demos, to putting agents on rails (static workflows with LLM loops) to make them work, and back to building agents on top of harnesses like the Claude Agent SDK in the span of three years. Yet we still felt we had not found the right tool for the problem our customers’ need us to solve.

Most of our customers perform repetitive knowledge work in a regulated space, thousands of times a day, each case a little different from the last.

Take one: an anti-money-laundering alert. A bank screens a customer against sanctions, politically-exposed-person and adverse-media lists, and the screening vendor returns every profile that shares the name. Around 95% of those alerts are false positives, but each one has to be cleared: read the articles, find the customer's own footprint in registries and press, compare age, location, occupation and family, apply the bank's procedure, and write down why. An analyst spends the better part of an hour on one alert . A frontier-model agent does the same work for about $3 an alert, because it reads the news, the court reports and the registries the way the analyst does, then reasons over all of it in one long session. The stakes are high: clearing a real match is a regulatory failure, so the procedure is written with an asymmetry in it, and the harness has to keep that asymmetry.

One could script the procedure: fixed queries against fixed sources, a conditional for each list. That's how we tackled this before agents. Or build an agent with the procedure as its system prompt, a browser, search, and other tools. That's how we solved it when agents started to "work". Both did their job. But the script breaks the first time a source changes its pages, and the agent is expensive at scale.

With the static workflow, you end up maintaining and constantly updating code. With a frontier model, an agent can figure out pretty much anything we throw at it, but the result is expensive and hard to govern. We work in regulated industries. Customers need to know how a decision was made, which evidence supported it, and which procedure the agent followed. They need to explain that to their own teams, auditors, and regulators. Agents make this hard.


## Let the agent do the job, then fire itself from it

AgentRun takes a different approach: let a smart agent do the job, then find a way to automate it as a workflow, so most cases never need the agent at all. It starts with a job and an SOP, a standard operating procedure. The agent selects tools, writes code when it needs to, and finds a way to finish the job. In learning mode, it also leaves notes about how it worked: which tools it used, what approach worked or failed, and any tips for the next agent to do the work more easily.

After several runs, the agent uses the traces and notes from the previous runs to write a workflow in the AgentRun DSL, our language for workflows. The workflow can contain research agents, judgment agents, typed questions, and ordinary code. We use agents we've tuned for those jobs, so they perform well at low cost.

As the workflow takes shape, the agent routes a subset of cases to it instead of doing the job itself. It keeps iterating on the workflow as more cases come in and increases the share of cases it does not handle directly. By the end of the run in the last section, 98 of every 100 cases go through the workflow, and the full agent handles the rest, the odd ones.

This approach has two main benefits:

1. Cost. We've seen cost savings of 50–95% using workflows vs an agent with a frontier model. The workflow is more efficient than LLM turns and even when nodes in the workflow are agents, these tend to be very inexpensive.

2. The workflow is a system whose steps we can inspect and evaluate separately. Model Governance becomes easier when a system can be decoupled into clear subsystems. And likewise, humans can inspect the workflow by looking at specific steps, their inputs, outputs, and routing logic instead of having to deduct these from agent traces and chain-of-thought.


### The shape of the program

We owe one framing to 12-Factor Agents, and we lean on it every day. Take a tool call apart and there are two things inside: a decision, which action and with what arguments, and an action, the code that carries it out. Nothing about the decision needs a text generator. It needs something that can pick an option well and say how sure it is. Follow that seam through a whole procedure and it shows up everywhere: a long compliance job is mostly decisions and mechanical steps, with a few places where someone genuinely has to go and look.

We build AI programs along that seam. An agent where something must be found or done in the world. A typed question where something must be decided. Plain code where the step is mechanical. And a handful of shapes that hold the three together. Until now the decisions were made by the same instrument as the searching, a language model with a prompt, and that is where much of the cost and the opacity came from. The AgentRun DSL gives each kind of step its own place, and the decisions get their own engine, further down.


## Built on Pi

Our harness is built on top of Pi. We use their loop for the general agent and the agents inside our workflows. Around it, AgentRun provides the instructions, tools, skills, workspace, budgets, and output requirements.

[media block]
Our tool gateway, Executor, connects the agent to hundreds of tools for web research, registries, and other work. Each run gets the tools it's allowed to use. The gateway also supports tool discovery, so we can expose a large catalog without putting every tool definition into every prompt.

We have been building our skill library for three years, most of it for compliance work. The same authoring and evaluation system we use for those skills is the one our users get to build skills for their own Grep.ai agents.


### Traces and notes

We save the tool calls, results, submissions, errors, and usage for each run. In the tuning process, each case has a JSONL trace file and a ledger entry with its outcome, cost, and latency. We also retain the reasoning or summaries the model exposes, where available.

Those records let us inspect where the agent spent its time and money. We can see whether a tool returned useful evidence, an empty result, or an error, and what the agent did next.

When learning is enabled, we ask the agent to write its most reusable lesson into the workspace. A note might say that a particular registry website found the company when web search didn't, or that a page required a browser because a fetch couldn't expose the information. One to three sentences is enough.


## The AgentRun DSL

As we built workflows ourselves, we kept needing the same primitives. The AgentRun DSL is the small language that gives them names: a few kinds of agent, plain code, the shapes that hold steps together, and, newest, a way to ask a typed question and get a calibrated answer. The agent that authors a workflow plays the same twelve cards we play by hand.

[media block]
Everything in a workflow is one of these. Reading a workflow is reading which cards it plays, in what order, over which data.


### The agents

The agents are the characters. Each one runs the Pi loop with its own instructions, tools, model, and effort budget, and each has one job.

Research searches, fetches websites, and drives a browser through multi-page sites and logins. We tuned it for accuracy at low cost. Today it runs on DeepSeek V4.1 Flash, and it has completed nearly every research task we have given it.

Judge reads everything when the question is hard. It takes the procedure and the evidence and makes a supported determination: whether this record is our customer, whether this risk is low, medium, or high. We give it a smarter model and let it open a source to verify a claim. It is also the agent the rest of the workflow escalates to.

Report renders the record into the document the customer receives. It never re-decides. If a report is wrong, the separation tells us whether the fault was in gathering the facts, judging them, or writing them up.

Operator is the only agent that writes outside of the sandbox. It acts in the world: fills the portal, files the paperwork, submits the case to the system of record. It has a browser and the other write tools. Its effects leave the workspace, so it runs under the tightest tool grants and every action is on the trace.


### Code and mechanical steps

Some steps need no model at all. Code parses a payload, counts, compares, and decides what follows: the arithmetic of ages and dates, the tally of discrepancies, the rule that says two of these make a stop. A call performs one external action with no model in the loop. We can see the tool or command being invoked and its deadline. An artifact declares the deliverable and checks that the promised file exists.

Suppose an external service is generating a document. A call can poll until the service completes, reports failure, or reaches the deadline. We don't need a model turn for every status check. This excerpt specifies the polling behavior; the surrounding call names the tool and its inputs:

[media block]
The engine waits five seconds between checks, allows up to 15 seconds for each call, and stops polling after two minutes. The status values come from the service's contract.


### The shapes that hold them together

Steps run in a chain. Map / Reduce runs the same step over every item of a list at once, then folds the answers into one: twenty-four cards researched in parallel, one verdict. Loop until repeats a body within a bound and stops on a condition the state can answer: until nothing left can change the rating, at most three waves. Escalate routes a case off the cheap path, to the judge agent when a question is hard, or to a person with the record when the procedure requires review. It is the shape that makes the rest safe to run cheap.

We pass structured records between steps. Nodes declare required inputs, and the engine checks every submission against its schema. We can inspect the data at each handoff and test the next step with those same inputs.


### Decisions as typed questions: the Jev nodes

Routing, classification, and probabilistic decisions were always part of the AgentRun DSL. Before we had anything better, a constrained LLM call with a schema did them: cheap, but a language model all the same. It answers with a label and no calibrated sense of how sure it is, and it has to be prompted again for every new case.

AgentRun is the first harness built around Jev, TypeSafe's System One model, and it is the first time classification inside a procedure gets its own engine. Jev does not write text. It answers typed questions over a structured state: a choice among named options, a yes/no, a score on a scale, each with a probability distribution and a confidence that is calibrated, meaning a 0.7 is right about seven times in ten. An answer takes about 150 milliseconds and costs a few hundred-thousandths of a dollar. That turns out to be exactly the shape of most decisions inside a procedure.

The four purple cards are the four ways to ask:

- Route is a choice among branches of the workflow. Each branch carries the criteria under which the procedure takes it, and an "unsure" branch for when no option is confident.
- Classify compiles an output schema into questions. Every enum becomes a choice, every boolean a yes/no, and the criteria the author writes for each option are the criteria Jev applies. The values land in the workflow state; the probabilities land beside them, so the code that follows can ask not only what but how sure. (In the DSL this is the judge node.)
- Sift asks the same yes/no of every item in a list in one request and keeps the ones that pass: which of these sources is about our subject.
- Pick compares the items and chooses one, or none: which of these records is our customer.
Two more places Jev shows up without a card of its own. A verify clause on any agent turns the agent's own output schema into a checklist: when the agent submits, Jev reads the submission against the evidence the agent cited and answers, per field, whether the evidence supports the claim. A submission that fails comes back to the same agent session with exactly the fields it could not support, and the agent keeps working; claims still unsupported after the last round are removed before anything downstream sees them. And predicates on loops and escalations let a confidence gate a retry or a hand-off to review.

The rule we follow: Jev decides what a thing is; code decides what follows. A hit's name link is a nickname or it is not; its location is a different country or it is not; the arithmetic of ages and dates and the counting of discrepancies is code. Two things fall out of that. The decision is auditable, because every rating names the questions and probabilities it rests on. And the decision is replayable, because the answers are retained: we can change the code that reads them and re-run the judgment of a thousand cases in two minutes for a few cents, with no agent and no model call. Auto-tuning the judgment stops being a series of runs and becomes a replay.


### Evaluate one node or the whole workflow

Every card in a workflow has one model, one input contract, and one output schema. That is what makes it evaluable on its own. Say the final risk decision is wrong. We freeze the evidence the decision was made on and change only the thing under suspicion. We never run the whole investigation again for every experiment.

Evaluating a Jev node. The judgment in the alert review is 21 typed questions. Every answer and its probability is kept with the case, so the judgment of a thousand alerts is a table we can re-run without an agent and without a model call: change the code that reads the answers, replay, compare. When we rewrote the verdict rules to follow the rulebook clause by clause and deleted thirteen special cases, the replay showed the same rating on all 868 hits in two minutes for a few cents. When a question is the problem, we can see it: the one with the lowest confidence on the cases we got wrong is the one whose criteria need rewriting, and we can test the rewrite on the same table. A Jev question set is also something a compliance officer can read. It is the rubric, stated once, in the bank's own words.

Evaluating an agent. The research agent is judged on what it hands over, not on how it got there. Did the judge downstream have the facts the rulebook needed, or did a case escalate because a fact was missing? Did the verify pass hold up: how often did Jev send a submission back, and for which fields? Those two numbers, per card kind, tell us whether to fix the instructions, the tools, or the model. When we switched the research agent from a frontier model to DeepSeek V4.1 Flash, that is the test it had to pass, on the same cards, before the swap. The report agent is judged against the record it was given: every finding in the record is in the report, nothing in the report is outside the record.

Evaluating code. A code node is deterministic, so it is tested like any other code: the arithmetic of ages and dates, the counting of discrepancies, the rule that says two of these make a stop. When it changes, the replay above tells us exactly which cases changed rating and why.

Evaluating the whole workflow. After a node is fixed, we run the whole workflow on the same cases and compare versions side by side: the rating, the cost, the cards opened, the confidence behind each stop. A candidate version goes live only after it beats the current one on a development set. We learned to keep a second set the author never sees: on the banking messages, the author eventually matched every development label and still scored the same on the reserved set, so the development score alone is not a promotion criterion.

Two things we check on every judgment node. It needs the complete rubric relevant to its decision: criteria, exceptions, blockers, scoring. Splitting a workflow into smaller steps can leave a judge with half the policy, and the prompt a node actually receives is the thing to read. And every version is fingerprinted: the workflow, the SOP, and the skill used in each run are recorded, structure is validated before execution (required inputs, schemas, loop bounds), and the results behind each promotion are kept.

What this does for model governance. Model risk teams ask four questions of any automated decision: which model made it, on what inputs, under which version of the policy, and can you show me. A workflow answers all four by construction. Each decision names its node, its model, its inputs, and the questions and probabilities it rests on, so an examiner can follow one alert from the card to the rating. A model change is a change to one card, evaluated on that card's own inputs, with the before and after kept; swapping the research model does not touch the judgment, and the replay proves it. When the policy changes, the last quarter's judgments can be re-run under the new rules without re-researching a single case, which turns a lookback from a project into an afternoon. Confidence is a number the policy can set: clears below a chosen confidence go to review, and the threshold itself is versioned. And the boundary between what a model decides and what code decides is written down, so the parts a validator needs to test deterministically are deterministic.

Saved traces let us reconstruct inputs for all of this. For reviewed workflows, crash recovery currently covers flat chains of code, calls, polling, and file artifacts; model nodes and nested workflows are not on that path yet. If an external action's outcome is unknown, we preserve it for reconciliation before anyone retries it.

Sometimes the fix is a new step. Often it is a better instruction, a better query, or a smaller classification task that can move to a cheaper model, or to a Jev question.


## Auto-tuning an AML alert review

The alert review is the clearest picture of what the harness does: let a smart agent do the job once, the expensive way, and let it write down how the job should really be done.


### The job

You are an analyst. On your desk: one customer's file, and 24 cards from the watch-list company, 24 people with the same name who have been in trouble somewhere. Two questions. Is our customer any of these people? And if so, is it bad enough to turn them away?

The customer is 18, an apprentice, from a small town. The cards carry a name, what happened, and a country. Nothing more. The bank's procedure is 508 lines long.

[media block]

### Day one: by the book

The agent does what the procedure says an analyst does. Open every card. Look the person up. Find their age, their job, their town, their family. Compare with the customer. Write it down. Twenty-four times.

On this alert that was 826 tool calls, 271 web pages, 51 minutes, and in two other runs it hit the hour and never finished. The answer was right: STOP. Cards 9 and 21 are one drug trafficker with a long sentence, and nothing shows he is not our customer. The explanation was a page of prose.

[media block]

### What it writes down

While it works, the agent keeps notes, not about this customer, about the job. Where did the time go? What did it wish it had known at the start?

- "Cards 9 and 21 decided the whole case. If I had known up front they were the dangerous ones, I would have looked at those two and nothing else."
- "Card 3 was clearly not our customer, you could see it on the card. I did not need to open it. Same for 12 and 18."
- "After card 9 nothing could change the answer. I opened fifteen more anyway. Fifty minutes."
These are not answers. They are shortcuts, and they generalize: every alert has cards that could stop the customer and cards that could not, and every alert has a point after which nothing can change the answer.

[media block]

### The rulebook is turned into a program

The notes say which parts of the job are mechanical, which are a yes/no, and which need someone to go and look something up. So the rulebook is written again as a program, and every line says who does it. Simplified to its bones:

- jev(classify): which cards could stop the customer? The rulebook's list: sanctions, a senior official or their family, a serious conviction.
- agent(research): look those up, all at once.
- jev(classify): is any of them our customer? The rulebook's tests: age, place, job, family, how the names match.
- code(decide): if one of them is our customer, STOP. If not, look up the rest the same way. If none of the 24 is our customer, CLEAR. If we cannot tell for one of them, hand that card to the judge agent.
In the AgentRun DSL that program is a handful of nodes. Simplified, with the instructions and schemas left out:

[media block]
Three kinds of step. Code where it is mechanical. Jev where it is a closed question. An agent only where something must be found, or where no question could settle it. The agent, which used to be the whole job, is two lines.


### What a Jev question looks like

Jev never sees the procedure. It gets one question with the bank's criteria written into it, and one card. This is the request for "could this card stop the customer", simplified:

[media block]
The bank's policy is the instructions and the criteria, written once by whoever authors the workflow from the procedure. Jev's job is narrow: read the card, read the list, say how well the card fits. The answer is a probability, not a label, so a 0.44 can be treated differently from a 0.97, and the threshold is a number in code, not a feeling. If the bank adds a rule tomorrow, it is a line in that list, and the change replays over every past card in minutes.


### Day two: from the program

Same 24 cards, run through the program. Three are clearly not our customer from the card alone: gone. Jev asks the rest the rulebook's question; two could stop the customer. Both are looked up, together. One is him as far as anyone can show. STOP.

Two cards opened, at the same time. About 30 tool calls. About three minutes. The other 22 stay closed, and the record says why: three were not our customer on their face, and once card 9 was him, the other nineteen could not change the answer. The reason for the STOP is one line: card 9, the sentence.

[media block]

### Before and after

Side by side, the two ways of doing the same job. On the left, the agent by the book: a smart model with the procedure in its prompt, opening every card because nothing tells it which ones matter. 826 tool calls, 51 minutes, a page of prose. On the right, the same alert through AgentRun: the procedure written as a program, one question per card, two cards opened. Thirty tool calls, three minutes, and a record that names the card that decided it and says why the other 22 stayed closed. The agent on the left is not gone. It wrote the program on the right, and it is still there for the case the program cannot handle.

[media block]
[media block]



### What about cost and accuracy?

On 100 fresh alerts the same shape held: 30% of the card look-ups never ran, the case stopped early on 32 alerts, the customer's own web presence was established on 8 alerts instead of 100, and the cost per alert came down from $2.89 on the production agent to $0.25.

These are the arms we measured on those 100 alerts before the full run.

[media block]
The compliance judge's verdict on the disagreements with historical labels was the same across those arms, roughly two to one for the workflow, and the number of unsafe clears was lowest on the lean arms. The saving is not from a smaller model reading the same pages. It is from not reading pages whose content could not change the rating, and from a judgment that costs nothing.

We ran the 1,000 alerts in batches of 50. The allocation was fixed before the run: the first 100 alerts went to the full agent, so there would be traces and notes to learn from. Half of the next 200 could try the workflow, then 75% of the following 200, then 90%. The final 300 could all try it, with the full agent as the fallback. After every batch the author read the results, the traces, and the notes, and proposed a new version of the workflow. A candidate went live only after it beat the current version on the development set. Where a dot sits below the schedule, the workflow handed a case back to the agent: about one alert in twenty early on, about one in seventy over the last 300.

[media block]
Cost per alert fell in steps, not on a slope. Each step is a workflow version. The first version still researched every card, but replaced the agent's judgment with Jev questions and code. The second added the triage question and put the stoppers first. The third stopped a case once nothing left could change the rating. The fourth looked up the customer's identity only when a card needed it.

[media block]
Accuracy did not pay for the saving. The compliance judge scored every batch: 90% on the full agent in the first 100 alerts, 95% over the last 300. The versions that cut cost also raised accuracy, because each one removed a place where the agent could wander. Fewer pages read means fewer places to be wrong.

[media block]
The tuning run cost $1,049 in model usage, and $68 of that was the author's sessions and the development checks. Every alert on the full agent (Opus 5) would have cost $2,884, so the learning bill was recovered before alert 150. A bank does not review 1,000 alerts, it reviews 100,000, so we kept going. Over 100,000 alerts, with fallback and a retro every 1,000 included, the workflow cost $26K. The same alerts on Opus 5 would have cost $290K.

[media block]
The last 25 cents go to agents, not questions. Research agents on the cards that stayed open (DeepSeek V4.1 Flash) take more than half.  The escalation to the full agent (Opus 5) runs on 1.5%, and the report agent renders the record. The thirty or so Jev questions per alert cost a third of a cent together. The saving is in the 30% of cards that never opened.

[media block]

### That is AgentRun

A frontier agent does the job the expensive way once and writes down what it would do faster. Then it writes the program: a pipeline that moves between a typed question, a line of code, and a small agent loop only where something must be looked up or done. The program takes the ordinary cases. The agent stays for the strange ones. 

That is where we think agents are going. Not one long session that reads everything and reasons about all of it every time, but agents that learn a job, write it down as a program of cheap, inspectable steps, and get out of the way. The model that can figure anything out is the most expensive thing in the building. Use it to write the program, and to catch what the program cannot.

Jev is what makes that program possible, and it is a new kind of instrument. Until now, every judgment an agent made cost a model turn: slow, priced by the token, and answered with a label and no sense of how sure. A model that answers a typed question with a calibrated probability in 150 milliseconds for a few hundred-thousandths of a dollar changes what an agent can afford to ask. It can check every claim it makes against the source it cited. It can ask a question of every item in a list instead of a sample. It can gate its own next step on a confidence rather than a feeling. And when it learns a job, it can write its judgment down as questions and code, so the next run answers in milliseconds what took it minutes of reading. 

We keep that agent available for the cases the workflow can't handle. Operations teams already work this way: software automates the ordinary path and people handle the exceptions. We use the same division of work, with a frontier-model agent on the exceptions and human review where the SOP requires it. You want the ability to figure things out. You don't want to pay for it from scratch on every ordinary case.

We're rolling AgentRun out to Grep.ai enterprise customers this week, and to Pro customers in the days after. If you have a repetitive job you've wanted to hand to an agent but held back because agents felt too opaque or too expensive to run at volume, we'd like to prove you wrong on both counts. Go to grep.ai and try it, or DM me and I'll demo it on your job, not ours.
