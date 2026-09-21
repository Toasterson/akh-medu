# Long-context soak, HIP, 2026-09-19

Ladder of prompts against the memory gate committed in `d2ced782`.

| log | what it is |
| --- | --- |
| `bbpxqdifo.log` | first ladder: 8k/16k/32k/48k/64k. Generation lines filtered out by the runner's grep, which is why the 8k result was misread at first. |
| `bmv8gcizu.log` | 8k and 16k rerun with full output. 8k round 0 succeeds at 10,417 tokens; 16k is admitted and then fails on `hipMalloc`. |
| `bywzq4occ.log` | after the fix: 20,833 tokens refused cleanly instead of failing on allocation. |
| `bwmw6r5is.log` | after the fix: 10,417 tokens still admitted and completes. Round 1 refused, which is the unresolved staleness defect. |

The result: sizing the gate on KV alone was wrong by 4.28x, and the soak is
what caught it. Every test before this used 15-token prompts, which pass
under both the wrong and the right limit.
