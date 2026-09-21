---
type: grilling
status: closed
resolved: 2026-09-19
---

## Question

How does the harvester run, given it must be independent of `akhomed` and the
sessions it harvests are produced on several machines?

## Resolution

**An idempotent one-shot CLI driven by a platform timer.** Scan the three sources,
resume from a per-`(machine, source, session)` watermark, push new records, exit.
systemd user timer here, launchd on the Mac mini, SMF on illumos.

Rationale: the transcripts are already durable on disk, so polling loses nothing
that a live tail would catch; there is no supervised process to babysit; a
watermark makes reruns free and hand-debugging trivial; and it ports across three
operating systems without a rewrite. The cost is latency bounded by the timer
interval, which no known consumer cares about.

Rejected: a long-lived watcher daemon (near-real-time, but costs a supervised
process per machine plus inotify/FSEvents/FEN portability work); hooks-plus-timer
(zero latency for Claude Code only, at the price of two code paths that must stay
consistent, and crashed sessions still need the sweeper anyway).

A `--watch` mode remains available later and sits in the map's fog.
