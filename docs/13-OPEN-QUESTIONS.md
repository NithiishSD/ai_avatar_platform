# 13 — Open questions

Only the owner can answer these. None blocks building; each blocks the item
named. Reported in the final summary.

| ID | Question | Blocks | Default meanwhile |
|---|---|---|---|
| Q-01 | Is there target hardware (RTX 4090 / A100) for the final-gate numbers, or are the 6 GB host results the ones that count? | N-06, N-07, N-08, N-10 as "met" | Measure on what exists, report not met where not met |
| Q-02 | What does "> 95% lip sync accuracy" mean as a measurement? | N-05 as "met" | D-12 |
| Q-03 | Cloning similarity target: 85% (problem statement) or 90% (roadmap)? | N-02 wording | Report against both |
| Q-04 | Intact copy of the problem-statement PDF (the repo copy is corrupted) | confirming every PS threshold | Use the transcribed thresholds |
| Q-05 | A consented human face for demo evidence (photo of a team member with written consent, or a licensed stock photo with a model release) | admissible face evidence | Synthetic `demo` face: usable, never evidence |
| Q-06 | ~~m1 tracking CLAUDE.md~~ resolved: intended (D-01). Still open: 6 older `main` commits use `feat:`/`fix:` prefixes, and `00bffa1` on `m1` has a co-author trailer. Rewrite either, or leave? | nothing (main has no assistant wording) | Leave; never rewrite pushed history without the owner |
| Q-07 | Uptime > 99.5% needs a deployed environment. Is there one? | N-11 | Not measured |
| Q-08 | Cloud deployment / GPU autoscaling budget? | roadmap Phase 6 K8s item | Docker Compose |

## Manual checks for the owner

Things that need a person or the host GPU. Each row: what to run or open,
what to look for, and the answer once given. A task waiting on one stays
open; independent tasks continue meanwhile.

| ID | Task | Steps | Look for | Result |
|---|---|---|---|---|
