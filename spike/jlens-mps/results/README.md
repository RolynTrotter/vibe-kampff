# Run reports

`smoke.py` writes a JSON report here on every run. Three are committed as
reference points, all described in `../FINDINGS.md`:

| File | What it is |
|---|---|
| `baseline-linux-cpu-qwen3-1.7b.json` | Linux, CPU, fp32. Established the code path; doubles as the fp32 reference for the precision comparison. |
| `qwen3-1.7b-mps-bfloat16.json` | Mac Studio, MPS, bf16. Same model as the baseline — the pair is the cross-device/cross-dtype check. |
| `qwen3-8b-mps-bfloat16.json` | Mac Studio, MPS, bf16, the project's actual analysis model. |

Anything else written here is scratch. Not worth committing unless it shows
something.
