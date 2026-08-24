# Run reports

`smoke.py` writes a JSON report here on every run.

`baseline-linux-cpu-qwen3-1.7b.json` is the verified reference run described in
`../FINDINGS.md` — Linux, CPU, float32, Qwen3-1.7B, all checks passing. Keep it:
it's what an Apple Silicon run should be compared against.

Everything else here is scratch. Not worth committing unless it shows something.
