# Agent demo over MCP (2026-11-02)

A real agent drove the verifier through `cumlx-mcp` on a port with an injected fault. This is roadmap T4.

- Agent: Claude Code headless, model `claude-haiku-5-5`, with only the `mcp__cumlx__*` tools allowed.
- Port: the `tiny-llama` fixture with fault F7 (RoPE base x10) in `model.layers.1.self_attn`, loaded through `--loader python:faulty:load`. The agent was not told about the fault.
- Result: 4 turns, cost USD 0.0058, no denied tool call.

The agent called `cumlx_capture` and `cumlx_diff`. Its answer:

> The port is not correct: `cumlx_diff` returned `diverged` (exit code 1), with 50 records passing and 8 failing against the golden trace. The module at fault is `model.layers.1.self_attn`, the first divergence in isolated mode with high confidence (rel_l2 0.063 against a 0.0001 threshold), and the tool's explanation places the defect inside that module.

`report.json`, read back after the run, gave the same verdict and module. One run on a tiny fixture shows the interface works for an agent. It does not measure how often agents interpret results correctly.
