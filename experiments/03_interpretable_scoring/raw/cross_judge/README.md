# Cross-judge consolidation calls

The two GPT-5.4 (high reasoning) calls behind the paper's cross-judge constructs
(Prompt `prm:cross-judge`), one per corpus, as sent and as answered:

| File | What it is |
|---|---|
| `<corpus>_input.json` | the exact `system` and `user` messages sent, and the response id |
| `<corpus>_raw_response.json` | the verbatim model output (`output_text`), token usage, model, status, response id |

`ellipse_*` is response `resp_0579a07a…` (34 criteria, 15 constructs); `asap2_*` is
`resp_0acea532…` (47 criteria, 19 constructs). Both were read back by response id from the
OpenAI Responses API (`responses.retrieve` + `input_items.list`); the `recovered_from` field
records this.

`scripts/07_cross_judge_cluster.py` (default replay mode) checks on every run that

1. the re-rendered prompt (`prompts/cross_judge.py` + the runs' reliable criteria) is
   byte-identical to the saved `user` message, and the system message matches;
2. the saved raw text parses to the clusters written to `results/cross_judge/`;
3. those clusters validate: every criterion placed exactly once (34/34, 47/47), verbatim
   names, at most one criterion per judge per construct.

A live run (`--live --yes`) writes the request before the call and the raw output before
parsing, so a response that fails validation is still kept.
