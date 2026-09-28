# Conventions

These rules apply to anyone changing this repository, human or AI assistant.
The code rules describe what the codebase already does, so new code should
match them. The research rules exist because breaking them has already cost
the project a full run (see [`RUN_HISTORY.md`](RUN_HISTORY.md)).

## Research integrity rules

These rules come first. Breaking any of them invalidates results.

1. **Held-out split.** Build the KG only from CultureBank and evaluate only on
   CulturalBench. Nothing under `src/construction/` may read CulturalBench, and
   evaluation must not tune on CultureBank-derived answers.
2. **Only the evidence changes.** Every system shares the model, decoding,
   `top_k`, and prompt scaffold. A new system or ablation may change only the
   evidence block (and its header in `llm.prompts.MCQ_EVIDENCE_HEADERS`).
3. **No silent fallback in research runs.** `--allow_fallback` is for tests
   and development only. Code that falls back (schema triples, rule-based
   critique, a built-in sample dataset) must *tag* the output (`llm_failed`,
   `extraction_method`, `generated`), and downstream stages must be able to
   exclude it. Eval refuses fallback CulturalBench IDs; keep it that way.
4. **Provenance is never dropped.** Every triple keeps `dataset_id`,
   `instance_id`, `source_text`, `evidence`, `extraction_method`, `confidence`,
   and `generated`. Critique suggestions are `generated=True` with low
   confidence and must never be presented as ground truth.
5. **Fail loud.** A stage that produces implausible output (near-zero
   extraction success, zero FULL pathways) must raise or fail a gate
   (`scripts/validate_corpus.py --strict`). Never log a warning and carry on.
6. **One run, one artifact set.** Reports quoted in docs must come from the same
   checkpoint as the corpus they describe. When they cannot, label them
   historical (as `phase2_validation.md` is now).
7. **Report negative results as they are.** Docs state the checked-in result,
   even when it contradicts the hypothesis. Label preliminary or small-sample
   results as such, with n and significance.
8. **Scoring definitions are fixed.** An Easy question is one unit. A Hard
   question is correct only if *all* of its True/False rows are correct.
   Combined accuracy pools both. Any change needs a matching test and a
   RUN_HISTORY entry.

## Code layout

- Put all logic in `src/<subpackage>/`. `scripts/` holds thin CLIs that parse
  arguments, call library functions, and write outputs.
- Respect the dependency order in [`ARCHITECTURE.md` §2](ARCHITECTURE.md#2-package-layering).
  `tests/test_architecture.py` enforces it. To add an edge, update the doc
  table and the test's `ALLOWED` together.
- Inside `src`, use **relative imports**: `from ..core import config`,
  `from .pathway_builder import build_pathways`. Scripts and tests use absolute
  `src.` imports after the `sys.path.insert(...)` line (scripts need
  `# noqa: E402`).
- Import config as a module (`from ..core import config`, then
  `config.RETRIEVAL_MIN_SCORE`), not `from config import X`. Scripts override
  values at runtime by assigning to the module attribute, which only works when
  callers read the attribute at call time.
- Keep `src/legacy/` and `scripts/legacy/` frozen. Nothing new may depend on them.

## Configuration

- Every tunable goes in `src/core/config.py` as
  `NAME = type(os.getenv("NAME", "default"))`. Use `_get_bool` for flags.
- When a default was chosen because of a measured failure, put the number in a
  comment next to it (for example, "a single global 512-token budget ... made
  96% of the corpus fall back"). This is house style; keep doing it.
- Stage-specific LLM budgets go in `STAGE_MAX_TOKENS`, never a new global.
- Add new variables to the table in `ARCHITECTURE.md` §6 when they matter for
  experiments, and to `.env.example` when they matter locally.

## Data models and IO

- Cross-stage records are pydantic models in `src/core/schemas.py`. Use
  `field_validator(..., mode="before")` to coerce messy LLM output, such as
  `LayerLabel.coerce`.
- Build IDs with `core.utils.stable_id(*parts, prefix=...)` so reruns are
  deterministic. Never use random or uuid IDs.
- Store large collections as JSONL through `core.utils`/`core.storage` helpers,
  with a Parquet mirror for anything evaluation reads (`write_parquet`).
- Default paths come from `config`. Scripts expose `--*_path` / `--*_output`
  flags, and SLURM scripts pass them explicitly for tagged runs.
- LLM calls in batch stages go through `core.storage.generate_with_cache`. The
  cache key must include everything that changes the output (prompt, system
  prompt, stage, model, `max_tokens`).

## LLM access

- Use the `LLMBackend` interface (`src/llm/llm_backend.py`,
  `get_backend(provider, model=...)`) and batch prompts into one `generate()`
  call where possible; vLLM depends on this.
- All prompt text lives in `src/llm/prompts.py`. Do not inline prompts in
  other modules. Changing a prompt changes the experiment, so note it in the
  commit message.
- No paid APIs. Models are local or open (Ollama locally, vLLM on Narval).

## Style

- Python 3.11, `from __future__ import annotations`, type hints on public functions.
- Every module has a docstring saying what it does and, where relevant, why it
  exists in its current form.
- Use a module-level `LOGGER = get_logger("<module>")` from `core.utils`.
  Use `print` only for end-of-run summaries in scripts.
- Comments explain *why* (the constraint or incident), not *what*.
- Put private helpers behind a leading underscore; test through public
  functions where possible.

## Tests

- Tests are offline and deterministic. `tests/conftest.py` forces
  `EMBEDDING_BACKEND=tfidf` and `HF_HUB_OFFLINE=1` before importing `src`, and
  provides `FakeBackend` for scripted LLM responses. Never call a real model
  in tests.
- Every bug fix to the pipeline gets a regression test, preferably in
  `tests/test_pipeline_integrity.py` if it concerns corpus correctness.
- Patch objects, not strings: `monkeypatch.setattr(module.local_llm, "generate_json", ...)`.
- Run the whole suite before committing. It must be green; see `AGENTS.md`
  for how to run it on Narval.

## Git and artifacts

- Never commit `data/processed/`, caches, checkpoints, SLURM logs, or `lib/`
  (pyvis output). Only the files `.gitignore` explicitly allows under
  `data/outputs/` are tracked. The large graph exports go through Git LFS
  (`.gitattributes`).
- Branch per experiment or change, and write commit messages that say what
  changed in the *experiment* (prompts, thresholds, system definitions) as well
  as in the code.
- When a run finishes or a branch is parked or merged, update `docs/RUN_HISTORY.md`.
