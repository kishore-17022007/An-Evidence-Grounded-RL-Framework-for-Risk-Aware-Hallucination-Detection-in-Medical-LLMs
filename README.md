# MedVerify-RL

MedVerify-RL is a research and educational prototype for evidence-linked medical claim verification. It is not a clinical diagnostic system, is not clinically validated, and must not be used for treatment decisions.

## Current status

Phases 1 through 5 provide environment-based configuration, typed schemas, structured logging, a fictional sample fixture, a Streamlit shell, a budget-limited Gymnasium environment, a MedHallu data adapter, flat DQN, options-based hierarchical SMDP-DQN, and a live Europe PMC evidence-verification service. The service uses a deterministic sentence segmenter and transparent lexical baseline; it is not clinically validated and does not establish medical truth. The RL policies remain simulator-only. MAPPO, an entailment model, and live dashboard wiring are not implemented.

The sample file in `data/sample_cases.json` is synthetic development data only. It is not a medical dataset or benchmark, and its fictional statements must not be interpreted as medical information.

## Requirements

- Python 3.11 or newer
- pip

Dependencies include Streamlit, Gymnasium, Hugging Face `datasets`, PyTorch, `requests`, and pytest. No language-model package or API key is required.

## Setup

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install "requests>=2.32,<3"
cp .env.example .env
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install "requests>=2.32,<3"
Copy-Item .env.example .env
```

If PowerShell blocks environment activation, use `.venv\Scripts\activate.bat` from Command Prompt instead.

The settings loader reads environment variables directly. The `.env` file documents local values but is not auto-loaded; set variables in the shell before launching when overriding defaults. No API key is required.

## Run checks and app

```bash
python -m pytest
streamlit run app.py
```

## Phase 5 Evidence Verification

The Python service is `src/verification/workflow.py`. It validates bounded input, preserves exact original answer text/claim offsets, uses deterministic sentence segmentation, builds a query from each checkable claim and question, calls the official Europe PMC REST Search API for `core` metadata, ranks at most one verbatim abstract sentence per publication, and emits a typed claim assessment/report plus append-only JSONL audit events. `user_context` is returned to the caller but is not sent to Europe PMC or stored in the audit ledger. Potential phone, email, government-ID, or credential patterns are rejected before any network request. Search query terms are included in the audit because query provenance is required, but raw question/answer and optional context are omitted.

Run an explicit local fixture check (no network request):

```bash
python scripts/verify_answer.py \
	--offline \
	--question "Does compound X affect outcome Y?" \
	--answer "Compound X changes outcome Y in adult patients." \
	--output results/verification/offline_demo.json
```

The offline fixture is intentionally non-biomedical. It is not presented as evidence and should produce insufficient evidence.

Run the live Europe PMC workflow (uses the internet):

```bash
python scripts/verify_answer.py --live --question "Does aspirin reduce cardiovascular events in adults?" --answer "Aspirin reduces cardiovascular events in some adults." --output results/verification/live_report.json
```

The live API connectivity/schema smoke can be run separately:

```bash
python scripts/smoke_europe_pmc.py
```

### Europe PMC configuration

`VerificationConfig.from_env()` reads these optional settings; no API key is used or logged:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MEDVERIFY_EPMC_BASE_URL` | `https://www.ebi.ac.uk/europepmc/webservices/rest` | Official REST API base |
| `MEDVERIFY_EPMC_TIMEOUT` | `12` | Per-request timeout seconds |
| `MEDVERIFY_EPMC_RETRIES` | `2` | Bounded retry count |
| `MEDVERIFY_EPMC_BACKOFF` | `0.5` | Exponential retry base seconds |
| `MEDVERIFY_EPMC_INTERVAL` | `0.25` | Minimum interval between requests |
| `MEDVERIFY_MAX_RESULTS_PER_CLAIM` | `8` | Max core records per claim |
| `MEDVERIFY_MAX_CLAIMS` | `12` | Maximum sentence claims per answer |
| `MEDVERIFY_ACTION_BUDGET` | `24` | Search/assessment action budget |
| `MEDVERIFY_AUDIT_PATH` | `results/verification/audit.jsonl` | Append-only audit location |

Observed example from the explicit live run in this workspace (2026-10-18):

```text
Europe PMC smoke: SUCCESS, 2 records
End-to-end sample: 1 claim, 2 ranked excerpts, 2 cited PMIDs
Claim status: INSUFFICIENT_EVIDENCE (stance UNCLEAR)
Heuristic confidence: 0.4 (uncalibrated)
Run action count: 2 / configured budget 24
```

The actual identifiers in that run were `PMID:38117924` and `PMID:40497467`. The result was abstention/insufficient evidence because lexical relevance and related abstract language did not meet the exact-text or explicit-polarity rules. This is an example service execution, not a benchmark, model-accuracy result, clinical conclusion, or recommendation.

### Stance and status mapping

The baseline does not perform semantic entailment. Each excerpt has a stance `SUPPORTS`, `CONTRADICTS`, `MIXED`, or `UNCLEAR`. It marks `SUPPORTS` only when normalized claim text appears as a substring of a retrieved abstract and coverage, recency, and citation constraints pass. A `CONTRADICTS` flag requires high lexical overlap and a difference in explicit negation markers; this is only a review flag. Mixed, unclear, low-coverage, stale-only, citation-unavailable, and no-result cases map to `INSUFFICIENT_EVIDENCE`. A claim excluded by the deterministic checkability rule maps to `UNVERIFIABLE`. Retrieval/API failure maps to `INSUFFICIENT_EVIDENCE` with the failure stated, never to `CONTRADICTED`. Mixed evidence is not coerced to support or contradiction.

The `confidence` field is an uncalibrated heuristic ranking value, not a probability. Unclear/mixed evidence is capped at low heuristic confidence. `source_quality_assessment` describes available metadata only; it does not judge study design, bias, or clinical quality. Low lexical overlap and stale publication dates are exposed as missing-information limitations. Specialized diagnosis, medication, dosage, emergency, pregnancy, infant, or pediatric claims are forced to insufficient evidence with a clinical-judgment warning.

### RL boundary

The current DQN and HRL checkpoints accept a `(15,)` simulator observation, not raw claim text, abstracts, publication metadata, or evidence stances. `SimulatorPolicyBridge.from_checkpoints()` validates DQN dimensions, but `live_workflow_decision()` refuses to run unless given an actual compatible 15-feature simulator state. The live workflow therefore uses deterministic orchestration and records that DQN/HRL were not invoked. No live action is rewarded from lexical similarity, and no policy is updated from baseline judgments.

### Provenance, access, and limits

The client uses only Europe PMC's official REST Search API with JSON `core` results. It does not retrieve full text or scrape pages. Reports retain exact API identifiers and a source URL only when a PMID, PMCID, or DOI is actually returned; otherwise citation availability is explicitly false. Excerpts are verbatim substrings with checked offsets. Results are deduplicated across returned PMID/PMCID/DOI aliases. Article access/license fields are recorded when provided; `isOpenAccess` is not treated as a license grant, and users remain responsible for article-specific terms. The result cache is process-local and bounded; abstracts are not written to a cache file. Audit logs contain run IDs, claim IDs, query, API, identifiers, excerpts, action/decision and rationale, but not raw question/answer/context or credentials. The report may be saved locally and includes excerpts, not complete abstracts.

No NLI/LLM component is configured. Search ranking is not evidence of support. The claim extractor is sentence segmentation, not robust atomic-claim decomposition. Europe PMC abstracts may omit relevant methods/results and may not represent current guidelines. This tool is educational research software, not a diagnostic system; it must not provide personalized dosage or treatment recommendations, change prescriptions, or replace a clinician.

## MedHallu Data

The optional adapter uses the official `UTAustin-AIHealth/MedHallu` dataset and keeps `pqa_labeled` and `pqa_artificial` as separate configurations. The downloaded source schema observed in this workspace is `Question` (string), `Knowledge` (list of strings), `Ground Truth` (string), `Difficulty Level` (string), `Hallucinated Answer` (string), and `Category of Hallucination` (string). The source provides answer texts, not an explicit binary label column and not the four verification labels. The adapter derives an answer-origin target (`ground_truth_answer` or `hallucinated_answer`) from the selected source field and preserves the original target meaning separately; it does not relabel source records into the four-class task.

Raw records are saved without field changes under `data/raw/MedHallu/`. Processed wrappers preserving `original_fields` are saved under `data/processed/MedHallu/`, with provenance in `medhallu_download_metadata.json`. The downloader refuses to overwrite existing paths by default. To intentionally refresh existing downloads, pass `--allow-overwrite`; it prints a warning first.

```bash
python -m pip install -r requirements.txt
python scripts/download_medhallu.py
python scripts/split_medhallu.py --seed 17
python -m pytest
```

On this machine, the project-selected Python 3.11.9 is missing its standard-library `_lzma` extension, which prevents Hugging Face `datasets` from importing. The data was downloaded with the installed LZMA-capable Python 3.14.2 instead:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pip install "datasets>=3.0,<5"
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 scripts/download_medhallu.py
```

If download outputs already exist, append `--allow-overwrite`. Once data is present, use the project Python to split, adapt, and test:

```bash
python scripts/split_medhallu.py --seed 17
python -m pytest
```

The split script groups normalized exact duplicate questions, including duplicates across configurations, before assigning 80/10/10 train/validation/test groups with a fixed seed. Configuration folders remain separate. These are project-generated splits, not official dataset splits. Semantically similar questions with different wording may remain in separate groups; inspect split metadata and avoid using test data for training or tuning.

`load_medhallu_scenarios(..., configuration="pqa_labeled", split="train")` loads paired candidate-answer episodes from the selected split. The adapter defaults to `train` to discourage accidental held-out test access. Reset policy context contains only the question and selected candidate answer. A `RETRIEVE_EVIDENCE` or `INVESTIGATE_CONTRADICTION` action reveals one dataset `Knowledge` passage. MedHallu rows do not supply publication identifiers for these passages, so no independent source identifier is invented; requesting an independent source reports it unavailable. Lexical overlap is a relevance signal only, never proof of claim support.

The hidden answer-origin target is read only when a terminal action reward is calculated in the simulation. Evaluation tooling should obtain predictions first and access targets afterward. Do not use dataset answers as retrieved biomedical passages or as live evidence.

## Phase 3 DQN

`src/rl/dqn.py` implements a feed-forward PyTorch Q-network, online and target networks, replay buffer, epsilon-greedy selection, Huber loss, gradient clipping, hard/soft target updates, and checkpoint save/load. Network dimensions derive from the Gymnasium observation/action spaces. Device selection uses MPS when available with CPU fallback (`--device auto`); the selected interpreter on the development machine is Python 3.11.9 with PyTorch 2.14.1 and MPS available.

The CLI loads only the configured project `train` and `validation` splits. It rejects non-train training episodes and any train/validation overlap by source record ID. Validation runs with target-dependent environment rewards disabled; it creates predictions first, then reads answer-origin targets to calculate a clearly labelled answer-origin alignment metric. The test split is not loaded by the trainer. This remains simulator learning against answer-level MedHallu origins, not claim-level medical truth verification.

Quick real-data smoke training (small subset, 10 episodes, batch size 2, CPU; sized to perform optimizer updates):

```bash
python scripts/train_dqn.py --smoke --configuration pqa_labeled --seed 17
```

Fuller configurable training example:

```bash
python scripts/train_dqn.py \
	--configuration pqa_labeled \
	--episodes 1000 \
	--seed 17 \
	--max-action-budget 8 \
	--learning-rate 0.001 \
	--batch-size 64 \
	--replay-capacity 20000 \
	--gamma 0.99 \
	--epsilon-start 1.0 \
	--epsilon-end 0.05 \
	--epsilon-decay 0.995 \
	--target-update-mode hard \
	--target-update-interval 100 \
	--device auto
```

Best/final model checkpoints are written under `models/dqn/`; per-run JSON metadata and episode CSV logs are written under `results/dqn/`. Run metadata records the split/configuration, seed, scenario counts, device/runtime, episode rewards/losses/lengths/epsilon, action frequencies, validation predictions and answer-origin metric, and checkpoint paths. These are actual run outputs, not benchmark results. Do not tune against the test split.

## Phase 4 Options-Based HRL

`src/rl/hrl.py` implements a learned manager over four temporally extended options and four independent intra-option DQN workers. The manager has a separate Q-network over the existing 15-state observation and four options. It does not share parameters with the flat DQN or workers. Each worker is initialized independently from a seed and learns with its own DQN/replay/target networks, restricted to its option's actual `MedicalAction` subset. The manager uses SMDP Q-learning with a separate SMDP replay buffer.

For an option lasting $k$ primitive steps with rewards $r_0, \ldots, r_{k-1}$, its stored return and manager target are:

$$
R_o = \sum_{i=0}^{k-1} \gamma^i r_i,
\qquad
y_o = R_o + \gamma^k \max_{o' \in I(s')} Q_{\text{target}}(s', o'),
$$

where $I(s')$ is the set of options whose initiation predicate holds at the next state. No bootstrap is used when the environment terminates or truncates at the budget limit. Each option also has a maximum duration no longer than its contract; the environment action budget remains the global upper bound.

| Option | Initiation | Worker primitive choices | Termination |
| --- | --- | --- | --- |
| Evidence gathering | Budget remains and observed evidence coverage is below 1 | `RETRIEVE_EVIDENCE`, `REQUEST_INDEPENDENT_SOURCE` | State coverage reaches 0.5, option duration 2, or environment ends |
| Contradiction investigation | Budget remains and contradiction signal is below 1 | `INVESTIGATE_CONTRADICTION`, `RETRIEVE_EVIDENCE` | Contradiction signal reaches 0.5, option duration 2, or environment ends |
| Verification assessment | Budget remains and no verification has yet occurred | `VERIFY_CURRENT_CLAIM`, `RETRIEVE_EVIDENCE` | Verification count increases, option duration 2, or environment ends |
| Finalization | Budget remains | Four supported/contradicted/insufficient/abstain final actions | Final action ends the environment; duration 1 |

State-based termination takes precedence over the generic duration cap when both happen on one action, so logs retain the useful reason. When alternatives are initiable, the manager cannot immediately reselect the option it just completed; if it is the only initiable option, it remains eligible. Initiation and intra-option selection read only observations. Workers are trained on the same environment reward used by Phase 3; no option-specific intrinsic reward has been added. On MedHallu, the environment provides answer-origin simulator feedback only after a final action during training. It does not constitute medical truth or claim-level verification. The independent-source primitive reports unavailable for MedHallu; the contradiction option can only reveal another supplied `Knowledge` passage, not search an external database.

Run a short real-data HRL training/checkpoint smoke job:

```bash
python scripts/train_hrl.py --smoke --configuration pqa_labeled --seed 17
```

Fuller training example:

```bash
python scripts/train_hrl.py \
	--configuration pqa_labeled \
	--episodes 1000 \
	--seed 17 \
	--max-action-budget 8 \
	--batch-size 32 \
	--worker-batch-size 32 \
	--gamma 0.99 \
	--epsilon-decay 0.995 \
	--device auto
```

The HRL CLI loads only `train` and `validation`, rejects train/validation source-record overlap, validates that split metadata is project-generated, and disables target-dependent rewards during validation. Legacy callers retain validation-best checkpoint behavior by default. Manager checkpoints and one separate worker checkpoint per option are saved in unique `models/hrl/<configuration>_<timestamp>/{best_episode_<n>,final}/` directories. Per-episode returns, option choices/durations/termination reasons, primitive action counts, component losses, validation predictions/answer-origin alignment, and split provenance are under `results/hrl/`.

### Reproducible paired evaluation

Both training CLIs accept `--max-environment-steps`. This counts primitive `env.step()` calls and overrides the legacy `--episodes` limit. A new episode starts while budget remains; its environment action cap is reduced to the remaining global budget for the final partial episode, so the count reaches the requested budget without overshooting. The final partial episode is marked `truncated`, not naturally terminated. In HRL, an option interrupted by this global cap records its actual primitive duration and a truncated SMDP transition, so no continuation target is bootstrapped. Budgets smaller than one normal episode are allowed as a single partial episode. Manifests report requested/actual/unused interactions, completed episodes, stopping reason, and completion status.

The primary experiment must explicitly use `--checkpoint-policy fixed_final_at_budget` for both trainers. It evaluates only the final checkpoint after the requested interaction budget is reached; validation metrics remain descriptive and do not select a checkpoint. The shared evaluation protocol is ordered validation scenarios with greedy action selection and the same reset-seed schedule. A bounded final partial episode is allowed to consume the last remaining interactions; the final DQN transition and HRL SMDP option transition are marked truncated. A run that fails to reach the declared budget is marked `incomplete`. The comparison CLI requires both run manifests in this mode and checks agent identity, matching seed/configuration/split provenance/budget/action cap/evaluation protocol/validation count, completed status, selected final path, exact budget completion, and checkpoint hash. The legacy default `--checkpoint-policy legacy_validation_best` remains available for older callers and selects the best target-free validation-return checkpoint; it is not the primary experiment protocol. In either mode, answer-origin alignment and coverage are reported separately from target-free return/cost, with explicit denominators.

The primary 10,000-interaction training invocations are:

```bash
python scripts/train_dqn.py --configuration pqa_labeled --seed 17 --max-action-budget 8 --max-environment-steps 10000 --checkpoint-policy fixed_final_at_budget
python scripts/train_hrl.py --configuration pqa_labeled --seed 17 --max-action-budget 8 --max-environment-steps 10000 --checkpoint-policy fixed_final_at_budget
```

Before selecting a full training interaction budget, measure simulator-only throughput on train data with `python scripts/check_experiment_feasibility.py --configuration pqa_labeled --interactions 256 --seed 17`. The probe loads only the train split, uses seeded random actions, and prints an environment-stepping projection for 10,000 interactions; it excludes policy inference, optimizer updates, validation, and checkpoint I/O, so it is a lower-bound estimate rather than a training runtime guarantee.

Run outputs use unique timestamped run directories and checkpoint subdirectories. A run manifest is written only after its final checkpoint and episode log exist; its status distinguishes completed and incomplete runs. The manifest includes the selected checkpoint path/hash and a source diff hash (excluding `.env*` files) alongside the base commit and dirty-state flag. Comparison JSON output refuses an existing path. Answer-origin alignment is derived from the candidate's MedHallu source-field origin and must not be described as medical accuracy, clinical validation, or evidence correctness. No test split is used for training or checkpoint selection.

Compare the existing flat DQN checkpoint and an HRL checkpoint directory on the same held-out validation scenarios:

```bash
python scripts/compare_dqn_hrl.py \
	--configuration pqa_labeled \
	--checkpoint-policy fixed_final_at_budget \
	--dqn-checkpoint models/dqn/<run-id>/final.pt \
	--hrl-checkpoint-dir models/hrl/<run-id>/final \
	--dqn-run-manifest results/dqn/<run-id>/run.json \
	--hrl-run-manifest results/hrl/<run-id>/run.json \
	--max-action-budget 8 \
	--seed 17 \
	--output results/comparisons/dqn_vs_hrl.json
```

Both policies are evaluated greedily with the same ordered validation examples, seed, budget, and target-free environment rewards. The test split is not loaded. Reported answer-origin alignment is computed after all predictions and is not medical truth accuracy. A 10-episode smoke run only verifies pipeline execution; it is not evidence of HRL superiority.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `MEDVERIFY_SEED` | `17` | Reproducibility seed |
| `MEDVERIFY_OFFLINE` | `true` | Offline-mode status shown in the app |
| `MEDVERIFY_LOG_LEVEL` | `INFO` | Structured log severity |
| `MEDVERIFY_DATA_DIR` | `data` | Local data directory |
| `MEDVERIFY_DB_PATH` | `data/medverify.sqlite3` | Planned local history database path |

## Phase 2 Environment

`MedicalVerificationEnv` is a Gymnasium environment for one candidate answer per episode. It supports the existing synthetic scenario path and an opt-in MedHallu adapter path. In MedHallu episodes, initial policy context contains the source question and candidate answer; dataset `Knowledge` passages are revealed only after retrieval actions. Each scenario's answer-level training target remains separate from the numeric state and is read only for terminal reward. This is simulation infrastructure, not a medical verifier.

### Observation

The observation is a `float32` vector of shape `(15,)`; every element is clipped to `[0, 1]`. Feature order is fixed in `src/rl/state_encoder.py`:

| Index | Feature | Meaning |
| --- | --- | --- |
| 0 | `claim_complexity` | Scenario-provided normalized complexity signal |
| 1 | `current_evidence_relevance` | Current relevance signal for gathered evidence |
| 2 | `evidence_coverage` | Estimated coverage from observable scenario signals |
| 3 | `verification_confidence` | Confidence signal after the verify action |
| 4 | `contradiction_signal` | Observable possible-contradiction signal |
| 5 | `search_count_normalized` | Search actions divided by the action budget |
| 6 | `remaining_budget_fraction` | Remaining actions divided by the action budget |
| 7 | `workflow_stage_normalized` | Workflow stage mapped to `[0, 1]` |
| 8 | `previous_verification_confidence` | Confidence signal before the latest verification |
| 9 | `verification_count_normalized` | Verification count divided by the action limit |
| 10 | `retrieval_redundancy_fraction` | Repeated retrieval count, normalized by budget |
| 11 | `independent_source_fraction` | Independent-source requests, normalized by budget |
| 12 | `contradiction_investigation_fraction` | Contradiction investigations, normalized by budget |
| 13 | `uncertainty_estimate` | `1 - verification_confidence` |
| 14 | `evidence_item_count_normalized` | Observed evidence-item count normalized by budget |

Ground-truth labels, scenario type names, and future transition information are not observation features or reset/step info fields.

### Actions and transitions

`action_space` is `Discrete(8)`. Integer IDs are defined by `MedicalAction`:

| ID | Action | Transition |
| --- | --- | --- |
| 0 | Retrieve evidence | Synthetic scenarios update signals; MedHallu episodes reveal the next `Knowledge` passage |
| 1 | Verify current claim | Updates confidence from current coverage, relevance, and the prior-confidence signal |
| 2 | Investigate contradiction | Synthetic scenarios update a signal; MedHallu episodes reveal the next `Knowledge` passage |
| 3 | Request independent source | Synthetic scenarios update independent-source signals; MedHallu returns `independent_source_available=False` |
| 4 | Finalize as supported | Terminal decision with `SUPPORTED` label |
| 5 | Finalize as contradicted | Terminal decision with `CONTRADICTED` label |
| 6 | Finalize as insufficient evidence | Terminal decision with `INSUFFICIENT_EVIDENCE` label |
| 7 | Abstain or escalate | Terminal abstention with `UNVERIFIABLE` label |

An action consumes one unit of the finite budget. Repeating retrieval, verification, investigation, or independent-source requests is allowed but reported as redundant and penalized; repeated retrieval does not add new evidence signals. An action outside the discrete space raises `ValueError`. Stepping after an episode ends raises `RuntimeError`.

### Reward

`RewardWeights` exposes quality and cost scales and individual magnitudes. Terminal correct decisions earn `correct_decision`; marking a contradicted claim as supported receives the larger `missed_hallucination_penalty`; marking supported content as contradicted receives `false_alarm_penalty`. Insufficient-evidence decisions under inadequate or conflicting evidence receive a reasonable-uncertainty reward. Abstention is rewarded when evidence is inadequate, conflicting, or the hidden training target indicates an insufficient/unverifiable scenario; otherwise it is penalized. For MedHallu, the adapter's answer-origin target is mapped to an expected supported/contradicted action only inside the terminal reward calculation. This is an answer-level simulation target, not a source-provided four-class annotation. Retrieval/action costs, redundancy, unnecessary retrieval, unsupported assertion, and budget exhaustion are penalized.

Positive evidence-quality reward requires adequate observed coverage, relevance, a verification action, and confidence. Retrieval similarity or relevance alone is not enough. These reward labels are for simulated training/evaluation only and are never used to select actions or construct observations.

### Termination

Any of the four final-decision actions sets `terminated=True`. Exhausting the action budget or reaching `max_action_count` without a final decision sets `truncated=True` with a budget penalty. Gymnasium `reset(seed=...)` and five-value `step()` signatures are used. The default action budget is eight.

## Project layout

```text
app.py
config/                 Environment-based settings
src/common/             Schemas, exceptions, and structured logging
src/rl/                 Gymnasium environment, DQN, options/SMDP hierarchy
src/experiments/        MedHallu adapter, DQN/HRL training, validation comparison
scripts/                Download, split, DQN, HRL, and comparison CLIs
data/raw/MedHallu/      Downloaded original records (local data)
data/processed/MedHallu/Processed records and generated grouped splits
models/dqn/             Flat DQN checkpoints
models/hrl/             Manager and per-option worker checkpoints
results/dqn/            Flat DQN run metrics and episode logs
results/hrl/             HRL run metrics and option/action logs
results/comparisons/     Paired validation-only DQN/HRL reports
artifacts/              Reserved for generated artifacts
logs/                   Reserved for runtime logs
tests/                  Configuration and schema tests
```

## Phase 10 Curated Research Figures

The final 15-figure collection is in `results/graphs/phase10_curated_20261019/`; its `README.md` indexes each figure's research question, saved source artifacts, and interpretation. Existing graph outputs remain in their prior locations. The collection uses only saved run manifests, episode CSVs, and validation comparison reports; it does not load the test split or run training/evaluation.

Install the plotting extra and regenerate into the same new, non-existing destination only after preserving any prior output:

```bash
python -m pip install -e ".[graphs]"
python scripts/export_phase10_curated_graphs.py
```

The exporter validates the common 186-scenario validation order/reset schedule and refuses to overwrite its output directory. The Phase 9I guard figure is an evaluation-time constraint on the existing Candidate A checkpoints, not a retrained policy. Candidate A validation had 100% abstention, zero definitive decisions, and zero insufficient-evidence finalizations. Observable acquisition is a simulator state-change diagnostic, not evidence quality. No figure represents medical accuracy or clinical validity. Only the curated Phase 10 directory is unignored; other `results/` data remain ignored by Git.# An-Evidence-Grounded-RL-Framework-for-Risk-Aware-Hallucination-Detection-in-Medical-LLMs
