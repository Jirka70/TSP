# Methodology validation

Before an experiment runs, the project checks whether it is **methodologically sound**:
no data leakage, a split that matches the research question, reproducible settings, and so on.
A flawed experiment is reported or blocked **before** any data are loaded or models trained.

Three places work together:

| Where | What it contains | Who edits it |
|---|---|---|
| `configs/protocol/` | **What the experiment claims**: research question, evaluation scheme, usage, strictness | the experimenter, per experiment |
| `configs/methodology/` | **The rules**: which methodological errors must not happen | whoever maintains the rules |
| `src/validation/methodology/` | **The engine**: loads the rules and evaluates them against the experiment config | developers |

## How it fits together

```
configs/config.yaml  ──(Hydra)──►  ExperimentConfig ─────────────┐
   └─ protocol: within_session      (incl. protocol section)     │
                                                                 ▼
configs/methodology/rules_*.yaml ──► RuleValidator ──► MethodologyValidator.validate()
                                                                 │
                                                                 ▼
                                    MethodologyReport ──► methodology_report.json
                                                     └──► is_valid = False → experiment stops
```

In `src/main.py`, after the normal schema validation:

```python
methodology_report = MethodologyValidator.from_file().validate(ex_conf)
methodology_report.save(resolve_report_dir(ex_conf))
if not methodology_report.is_valid:
    return  # blocked
```

## 1. Protocol: what the experiment claims (`configs/protocol/`)

A Hydra config group; one file is selected per experiment (`protocol=cross_subject`).
It does not change the pipeline; it is a declaration the rules compare against.

```yaml
task: "motor imagery classification"
evaluation_scheme: cross_subject   # what is new in the test data: trials / session / subject / dataset
usage: offline                     # offline | simulated_online | online
rigor: standard                    # none | exploratory | standard | publication
```

- `evaluation_scheme`: which question the result answers, e.g. "does it work for a new user?".
  The split must match it.
- `usage`: whether a real-time claim is made (then only causal processing is allowed).
- `rigor`: how strict the validation is (see below); `none` skips the methodology validation entirely.

Details: `configs/protocol/_README.md`.

## 2. Rules: what must not happen (`configs/methodology/`)

Each `rules_*.yaml` file holds the rules of one section of the methodology document
(data leakage, reproducibility, ...). All files are merged into one catalog.

```yaml
  - id: LEAK-001
    title: Supervised transformation (CSP) fitted before the split
    certainty: proven
    when:                                   # condition over the config ...
      path: epoch_preprocessing.csp.enabled
      eq: true
    remediation: Use a model whose pipeline contains CSP.
```

A rule detects the error in one of two ways:

- `when`: a simple declarative condition over the config (`eq`, `in`, `all`, `any`, ...),
- `check`: the name of a Python function for more complex logic, with optional `params`.

Each rule also says how sure the detection is (`certainty`):

- `proven`: when it fires, the error is certainly there,
- `suspected`: the error is likely, but depends on things the validator cannot see.

Details and a checklist for writing rules: `configs/methodology/_README.md`.

## 3. What happens when a rule fires

The action depends on the rule's `certainty` and the protocol's `rigor`
(a rule can override this with its own `severity`):

| certainty \ rigor | none | exploratory | standard | publication |
|---|---|---|---|---|
| `proven` | off | warn | **error** | **error** |
| `suspected` | off | info | warn | **error** |

`error` blocks the experiment; `info` and `warn` are only logged and written to the report.
So the same config can run as an exploratory experiment but be blocked for publication.

`rigor: none` disables the methodology validation: `src/main.py` does not run the validator
at all (no report is written, a warning is logged) and the pipeline starts right after the
config validation. If the validator is called anyway, it skips every rule and the report is valid.
A rule's `severity` cannot set any other action than `off` for `none`.

## 4. The engine (`src/validation/methodology/`)

| File | Responsibility |
|---|---|
| `rule_model.py` | Schema of a rule (`MethodologyRule`), the condition (`Condition`), enums `Certainty`, `RuleAction`, `RulePhase`, default severities |
| `rule_validator.py` | Validates and loads rule YAML before methodology evaluation |
| `rule_loader.py` | Parses and merges rule files, validates the Pydantic schema, unique ids, and registered checks |
| `condition_evaluator.py` | Evaluates `when` conditions against the dumped config |
| `check_registry.py` | Registry of Python checks (`@methodology_check("name")`), `CheckContext`, `Finding` |
| `checks/static_checks.py` | Python checks evaluated from the config (e.g. `split_matches_protocol`) |
| `checks/runtime_checks.py` | Planned checks on real data after the split (not wired yet) |
| `methodology_validator.py` | `MethodologyValidator`: goes through the rules and builds the report |
| `methodology_report.py` | `MethodologyReport`: violations, passed / skipped rules, `is_valid`, JSON export |

What `MethodologyValidator.validate(config)` does for each rule:

1. **Skip** it if it is a runtime rule, if the experiment `mode` is not in its
   `pipeline_modes`, or if its action at this `rigor` is `off` → listed in `skipped_rules`.
2. **Evaluate** it: the `when` condition, or the registered Python `check` with `params`.
3. **No finding** → listed in `passed_rules`.
   **Findings** → each becomes a violation with the action for the current `rigor`, and is logged.

The result is saved as `methodology_report.json` in the Hydra output directory of the run.
It contains the protocol, the rigor, all violations (with impact and remediation),
and the lists of passed and skipped rules.

## Typical tasks

- **Run a stricter check** → `protocol.rigor=publication`.
- **Skip the methodology check** → `protocol.rigor=none`.
- **Test a new user scenario** → `protocol=cross_subject` + `split=cross_validation/moabb_cross_subject`.
- **Add a rule** → add it to the matching `configs/methodology/rules_*.yaml`
  (a Python check goes to `checks/static_checks.py`), then run
  `pytest tests/unit/validation/methodology`.
- **Add a protocol** → new file in `configs/protocol/`.

Tests: `tests/unit/validation/methodology/`. The engine is tested in
`test_methodology_engine.py`; demonstration scenarios A-F (faulty and correct experiments
in `configs/test/methodology/`) are tested in `test_methodology_scenarios.py`.
