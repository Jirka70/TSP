# Methodology rules

Rules describing methodological errors that must **not** happen in an experiment.
They are evaluated by `MethodologyValidator` (`src/validation/methodology/`) after the
schema validation and before the pipeline is built. The result is written as a JSON
report next to the experiment outputs.

These files are **not** a Hydra config group: they are loaded directly and are not
overridden per experiment. Strictness is chosen per experiment via `protocol.rigor`
(`exploratory | standard | publication`).

## Folder layout

| File | Content |
|---|---|
| `README.md` | this guide |
| `rules_<section>_<name>.yaml` | rules of one section of the methodology document, top-level key `rules` |
| `rules_runtime_planned.yaml` | planned runtime rules (commented out, not active yet) |

Before methodology rules are evaluated, `RuleValidator` loads all `rules_*.yaml` files
through the Pydantic schemas in `rule_model.py`. The files are read in alphabetical order
and merged into one catalog; rule ids must be unique across all files. A new section = a
new `rules_*.yaml` file, nothing else has to be registered.

## Rule fields

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | Unique id; prefix = category: `LEAK`, `SEL`, `GRP`, `PROT`, `STAT`, `ONL`, `MET`, `REP`, `EVAL`, `PRE` (preprocessing and its compatibility with the model) |
| `title` | yes | Short name of the violation |
| `description` | no | What is wrong when the rule fires (used as the finding text for `when` rules) |
| `phase` | no | `static` (config, before the run; default) or `runtime` (data, during the run; not wired yet) |
| `certainty` | yes | `proven` or `suspected`: how reliable the detection is, see [Certainty](#certainty) |
| `pipeline_modes` | no | Pipeline modes the rule is applied to (= experiment `mode`: `training`, `experiment`, `benchmark`); omitted = all modes |
| `when` | one of `when` / `check` | Declarative condition over the config (static rules only), see [Condition DSL](#condition-dsl-when) |
| `check` | one of `when` / `check` | Name of a registered Python check, see [Python checks](#python-checks-check--params) |
| `params` | no | Values passed to the Python check as `ctx.params`; only allowed with `check` |
| `severity` | no | Action per rigor level, see [Severity](#severity) |
| `impact` | no | Possible impact on the reported results |
| `remediation` | no | How to fix the experiment |

Example:

```yaml
rules:
  - id: LEAK-002
    title: Raw augmentation before the split
    description: "raw_augmentation runs before the split stage; augmented copies can end up in both train and test."
    certainty: proven
    pipeline_modes: [ training, benchmark ]
    when:
      path: raw_augmentation.backend
      ne: none
    impact: Near-duplicate samples in train and test inflate the score.
    remediation: Use raw_augmentation=none and augment after the split (augmentation stage, train part only).
```

## Certainty

Says how sure we are that the experiment is really flawed **when the rule fires**.
It is about the reliability of the detection, **not** about how large the impact is
(a proven violation can have a small impact, a suspected one a large impact).

**Decision question:** *Can an experiment that triggers this rule still be methodologically correct?*

- no → `proven`
- yes / it depends → `suspected`

**`proven`**: the violation is necessarily present whenever the rule fires. It follows from
the config and the known behaviour of the pipeline alone; there is no legitimate situation
in which the rule fires and the experiment is fine (no false positives by design).
Typical: a supervised step fitted before the split, a split that does not match the
declared protocol, a shuffled split without a seed.
Examples: `LEAK-002` (raw augmentation before the split), `PROT-001`, `REP-001`.

**`suspected`**: the violation is likely, but whether it really happens (or matters)
depends on facts the validator cannot see from the config:

- the actual data (class balance, number of sessions when ids are `null`),
- a recommended threshold rather than a hard limit,
- a step that can be legitimately justified (e.g. label-free cleaning documented in the experiment record).

Examples: `LEAK-003` (unsupervised ICA on all epochs), `STAT-001` (few subjects),
`MET-001` (accuracy only, which is fine if classes are balanced).

Guidelines:

- When unsure, use `suspected`.
- If a check can only prove some cases, report only the provable ones under a `proven`
  rule (as `enough_groups_for_protocol` does for explicit id lists), or split it into
  a proven and a suspected rule.

## Severity

Action taken when the rule fires, per rigor level: `off | info | warn | error`
(`error` = the experiment is blocked, `off` = the rule is not evaluated).
Levels that are not listed default by `certainty`:

| certainty | exploratory | standard | publication |
|---|---|---|---|
| `proven` | warn | error | error |
| `suspected` | info | warn | error |

Override example: `severity: {exploratory: info, standard: warn, publication: error}`.

Rigor `none` disables the methodology validation, so every rule is `off` there. Any other
action for `none` in `severity` is rejected (the catalog fails to load).

## Condition DSL (`when`)

```yaml
{path: a.b.c, <op>: value}     # ops: eq, ne, in, not_in, exists, gt, ge, lt, le, contains
all: [...]                     # all conditions hold
any: [...]                     # at least one holds
not: {...}                     # negation
```

- Paths point into the validated `ExperimentConfig`; integer segments index lists
  (e.g. `paradigm.window.baseline.1`).
- A missing path never satisfies an operator (except `exists: false`).
- Each leaf has exactly one operator; each composite node exactly one of `all` / `any` / `not`.

```yaml
when:
  all:
    - {path: split.backend, eq: general}
    - {path: split.shuffle, eq: true}
```

## Python checks (`check` + `params`)

Use `check` instead of `when` when the logic is too complex for the condition DSL
(branching, counting, combining several config fields).

1. Write a function `(ctx: CheckContext) -> list[Finding]` in a module of
   `src/validation/methodology/checks/` (`static_checks.py`, `preprocessing_checks.py`,
   `training_run_checks.py`; a new module must be imported in `checks/__init__.py`) and register it with
   `@methodology_check("<name>")` (runtime checks: `phase=RulePhase.RUNTIME`).
2. Reference it from a rule with `check: <name>` and optional `params: {...}`.

Experiment-mode checks in `training_run_checks.py` compare the experiment with `<run>/config.json`,
the config snapshot the artifact saver stores next to the loaded model (`<run>/models/<model file>`);
without that snapshot they report nothing.

The function receives:

- `ctx.config`: the validated `ExperimentConfig`,
- `ctx.params`: the rule's `params` (thresholds, mappings, ...), so they can be tuned in YAML without changing code,
- `ctx.data`: runtime data (`None` for static checks).

It returns a list of findings, each with its own `detail` and `location`; an empty list
means the rule is satisfied. When the catalog is loaded, every `check` must be registered
and its phase must match the rule's `phase`, otherwise loading fails.

```yaml
  - id: STAT-001
    title: Too few independent subjects for cross-subject statistics
    certainty: suspected
    check: few_subjects_for_cross_subject
    params:
      min_subjects: 5
```

```python
@methodology_check("few_subjects_for_cross_subject")
def few_subjects_for_cross_subject(ctx: CheckContext) -> list[Finding]:
    min_subjects = int(ctx.params.get("min_subjects", 5))
    ...
    return [Finding(detail="Only 3 subject(s) selected ...", location="source.subject_ids")]
```

## Adding a rule (checklist)

1. Pick the `rules_*.yaml` file of the matching section (or create a new one).
2. Choose a unique `id` with the category prefix.
3. Decide `certainty` using the decision question above.
4. Use `when` if a config condition is enough, otherwise write a Python check.
5. Restrict `pipeline_modes` if the rule makes sense only for some modes.
6. Override `severity` only if the defaults do not fit.
7. Fill in `description`, `impact` and `remediation` so the report is actionable.
8. Run `pytest tests/unit/validation/methodology`; the catalog is loaded and checked there.
