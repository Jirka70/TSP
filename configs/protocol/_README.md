# Experiment protocol

The protocol declares **which research question the experiment answers** and **how strictly
its methodology is checked**. It does not change what the pipeline does; it is a claim that
the methodology validator (`src/validation/methodology/`, rules in `configs/methodology/`)
compares against the rest of the config. For example, a split that does not match the
declared evaluation scheme blocks the experiment (rule `PROT-001`).

Schema: `ProtocolConfig` in `src/types/dto/config/protocol_config.py`.

## How to select a protocol

`protocol` is a Hydra config group. The default is set in `configs/config.yaml`:

```yaml
defaults:
  - protocol: within_session
```

Override it on the command line or in an experiment config:

```bash
python -m src.main protocol=cross_subject
python -m src.main protocol=cross_subject protocol.rigor=publication   # change a single field
```

```yaml
defaults:
  - override /protocol: cross_session
```

## Fields

| Field | Required | Values | Meaning |
|---|---|---|---|
| `task` | yes | free text | Research question / task definition, e.g. `"4-class motor imagery"` |
| `evaluation_scheme` | yes | see [Evaluation scheme](#evaluation-scheme) | What is **new** in the test data (trials / session / subject / dataset) |
| `usage` | no (default `offline`) | see [Usage](#usage) | For which kind of use the results are claimed |
| `rigor` | no (default `standard`) | see [Rigor](#rigor) | How strictly the methodology rules are enforced (`none` = not at all) |

### Evaluation scheme

#### The idea in one sentence

`evaluation_scheme` says **what the model has never seen** when it is tested, and therefore
**what kind of new data the reported accuracy is valid for**.

#### Why it matters

EEG data are organized hierarchically:

```
dataset                      e.g. BNCI2014_001 (BCI Competition IV-2a)
└── subject                  a person: 9 subjects
    └── session              one recording day (electrodes put on again): 2 sessions per subject
        └── run              a block of the session: 6 runs per session
            └── trial        one imagined movement (~4 s): 48 trials per run -> epochs / windows
```

EEG signals differ between people (anatomy, skill) and between sessions of the same person
(electrode placement, impedance, fatigue). Trials of one session are therefore much more
similar to each other than trials from different sessions or different subjects.
The more of this hierarchy the model has already seen, the easier the test is:

```
easiest                                                                 hardest
within_session  >  within_subject  >  cross_session  >  cross_subject  >  cross_dataset
```

The same model can get e.g. 80 % within-session and 55 % cross-subject. Both numbers are
correct, but they answer **different questions**. Reporting a within-session result as
"the model works for new users" is a methodological error. That is why the scheme is
declared explicitly, and the split must match it.

#### Values

| Value | What is new in the test data | Train contains | Test contains | Answers the question | Compatible split |
|---|---|---|---|---|---|
| `within_session` | new **trials** | other trials of the **same session** | remaining trials of that session | "Does it work later in the same recording?" | `split=evaulation/moabb_within_session`, `split=training/general` |
| `within_subject` | new **trials** | trials of the **same subject**, sessions pooled | remaining trials of that subject | "Does it work for this person (data mixed across days)?" | `split=evaulation/moabb_within_subject`, `.../moabb_within_session`, `split=training/general` |
| `cross_session` | a new **session** | whole sessions of the subject | a **different session** (other day) of the same subject | "Does it work for the same person on another day?" | `split=evaulation/moabb_cross_session` |
| `cross_subject` | a new **subject** | whole data of other subjects | all data of a **subject never seen** in training | "Does it work for a new user without calibration?" | `split=evaulation/moabb_cross_subject` |
| `cross_dataset` | a new **dataset** | one dataset / lab | a different dataset / lab | "Does it work with other hardware / another lab?" | none yet (always reported by `PROT-001`) |

`within_session` vs. `within_subject`: both test on new trials of a known person. In
`within_session` train and test come from one session; in `within_subject` sessions of the
person are pooled, so the test trials may come from a day the model has also trained on.

Example with BNCI2014_001, subject 1, which has two sessions recorded on different days (S1, S2):

- `within_session`: train on 80 % of trials of S1, test on the other 20 % of S1.
- `cross_session`: train on all of S1, test on all of S2.
- `cross_subject`: train on subjects 2-9, test on subject 1 (both sessions).

#### Naming

The values follow MOABB evaluations (`WithinSessionEvaluation`, `CrossSessionEvaluation`,
`CrossSubjectEvaluation`) and the split configs `split=evaulation/moabb_<scheme>`, so the
protocol `cross_subject` goes with the split `moabb_cross_subject`.

#### Rules that use it

- `PROT-001`: the split must match the scheme (table above). The mapping is in
  `params.compatible_splits` of the rule
  (`configs/methodology/rules_02_05_evaluation_protocol_and_sample_provenance.yaml`).
- `PROT-002`: `cross_session` needs at least 2 sessions, `cross_subject` at least 2 subjects in `source`.
- `STAT-001`: `cross_subject` with fewer than 5 subjects gives no reliable uncertainty estimate.
- `GRP-001`: `within_session` with a shuffled `split=training/general` pools sessions and subjects together.

### Usage

For which kind of use the results are claimed.

| Value | Meaning | Consequence |
|---|---|---|
| `offline` | Analysis of recorded data, no real-time claim | No online restrictions |
| `simulated_online` | Recorded data processed as if in real time | Only causal processing allowed: non-causal filters (`ONL-001`) and whole-recording statistics such as ICA / AutoReject (`ONL-002`) are reported |
| `online` | Real-time use with a live device | Not supported by the pipeline (`ONL-003`); use `simulated_online` |

### Rigor

How strictly the methodology rules are enforced. Together with the rule's `certainty` it
determines the action taken when a rule fires (see `configs/methodology/_README.md`):

| Value | Intended for | Rule with `proven` certainty | Rule with `suspected` certainty |
|---|---|---|---|
| `none` | Running the pipeline without any methodology check | off (not evaluated) | off (not evaluated) |
| `exploratory` | Quick experiments, debugging | warn | info |
| `standard` | Default | **error** (blocked) | warn |
| `publication` | Results meant to be reported | **error** (blocked) | **error** (blocked) |

Rules can override these defaults with their own `severity`, except for `none`.

`rigor: none` turns the methodology validation off completely: no rule is evaluated, the
protocol is not compared against the config, and no `methodology_report.json` is written.
Only the config itself is validated (schema of `ExperimentConfig`), then the pipeline runs.
A warning is logged so the run is not mistaken for a validated one. The other protocol
fields are still required, but they have no effect.

```bash
python -m src.main protocol.rigor=none
```

## Prepared protocols

| File | `evaluation_scheme` | `usage` | Question |
|---|---|---|---|
| `within_session.yaml` (default) | `within_session` | `offline` | New trials of an already seen session? |
| `within_subject.yaml` | `within_subject` | `offline` | New trials of an already seen subject? |
| `cross_session.yaml` | `cross_session` | `offline` | A new session of an already seen subject? |
| `cross_subject.yaml` | `cross_subject` | `offline` | A completely unseen subject? |
| `online_simulation.yaml` | `cross_session` | `simulated_online` | Would the model work in real time on a new session? |
| `no_methodology_validation.yaml` | `within_session` (no effect) | `offline` (no effect) | None; methodology validation is skipped |

All of them use `rigor: standard`, except `no_methodology_validation.yaml`, which uses `rigor: none`
(`python -m src.main protocol=no_methodology_validation`).

## Choosing a protocol (checklist)

1. **What should the result be valid for?** Pick `evaluation_scheme`: new trials, a new session (another day) or a new subject (new user).
2. **Choose the matching `split`** from the evaluation scheme table above.
3. **Check `source`**: enough sessions for `cross_session`, enough subjects for `cross_subject`.
4. **Real-time claim?** Use `usage: simulated_online` and causal processing only; otherwise `offline`.
5. **Choose `rigor`**: `exploratory` while experimenting, `standard` by default, `publication` for reported results
   (`none` only when the methodology check should be skipped entirely).
6. Run the experiment; the methodology report is saved next to the outputs. If any
   violation has the action `error`, the experiment does not run.

## Adding a new protocol

Create `configs/protocol/<name>.yaml` with the fields above and a short header comment
stating the question the protocol answers and the correct split. Name the file after the
`evaluation_scheme` it declares (prefix it if another field differs, e.g. `usage`).
