# Návrh metodických pravidel: preprocessing a ML

Stav (2026-11-22, větev `feature/SCRUM-174-Create-rules-preprocessing-ML`): **implementováno**.

- Sekce 0: výběr kanálů opraven pro ML a evaluaci, DL beze změny.
- Sekce 2.1–2.5: všechna pravidla přidána. Checky jsou v `src/validation/methodology/checks/`
  (`training_run_checks.py`, `training_run_config.py`, `preprocessing_checks.py`), testy v
  `tests/unit/validation/methodology/test_preprocessing_and_model_rules.py`.
  `riemannian_rf` má parametr `random_state` (výchozí config `random_state: 42`).
- Sekce 3: LEAK-003 a LEAK-004 omezeny na módy training/benchmark a splity, které dělí epochy jedné nahrávky.
  ONL-002 beze změny. Raw ICA dál nevyřazuje žádnou komponentu, implementace beze změny.
- Otevřené otázky 2–5 vyřešeny implementací v navržené podobě: chybějící `config.json` → checky mlčí,
  soubor `rules_06_preprocessing_and_model.yaml`, prefix `PRE`, EVAL-002 v katalogu pravidel.

Pravidla se řídí `configs/methodology/_README.md`.

Každé pravidlo je ověřené proti aktuálnímu kódu. U tvrzení, která jsem ověřoval simulací, je to uvedeno.
Simulace běžely v projektovém `.venv` (MNE 1.12.1, MOABB 1.5.0) na syntetických datech, ne na skutečném datasetu.

---

## 0. Kritický nález mimo pravidla: do modelu vstupují kanály STI a EOG

Tohle není config pravidlo, ale chyba v kódu, která přebíjí všechno ostatní. Proto je první.

**Co se děje:**

- MOABB vytváří nahrávky BNCI s 22 EEG kanály, 3 EOG kanály a 1 stim kanálem `STI`
  (`moabb/datasets/bnci/base.py`, `_convert_run`: `ch_types = ch_types + ["stim"]`).
  Stim kanál obsahuje v okamžiku cue **kód třídy** (např. 1 = left_hand, 2 = right_hand).
- Nikde v pipeline se nevybírají jen EEG kanály. Trainery i evaluátory volají `epochs.get_data(copy=False)`
  bez `picks`, a MNE pak vrací **všechny** kanály (`_picks_to_idx(info, None, "all", exclude=())`).
  Místa: `generic_sklearn_trainer.py:122`, `final_sklearn_trainer.py:102`, `training_evaluator.py:101`,
  `cross_validation_evaluator.py:142`, `experiment_evaluator.py:134`, `extract_learning_data.py:17`.
- Epocha `[-0.5, 4.0] s` obsahuje okamžik cue, takže kanál `STI` nese label přímo ve vstupu modelu.
  EOG kanály navíc dávají modelu přímo pohyby očí (cue je šipka doleva/doprava).

**Ověření simulací:** pipeline paradigmatu (bandpass 8–35 Hz, epochy s baseline, resampling na 128 Hz)
nad daty, kde EEG je **čistý šum**:

| Vstup modelu `riemannian_lda` (5-fold CV) | Accuracy |
|---|---|
| všechny kanály (EEG + EOG + STI), tj. současný stav | **1.000** |
| jen EEG | 0.367 (náhoda) |

Výsledky ML (a pravděpodobně i DL) nad MOABB daty jsou tedy nyní nedůvěryhodné.

**Stav opravy (2026-11-22): opraveno pro ML a evaluaci, DL beze změny.**

- Helper `get_brain_channel_signal` (`src/impl/model/util/extract/brain_channel_signal.py`) vrací jen kanály
  typu `eeg` a `csd` (po `re_referencing: CSD` mají kanály typ `csd`). Bez takových kanálů vyhodí `ValueError`.
- Používají ho `generic_sklearn_trainer.py`, `final_sklearn_trainer.py` a všechny tři evaluátory.
- DL trénování (`recording_validation.py`) na výslovné zadání beze změny, takže EEGNet se dál učí na všech
  kanálech včetně STI a EOG. Podle kódu (neověřeno během) proto DL nad MOABB daty spadne při evaluaci:
  model je natrénovaný na 26 kanálech, evaluátor mu předá 22. Synthetic data mají jen EEG, tam se nic nemění.
  Oprava DL patří kolegům: použít stejný helper i v DL vstupu.

**Navržené plánované runtime pravidlo** (do `rules_runtime_planned.yaml`, runtime fáze zatím není zapojená):

```yaml
  # - id: LEAK-101
  #   title: Non-EEG channels in the model input
  #   phase: runtime
  #   certainty: proven
  #   check: model_input_contains_only_brain_channels
  #   impact: Stim channels carry the label and EOG channels carry eye movements; the score is not EEG decoding.
  #   remediation: Pick EEG / CSD channels before the model input is built.
```

---

## 1. Přehled navržených pravidel

| ID | Soubor | Certainty | Módy | Typ | Priorita |
|---|---|---|---|---|---|
| LEAK-005 | `rules_04_data_leakage.yaml` | proven | experiment | check | vysoká |
| PRE-001 | `rules_06_preprocessing_and_model.yaml` (nový) | proven | experiment | check | vysoká |
| PRE-002 | `rules_06_preprocessing_and_model.yaml` (nový) | suspected | experiment | check | střední |
| PRE-003 | `rules_06_preprocessing_and_model.yaml` (nový) | proven | všechny | when | střední |
| PRE-004 | `rules_06_preprocessing_and_model.yaml` (nový) | suspected | všechny | when | nízká |
| PRE-005 | `rules_06_preprocessing_and_model.yaml` (nový) | proven | všechny | check | nízká |
| EVAL-002 | `rules_03_training_validation_test_roles.yaml` | proven | benchmark | when | vysoká |
| REP-003 | `rules_09_reproducibility.yaml` | proven | training, benchmark | when | střední |

Plus úpravy existujících pravidel `LEAK-003`, `LEAK-004` a `ONL-002` (sekce 3).

Nový prefix `PRE` (compatibility of preprocessing with the model / with the trained model) je potřeba doplnit
do tabulky prefixů v `_README.md`. Kód prefix nevaliduje. Číslo `06` v názvu souboru jsem zvolil jako volné;
pokud metodický dokument má jinou sekci pro preprocessing, název souboru se přizpůsobí.

---

## 2. Detail pravidel

### 2.1 Experiment mód: model je načtený, ale porovnání s jeho trénováním chybí

V experiment módu se načte hotový model z `model_path` a vyhodnotí se na datech podle aktuálního configu.
Training run ukládá vedle modelu `config.json` s celým `ExperimentConfig`
(`artifacts_saver.py`: model do `<run>/models/`, config do `<run>/config.json`).
Statický check tedy může najít trénovací config jako `Path(model_path.path).parent.parent / "config.json"`.

Tři pravidla níže sdílejí jednu pomocnou funkci `load_training_config(ctx)`. Když `config.json` neexistuje,
checky nic nehlásí (viz otevřená otázka 2).

#### LEAK-005: Experiment evaluates data the model was trained on

**Proč:** když experiment vyhodnocuje stejné nahrávky, ze kterých se model učil, část testovaných epoch
model viděl při trénování. V training módu se z nahrávek vezme train i test část, takže i „stejný subjekt,
stejná session“ znamená překryv s trénovacími epochami.

**Certainty `proven`:** hlásí se jen průnik, který je z configů jistý: stejný `source.name` a průnik
`subject_ids`, `session_ids`, `run_ids`, kde `null` znamená „všechny“. Jiné případy check nehlásí.

```yaml
  - id: LEAK-005
    title: Experiment evaluates data the model was trained on
    certainty: proven
    pipeline_modes: [ experiment ]
    check: experiment_data_overlaps_training_data
    impact: Part of the evaluated epochs were seen during training; the score is optimistically biased.
    remediation: Evaluate on subjects / sessions that were not part of the training run's source.
```

#### PRE-001 a PRE-002: Experiment preprocessing differs from the training preprocessing

**Proč:** model se naučil na datech s určitým předzpracováním. Když experiment data předzpracuje jinak,
nevyhodnocuje se natrénovaný model na jeho skutečném vstupu. Typicky jiné pásmo (jiné kovarianční matice),
jiné okno nebo vzorkovací frekvence (jiný tvar vstupu) nebo jiná reference (lineárně transformované kanály,
naučené prostorové filtry a Riemannův střed přestávají platit).

**Rozdělení podle README:** jeden check `experiment_preprocessing_matches_training` s parametrem `paths`
a dvě pravidla.

- **PRE-001 (`proven`):** pole, která definují vstup modelu. Rozdíl znamená, že model dostává jiný druh vstupu.
- **PRE-002 (`suspected`):** čisticí kroky. Rozdíl může být legitimní (např. jiná nahrávka s jinými bad kanály),
  ale je potřeba ho zdůvodnit.

```yaml
  - id: PRE-001
    title: Model input definition differs from the training run
    certainty: proven
    pipeline_modes: [ experiment ]
    check: experiment_preprocessing_matches_training
    params:
      paths:
        - paradigm.events
        - paradigm.filter
        - paradigm.window
        - paradigm.resampling
        - raw_preprocessing.resampling
        - raw_preprocessing.re_referencing
    impact: The trained model is evaluated on a different kind of input than it was trained on.
    remediation: Use the preprocessing from the training run's config.json for the experiment.

  - id: PRE-002
    title: Signal cleaning differs from the training run
    certainty: suspected
    pipeline_modes: [ experiment ]
    check: experiment_preprocessing_matches_training
    params:
      paths:
        - raw_preprocessing.high_pass_filter
        - raw_preprocessing.low_pass_filter
        - raw_preprocessing.notch_filter
        - raw_preprocessing.bad_channels_interpolation
        - raw_preprocessing.ica
        - epoch_preprocessing.alignment
        - epoch_preprocessing.ica
        - epoch_preprocessing.autoreject
    impact: The evaluated signal can differ from the training signal; the score may not describe the trained model.
    remediation: Use the training cleaning steps, or document why the difference does not change the model input.
```

### 2.2 Riemannovské modely a hodnost kovarianční matice

#### PRE-003 a PRE-004: Rank-deficient covariance with an unregularized estimator

**Proč:** Riemannovské modely (`riemannian_lda/svm/mdm/lr/rf`) počítají logaritmus kovarianční matice.
Když preprocessing sníží hodnost dat, nejmenší vlastní číslo je jen numerický šum. Neregularizovaný odhad
(`scm`, `cov`, `corr`, `mcd`, `hub`, `stu`, `tyl`) pak dá skoro singulární matici, jejíž logaritmus je ovládnutý
šumem. Příznaky v tangentním prostoru nejsou smysluplné, nebo výpočet spadne. Regularizované odhady
(`oas`, `lwf`, `sch`, výchozí je `oas`) tomu brání.

**Ověřeno simulací:** 22 kanálů, po `AVERAGE` hodnost 21/22, po `CSD` také 21/22.

**Rozdělení:**

- **PRE-003 (`proven`):** `re_referencing` `AVERAGE` nebo `CSD` snižuje hodnost vždy.
- **PRE-004 (`suspected`):** epoch ICA (snižuje hodnost jen tehdy, když vyřadí EOG komponenty)
  a interpolace bad kanálů (jen když nahrávka bad kanály má).

```yaml
  - id: PRE-003
    title: Re-referencing makes covariances singular for an unregularized Riemannian model
    description: "AVERAGE / CSD reference removes one rank; the configured covariance estimator does not regularize."
    certainty: proven
    when:
      all:
        - { path: model.model_name, in: [ riemannian_lda, riemannian_svm, riemannian_mdm, riemannian_lr, riemannian_rf ] }
        - { path: model.parameters.estimator, in: [ scm, cov, corr, mcd, hub, stu, tyl ] }
        - { path: raw_preprocessing.re_referencing.enabled, eq: true }
        - { path: raw_preprocessing.re_referencing.method, in: [ AVERAGE, average, CSD, csd ] }
    impact: Tangent-space features are dominated by numerical noise, or the fit fails.
    remediation: Use a shrinkage estimator (model.parameters.estimator=oas or lwf).

  - id: PRE-004
    title: Cleaning can make covariances singular for an unregularized Riemannian model
    description: "Epoch ICA removes components and bad-channel interpolation adds linearly dependent channels."
    certainty: suspected
    when:
      all:
        - { path: model.model_name, in: [ riemannian_lda, riemannian_svm, riemannian_mdm, riemannian_lr, riemannian_rf ] }
        - { path: model.parameters.estimator, in: [ scm, cov, corr, mcd, hub, stu, tyl ] }
        - any:
            - { path: epoch_preprocessing.ica.enabled, eq: true }
            - { path: raw_preprocessing.bad_channels_interpolation.enabled, eq: true }
    impact: Tangent-space features can be dominated by numerical noise.
    remediation: Use a shrinkage estimator (model.parameters.estimator=oas or lwf).
```

`csp_lda` jsem do pravidla nezahrnul: CSP v MNE má výchozí regularizaci `ledoit_wolf` a s hodností zachází samo.

### 2.3 Skutečné pásmo neodpovídá deklarovanému

#### PRE-005: Effective frequency band is narrower than paradigm.filter

**Proč:** `paradigm.filter` (výchozí 8–35 Hz) je pásmo, které se uvádí jako analyzované. Skutečné pásmo
ale může zúžit jiný krok:

- `raw_preprocessing.high_pass_filter.l_freq > paradigm.filter.fmin`,
- `raw_preprocessing.low_pass_filter.h_freq < paradigm.filter.fmax`,
- `raw_preprocessing.resampling.sfreq / 2 <= paradigm.filter.fmax` (resampling odfiltruje vše nad Nyquistem),
- `paradigm.resampling.sfreq / 2 <= paradigm.filter.fmax`.

Například `paradigm.resampling.sfreq: 60` při `fmax: 35` znamená skutečné pásmo 8–30 Hz, ne 8–35 Hz.

**Certainty `proven`:** z configu a chování MNE je zúžení jisté. Dopad je ale hlavně na popis a
reprodukovatelnost experimentu, ne na nezávislost odhadu. Proto navrhuji mírnější severity.

```yaml
  - id: PRE-005
    title: Effective frequency band is narrower than paradigm.filter
    certainty: proven
    check: effective_band_matches_paradigm_filter
    severity: { exploratory: info, standard: warn, publication: error }
    impact: The reported analysis band does not match the signal the model actually sees.
    remediation: Align raw filters / resampling with paradigm.filter, or report the effective band.
```

### 2.4 Benchmark ML bez foldových modelů

#### EVAL-002: Benchmark without cross-validated ML models

**Proč:** v benchmark módu se ML modely trénují jen tehdy, když `model.fold_training: true`
(`generic_sklearn_trainer.py:40`). Jinak trainer vrátí prázdný seznam a `CrossValidationEvaluator`
vrátí prázdné metriky (`cross_validation_evaluator.py:51–53`). Běh tedy „projde“, ale nemá žádný odhad výkonu.
Výchozí `riemannian_lda/lr/mdm/rf` i `csp_lda` mají `fold_training: false`. Je to obdoba `EVAL-001`.

```yaml
  - id: EVAL-002
    title: Benchmark without cross-validated ML models
    description: "mode=benchmark with model.fold_training=false; no fold model is trained, so no score is produced."
    certainty: proven
    pipeline_modes: [ benchmark ]
    when:
      all:
        - { path: model.backend, eq: sklearn }
        - { path: model.fold_training, eq: false }
    impact: The benchmark reports no performance estimate at all.
    remediation: Set model.fold_training=true for benchmark runs.
```

Alternativa: tohle je spíš chyba konfigurace než metodiky a mohla by patřit do schema validace
(`ExperimentConfig`). V katalogu pravidel ji navrhuji kvůli konzistenci s `EVAL-001`.

### 2.5 Reprodukovatelnost ML

#### REP-003: Random forest without a fixed seed

**Proč:** `riemannian_rf` vytváří `RandomForestClassifier(n_estimators=...)` bez `random_state`
(`model_factory.py:96`) a `RiemannianRfParameters` parametr `random_state` ani nepovoluje
(`extra="forbid"`). Opakovaný běh dá jiný model i skóre.

**Předpoklad:** doplnit `random_state: int | None` do `RiemannianRfParameters`, předat ho v `ModelFactory`
a doplnit `random_state: 42` do `configs/model/machine_learning/riemannian_rf.yaml` (stejně jako `riemannian_lr`).
Bez této změny by pravidlo pro RF hlásilo vždy a nešlo by ho opravit configem.

```yaml
  - id: REP-003
    title: Random forest without a fixed seed
    certainty: proven
    pipeline_modes: [ training, benchmark ]
    when:
      all:
        - { path: model.model_name, eq: riemannian_rf }
        - { path: model.parameters.random_state, exists: false }
    impact: Repeated runs give different models and scores.
    remediation: Set model.parameters.random_state.
```

---

## 3. Navržené úpravy existujících pravidel

### LEAK-003 (epoch ICA / AutoReject): omezit na případy, kdy únik opravdu může nastat

ICA i AutoReject se učí **zvlášť pro každou nahrávku** (`epoch_preprocessing.py:63`). Únik vzniká jen tehdy,
když se epochy jedné nahrávky dostanou do trainu i testu. U `moabb_cross_session` a `moabb_cross_subject`
jde každá nahrávka celá do jedné strany, takže pravidlo dnes hlásí falešně. V experiment módu se nic netrénuje.

```yaml
    pipeline_modes: [ training, benchmark ]
    when:
      all:
        - { path: split.backend, in: [ general, moabb_within_session, moabb_within_subject ] }
        - any:
            - { path: epoch_preprocessing.ica.enabled, eq: true }
            - { path: epoch_preprocessing.autoreject.enabled, eq: true }
```

Certainty zůstává `suspected`: krok nepoužívá labely a README ho uvádí jako zdůvodnitelný.

### LEAK-004 a ONL-002 (raw ICA): pravidlo hlídá krok, který nic nedělá

`RawPreprocessor` raw ICA naučí a aplikuje, ale **nevyřadí žádnou komponentu** (`ica.exclude` se nenastavuje).
**Ověřeno simulací:** maximální relativní rozdíl dat před a po raw ICA je `2.7e-15`, tedy data se nezmění.
`LEAK-004` i část `ONL-002` jsou proto dnes falešné poplachy.

Návrh: buď raw ICA dokončit (vyřazení EOG komponent jako u epoch ICA), nebo ji z configu odstranit.
Pravidla se pak upraví podle toho. Do té doby navrhuji pravidla nechat a doplnit k LEAK-004 stejné omezení
na `split.backend` a `pipeline_modes` jako u LEAK-003.

---

## 4. Další nálezy (nejsou to pravidla, ale stojí za opravu)

1. **`annotate_break` + `reject_by_annotation` může zahodit všechny epochy.**
   `annotate_break` má pevně `t_start_after_previous=1.0`. Pokud mají anotace cue nulovou délku, začíná
   `BAD_break` 1 s po cue a epocha do 4 s se zahodí. **Ověřeno simulací:** s nulovou délkou anotací
   `tmax=4.0` → 0/20 epoch, `tmax=0.9` → 20/20. U skutečných MOABB dat záleží na délce anotací, **neověřeno**.
2. **`epoch_preprocessing.alignment` jen přejmenuje časovou osu.** `shift_time(..., relative=True)`
   nemění vzorky ani okno, které vstupuje do modelu. Popis „compensate hardware latency“ je zavádějící.
3. **CSD tiše přeskočené.** Když `compute_current_source_density` selže, zaloguje se warning a nahrávka
   pokračuje bez CSD (`raw_preprocessing.py`). Různé nahrávky pak mohou mít různou referenci.
4. **Paradigm filtruje vstupní raw na místě.** `raw.filter(...)` v `paradigm_preprocessing.py` mění
   `entry.data` předchozí stage, ne kopii. Na výsledky to dnes nemá vliv, ale je to skrytý vedlejší účinek.
5. **`riemannian_svm` používá `SVC(probability=True)` bez `random_state`.** Predikce jsou deterministické,
   ale ukládané pravděpodobnosti (Platt scaling s vnitřní CV) ne.

---

## 5. Zvážené a zamítnuté nápady

| Nápad | Proč ne |
|---|---|
| Okno začíná hned po cue → model může dekódovat VEP / pohyb očí místo motorické představy | Je to standardní praxe (MOABB používá 0–4 s). Pravidlo by hlásilo skoro vždy, takže by bylo jen šumem. Skutečný problém s EOG řeší oprava v sekci 0. |
| Pásmo paradigmatu nepokrývá mu/beta rytmy u motor imagery | `protocol.task` je volný text a horší pásmo znamená horší výkon, ne metodickou chybu. |
| Baseline korekce před kovariančními modely | Kovarianční odhady data stejně centrují, na výsledek to nemá vliv. |
| Riemannovské modely cross-subject bez re-centeringu | Ovlivňuje výkon, ne platnost odhadu. |
| Neregularizované CSP po snížení hodnosti | MNE CSP má výchozí regularizaci a hodnost řeší samo. |

---

## 6. Otevřené otázky ke schválení

1. **Sekce 0:** opravit výběr kanálů hned (týká se i DL, tedy i cizího modulu), nebo jen zapsat a řešit
   s kolegy? Dokud to opraveno není, výsledky na MOABB datech nejsou důvěryhodné a nová pravidla to nezmění.
2. **Experiment pravidla (LEAK-005, PRE-001, PRE-002)** čtou `config.json` trénovacího runu z disku.
   Je to pro statický check v pořádku? A má chybějící `config.json` hlásit samostatné `suspected` pravidlo
   („trénovací preprocessing nelze ověřit“), nebo checky mají jen mlčet?
3. **Název souboru a prefix:** `rules_06_preprocessing_and_model.yaml` a prefix `PRE`. Odpovídá to
   číslování metodického dokumentu?
4. **EVAL-002:** v katalogu pravidel, nebo raději ve schema validaci configu?
5. **REP-003:** souhlasíš s doplněním `random_state` do parametrů `riemannian_rf`?
6. **Úpravy LEAK-003 / LEAK-004 / ONL-002 a raw ICA:** provést spolu s novými pravidly?

## 7. Testy při implementaci

Pro každé nové pravidlo jeden scénář, který pravidlo spustí, a jeden, který ho nespustí, ve stylu
`tests/unit/validation/methodology`. Pro checky s `config.json` vytvořit dočasný trénovací run v `tmp_path`.
Ověřit, že smoke configy (`test_ci_smoke_configs_are_not_blocked`) a scénář F zůstanou bez porušení.
