# Podklady pro research metodologické správnosti preprocessingu

Popis samotných kroků a parametrů je v [PREPROCESSING_ANALYSIS.md](PREPROCESSING_ANALYSIS.md). Tento dokument shrnuje kontext, který k posouzení metodologie potřebujete navíc.

U tvrzení je uvedeno, odkud plynou:

- **(K)**: přečteno z kódu nebo konfigurace,
- **(T)**: ověřeno testem v MNE na syntetické replice,
- **„ověřit“**: odhad, který je potřeba potvrdit na reálných datech.

## 1. Data, na kterých se preprocessing hodnotí

- **Výchozí dataset** ([configs/source/external.yaml](configs/source/external.yaml)): **BNCI2014_001** (BCI Competition IV 2a) načtený přes MOABB, subjekty 1–3, všechny sessions a runy. (K)
- **Vlastnosti datasetu** (dokumentace v MOABB):
  - 9 subjektů, 22 EEG a 3 EOG kanály (EOG jsou správně typované jako `eog`), 250 Hz, 2 sessions (různé dny) po 6 runech.
  - 4 třídy (levá ruka, pravá ruka, nohy, jazyk). Projekt používá jen `left_hand` a `right_hand`.
  - **Autoři data už při záznamu filtrovali pásmem 0,5–100 Hz a notch filtrem 50 Hz.** Raw HP a notch v projektu jsou tedy na tomto datasetu z velké části nadbytečné.
  - MOABB nastavuje montáž `standard_1005`, takže CSD a interpolace fungují.
- **Jednotka zpracování:** preprocessing se dělá **zvlášť pro každý run** (jeden `RecordingDTO` = jeden run). ICA, AutoReject i CSP se fitují na jednom runu, tj. zhruba 48 trialech. (K)
- **Další loadery:**
  - `filesystem`: EDF a další formáty plus TSV s eventy. Cesta v konfiguraci je natvrdo na stroj jiného člena týmu.
  - `synthetic`: šum s 10Hz/18Hz sinusovkami, amplituda kolem 0,05, tedy **ne ve voltech**.
  - Smoke testy běží na syntetických datech s vypnutým preprocessingem, takže **preprocessing nemá žádné testy**. (K)
- **Verze knihoven:** v `.venv` je nainstalováno `mne 1.12.1`, `autoreject 0.4.3`, `moabb 1.5.0`, `scikit-learn 1.8.0` a `pyriemann 0.10`. Pozor, `requirements.txt` pinuje `mne==1.11.0`, takže nainstalovaná verze s pinem nesouhlasí.

## 2. Celý řetězec, který signál prochází (výchozí konfigurace)

| Krok | Výchozí nastavení |
|---|---|
| Raw | HP 1 Hz → notch 50 Hz → interpolace `bads` (v BNCI nejsou označené žádné, takže krok nic nedělá) → **CSD** → anotace pauz ≥ 2,5 s |
| Paradigm | **pásmová propust 8–35 Hz** → epochy −0,5…4 s kolem cue → baseline (−0,5, 0) → **resampling epoch na 128 Hz** |
| Epoch | **posun časové osy −0,2 s** → **ICA (15 komponent, FastICA) s odstraněním EOG komponent** → AutoReject vypnutý → CSP vypnuté |
| Model | výchozí `riemannian_svm`: Covariances(OAS) → TangentSpace → StandardScaler → SVM. Dále jsou k dispozici `csp_lda`, `riemannian_lda/lr/mdm/rf` a EEGNet s normalizací per epocha a kanál. |

## 3. Nálezy, které přímo ovlivňují validitu výsledků

### 3.1 Do modelu jdou i kanály EOG a stim, a stim kanál nese label (K+T)

- `mne.Epochs` se vytváří bez `picks` a splitter i trainery volají `epochs.get_data()` bez `picks`. Model tak dostane 22 EEG + 3 EOG + 1 stim kanál.
- Na replice se stim kanálem ve stylu BNCI bylo ověřeno, že maximum stim kanálu v epoše je přesně kód třídy (1/2), po resamplingu na 128 Hz pak 0,49 nebo 0,98. **To je přímý únik labelu do příznaků.**
- **Ověřit na reálných datech**, zda MOABB raw pro BNCI2014_001 obsahuje stim kanál s kódy tříd. Stačí porovnat:

  ```python
  epochs.get_channel_types()
  epochs.get_data()[:, -1].max(axis=1)   # vs. epochs.events[:, -1]
  ```

- Kód MOABB (`_convert_run`) stim kanál přidává. Pokud se to potvrdí, je potřeba přeměřit všechny dosavadní výsledky.

### 3.2 CSP v epoch stage (K)

CSP se fituje s labely na celém runu ještě před splitem, tedy s únikem informací z testovacích dat. Model `csp_lda` má navíc vlastní CSP, takže by vzniklo zdvojené CSP.

### 3.3 AutoReject a ICA se fitují před splitem (K)

Fitují se na všech epochách včetně budoucích testovacích. Jsou unsupervised, takže jde o mírnější (transduktivní) únik, v literatuře se ale obvykle zmiňuje.

### 3.4 Finální evaluace není čistá (K)

- **EEGNet:** finální trénování dostává `training_data = výstup epoch preprocessingu`, tedy **všechna data včetně validační sady**. Na té se pak vybírá nejlepší epocha a probíhá i finální evaluace.
- **sklearn:** finální model se trénuje na sjednocení trénovacích dat všech foldů. Při post-split validaci (`exclude_validation_data_before_split: false`) a více než 2 foldech se validační vzorky jednoho foldu objeví v trénovacích datech jiného foldu. Výchozí within-session konfigurace má 2 foldy, kde k tomu nedochází.
- **Dopad pro research:** srovnání variant preprocessingu dělejte na **fold-level metrikách**, ne na finální evaluaci.

### 3.5 Raw augmentace (K)

`raw_torcheeg` vytváří kopie celých runů s novým `session_id` (`…_0raw_augmented`). Kopie stejných trialů se tak dostanou do trénovací i testovací sady a cross-session split je považuje za samostatné sessions. Kód na to sám upozorňuje warningem. Výchozí konfigurace raw augmentaci nemá zapnutou.

### 3.6 Šum v augmentaci je pravděpodobně o několik řádů silnější než signál (K, ověřit)

`gaussian_noise.std` je 0,01 (epochy) a 0,05 (raw) a přičítá se k datům z `get_data()`, která jsou ve voltech (EEG kolem 1e‑5 V). Tuto škálu je potřeba ověřit. Výchozí konfigurace má zapnutou epoch augmentaci `torcheeg`.

## 4. Metodologické otázky k preprocessingu samotnému

### Pořadí a redundance

- Paradigm vždy filtruje 8–35 Hz, takže raw HP 1 Hz i notch 50 Hz jsou pro výsledek nadbytečné. Navíc filtruje přímo vstupní raw objekt, ne kopii.
- Pořadí kroků nejde konfigurovat. U epoch dat je pořadí ICA → AutoReject. Autoři AutoReject doporučují opačný postup: nejdřív AR (nebo AR na nalezení špatných epoch), pak ICA, pak znovu AR.
- Výchozí CSD změní typ kanálů na `csd`, takže AutoReject (vybírá `eeg=True`) se přeskočí. (K+T)

### ICA

- Raw ICA neodstraní žádné komponenty, `infomax` neběží ve variantě *extended* a `random_state` je natvrdo nastavený.
- Epoch ICA:
  - běží na datech po pásmu 8–35 Hz, kde je energie mrknutí (pod 4 Hz) z velké části odstraněná, takže korelace s EOG kanálem je slabá;
  - běží na baseline-korigovaných epochách, přičemž **warning MNE, který na tento problém upozorňuje, je v kódu explicitně potlačený**;
  - 15 komponent na zhruba 48 epoch × 22 kanálů z jednoho runu: je potřeba posoudit, jestli je dat dost.

### Re-reference a montáž

- Volí se mezi CSD a průměrovou referencí. U 22 kanálů je odhad CSD (sférické splajny) hrubý a okrajové elektrody budou zkreslené.

### Časování

- `alignment.shift_time` mění jen časovou osu, ne vybrané vzorky. Kompenzaci latence tedy reálně neprovede a jen posune popisky času, což zkresluje interpretaci a vizualizace.
- Okno −0,5…4 s obsahuje 0,5 s před cue, které jde do klasifikace. Standardní MOABB interval pro BNCI2014_001 je 0…4 s (odpovídá 2–6 s od začátku trialu).
- Baseline korekce na pásmově filtrovaných oscilačních datech je u MI sporná. Neškodí, ale ani nic nepřináší.

### Resampling

- Resampluje se až na epochách. MNE doporučuje resamplovat raw data před epochováním kvůli okrajovým efektům a jitteru eventů.

### Anotace pauz

- `annotate_break` s okraji 1 s v kombinaci s `reject_by_annotation` může vyřazovat i platné triály. Stojí za to zkontrolovat počet zahozených epoch.

## 5. Jak experimenty prakticky spouštět

```bash
python src/main.py mode=training                                   # výchozí trénování (riemannian_svm, within-session, 2 foldy)
python src/main.py mode=training raw_preprocessing.re_referencing.method=AVERAGE epoch_preprocessing.ica.enabled=false
python src/main.py mode=training augmentation=none split=moabb_cross_session model=machine_learning/riemannian_lda
python src/main.py -m mode=training raw_preprocessing.re_referencing.enabled=true,false   # Hydra multirun (neověřeno)
python src/main.py mode=training logging=trace                     # detailní log každého kroku
```

- Výstupy (výsledná konfigurace, logy, metriky, grafy) se ukládají do `outputs/` (Hydra).
- Fold-level přesnosti vypisuje `GenericSklearnTrainer` a agreguje je `MetricsAggregator`.
- Pro čisté srovnání variant preprocessingu doporučujeme:
  - nastavit `augmentation=none` a `raw_augmentation=none`,
  - používat fold-level metriky,
  - vyřešit bod 3.1 (vybrat jen EEG kanály).
- Jako referenci lze použít MOABB `LeftRightImagery` + `WithinSessionEvaluation` na stejném datasetu a pipeline, což jsou publikované baseline hodnoty.

## 6. Klíčové soubory

| Oblast | Soubory |
|---|---|
| Preprocessing | [src/impl/raw_preprocessing/raw_preprocessing.py](src/impl/raw_preprocessing/raw_preprocessing.py), [src/impl/paradigm/paradigm_preprocessing.py](src/impl/paradigm/paradigm_preprocessing.py), [src/impl/epoch_preprocessing/epoch_preprocessing.py](src/impl/epoch_preprocessing/epoch_preprocessing.py) |
| Konfigurace | `configs/{raw_preprocessing,paradigm,epoch_preprocessing}/default.yaml`, validace v `src/types/dto/config/*_config.py` |
| Orchestrace | [src/pipeline/training/training_pipeline.py](src/pipeline/training/training_pipeline.py), [src/pipeline/experiment/experiment_pipeline.py](src/pipeline/experiment/experiment_pipeline.py) |
| Split a evaluace | [src/impl/split/moabb_splitter.py](src/impl/split/moabb_splitter.py), `src/impl/model/machine_learning/{generic,final}_sklearn_trainer.py`, [src/impl/model/deep_learning/eegnet/trainer/final_eegnet_trainer.py](src/impl/model/deep_learning/eegnet/trainer/final_eegnet_trainer.py), [src/impl/evaluator/standard_evaluator.py](src/impl/evaluator/standard_evaluator.py) |
| Modely | [src/impl/model/model_factory.py](src/impl/model/model_factory.py) |
