# Analýza preprocessingu (raw a epoch)

Program obsahuje dvě vlastní preprocessing stage:

- `RawPreprocessor` ([src/impl/raw_preprocessing/raw_preprocessing.py](src/impl/raw_preprocessing/raw_preprocessing.py)) pracuje se spojitým signálem.
- `EpochPreprocessor` ([src/impl/epoch_preprocessing/epoch_preprocessing.py](src/impl/epoch_preprocessing/epoch_preprocessing.py)) pracuje s epochami.

Mezi ně je vložená stage **paradigm** ([src/impl/paradigm/paradigm_preprocessing.py](src/impl/paradigm/paradigm_preprocessing.py)). Ta signál také upravuje (filtruje, epochuje, dělá baseline a resampling), proto je v analýze popsaná taky.

## 1. Kde se preprocessing v pipeline spouští

Obě pipeline mají stejné pořadí kroků:

```
load → RAW PREPROCESSING → (raw augmentation, jen training) → PARADIGM → EPOCH PREPROCESSING → split/… nebo model/eval
```

- **`mode=training`** ([src/pipeline/training/training_pipeline.py](src/pipeline/training/training_pipeline.py)): kompletní řetězec včetně raw augmentace mezi raw preprocessingem a paradigmem.
- **`mode=experiment`** ([src/pipeline/experiment/experiment_pipeline.py](src/pipeline/experiment/experiment_pipeline.py)): stejné kroky bez augmentace, potom se načte uložený model a vyhodnotí se.
- Po raw a po epoch preprocessingu se volá vizualizér (`visualize_raw`, `visualize_epochs`).

**Výběr implementace:** v `StageFactory` ([src/pipeline/stage_factory.py](src/pipeline/stage_factory.py)) je pro obě stage jen `backend: default`. Jiná hodnota konfigurací neprojde, protože typ je `Literal["default"]`.

## 2. Jak se preprocessing nastavuje a spouští

Konfiguraci řeší Hydra ([src/main.py](src/main.py)) a validuje Pydantic:

| YAML | Pydantic model |
|---|---|
| [configs/raw_preprocessing/default.yaml](configs/raw_preprocessing/default.yaml) | `RawPreprocessingConfig` ([src/types/dto/config/raw_preprocessing_config.py](src/types/dto/config/raw_preprocessing_config.py)) |
| [configs/epoch_preprocessing/default.yaml](configs/epoch_preprocessing/default.yaml) | `EpochPreprocessingConfig` ([src/types/dto/config/epoch_preprocessing_config.py](src/types/dto/config/epoch_preprocessing_config.py)) |
| `configs/paradigm/default.yaml` | `ParadigmConfig` ([src/types/dto/config/paradigm_config.py](src/types/dto/config/paradigm_config.py)) |

Možnosti spuštění:

```bash
python src/main.py                                    # výchozí config.yaml (mode: experiment)
python src/main.py mode=training
python src/main.py raw_preprocessing.ica.enabled=true raw_preprocessing.re_referencing.method=AVERAGE
python src/main.py raw_preprocessing.notch_filter.freqs=[50,100]
python src/main.py epoch_preprocessing.csp.enabled=true epoch_preprocessing.csp.reg=ledoit_wolf
python src/main.py raw_preprocessing=muj_preset     # vlastní configs/raw_preprocessing/muj_preset.yaml
python src/main.py logging=trace                    # zobrazí logy jednotlivých kroků (úroveň TRACE)
```

Každý blok se zapíná a vypíná přes `enabled`. **Pořadí kroků je ale napevno v kódu a konfigurací se změnit nedá.** Ukázky s částečným přepsáním konfigurace jsou v `configs/test/smoke/synthetic/*.yaml`: všechny preprocessing kroky tam vypínají a ostatní hodnoty se dotáhnou z `default.yaml`.

---

## 3. Raw preprocessing (`RawPreprocessor`, signál `mne.io.Raw`)

Každá nahrávka se nejdřív zkopíruje (`entry.data.copy()`) a pak projde těmito kroky v pevném pořadí:

| # | Blok (YAML) | Parametry a validace | Co se reálně volá | Default YAML |
|---|---|---|---|---|
| 1 | `resampling` | `sfreq > 0` | `raw.resample(sfreq)` | vypnuto, 250 Hz |
| 2 | `bad_channels_interpolation` | jen `enabled` | `raw.interpolate_bads(reset_bads=True)`, pokud `info["bads"]` není prázdné | zapnuto |
| 3 | `high_pass_filter` + `low_pass_filter` | `l_freq ≥ 0`, `h_freq > 0`, a pokud jsou zapnuté oba, musí platit `l_freq < h_freq` (model validator) | jedno volání `raw.filter(l_freq, h_freq, fir_design="firwin", skip_by_annotation="edge")`. Vypnutý filtr se předá jako `None`, takže vznikne čistý HP, čistý LP nebo pásmová propust | HP 1 Hz zapnuto, LP 40 Hz vypnuto |
| 4 | `notch_filter` | `freqs`: seznam hodnot > 0 | `raw.notch_filter(freqs, fir_design="firwin")` | zapnuto, `[50]` |
| 5 | `ica` | `n_components`: int, nebo float v intervalu (0, 1) jako podíl vysvětlené variance; `method`: `infomax` / `fastica` / `picard` | `ICA(n_components, method, random_state=42).fit(raw)`, potom `ica.apply(raw)` | vypnuto, 0.95, infomax |
| 6 | `re_referencing` | `method`: `CSD` / `AVERAGE` (zadání není citlivé na velikost písmen) | CSD: `compute_current_source_density(raw)`. AVERAGE: `set_eeg_reference("average")` | zapnuto, CSD |
| 7 | `annotate_break` | `min_break_duration > 0` | `annotate_break(raw, min_break_duration, t_start_after_previous=1.0, t_stop_before_next=1.0)`, výsledek se přidá k existujícím anotacím | zapnuto, 2.5 s |

### Poznámky k raw krokům (vyčteno z kódu)

- **Interpolace neumí špatné kanály sama najít.** Interpoluje jen kanály, které už jsou v `info["bads"]` (označil je loader nebo dataset). Komentář v kódu „Identification“ tedy neodpovídá skutečnosti. Interpolace sférickými splajny navíc potřebuje montáž.
- **Raw ICA nic neodstraní.** Po `fit` se nenastaví `ica.exclude` (na rozdíl od epoch ICA), takže `apply` signál prakticky zrekonstruuje beze změny. Krok jen spotřebuje výpočetní čas. Navíc `random_state=42` je napevno a `infomax` se nespouští jako *extended* (chybí `fit_params=dict(extended=True)`), i když to komentář v YAML tvrdí.
- **CSD selže potichu.** Když chybí montáž nebo nastane jiná `RuntimeError`/`ValueError`, vypíše se jen warning a re-referencing se přeskočí. Pokud CSD proběhne, typ kanálů se změní z `eeg` na `csd`, což ovlivní epoch AutoReject (viz níže).
- **Break anotace** dostanou popisek `BAD_break`. Paradigm má `reject_by_annotation=True`, takže epochy, které do pauzy zasahují, se zahodí. Okraje 1.0 s před a po události jsou napevno v kódu.
- **Pydantic:** `sfreq`, `l_freq`, `h_freq`, `freqs` a `min_break_duration` nemají default. Musí být v YAML vyplněné, i když je blok vypnutý (`default.yaml` je obsahuje).

---

## 4. Paradigm (mezikrok Raw → Epochs, `ParadigmPreprocessor`)

Formálně to není „preprocessing“ stage, ale signál výrazně mění:

| Část | Parametry | Chování |
|---|---|---|
| `filter` | `fmin=8.0`, `fmax=35.0`; musí platit `fmin < fmax` | **Pásmová propust se aplikuje vždy a nejde vypnout.** Volá se `raw.filter(...)` přímo na vstupním objektu, bez kopie. |
| `events` | dict nebo list názvů a ID | Názvy se normalizují (malá písmena, `_` místo mezer). Nahrávky bez odpovídajících eventů se přeskočí. |
| `window` | `tmin=-0.5`, `tmax=4.0`, `baseline=(-0.5, 0.0)` nebo `null` | Vytvoří `mne.Epochs` s baseline korekcí. Validuje se, že baseline leží uvnitř okna. |
| `resampling` | `enabled=false`, `sfreq=128` | `epochs.resample()` |
| `reject_by_annotation`, `preload` | `true`, `true` | Vyřadí epochy překrývající se s `BAD_*` anotacemi. |

Kvůli povinnému pásmu 8–35 Hz je raw HP/LP filtr do velké míry nadbytečný. Zároveň to má dopad na epoch ICA: mrknutí jsou převážně pod 4 Hz, takže v epochách je už z velké části odfiltrujete.

---

## 5. Epoch preprocessing (`EpochPreprocessor`, signál `mne.Epochs`)

Prázdné záznamy (0 epoch) se přeskočí s warningem. Každá nahrávka se zkopíruje a projde těmito kroky v pevném pořadí:

| # | Blok | Parametry a validace | Co se reálně volá | Default YAML |
|---|---|---|---|---|
| 1 | `alignment` | `tmin_offset ∈ [-10, 10]` s | `epochs.shift_time(tmin_offset, relative=True)`. Posune jen časovou osu, data se nemění. | zapnuto, −0.2 s |
| 2 | `ica` | `n_components`: **jen int** v rozsahu 1–100; `random_state`: int; `method`: `fastica` / `infomax` / `picard`; `eog_threshold > 0` (z-skóre) | `ICA(...).fit(epochs)`, pak `find_bads_eog(epochs, threshold)`, nastavení `ica.exclude` a `ica.apply(epochs)`. Warning „baseline-corrected“ se potlačí. | zapnuto, 15, 42, fastica, 3.0 |
| 3 | `autoreject` | `n_interpolate`: neprázdný seznam intů ≥ 0; `consensus`: neprázdný seznam hodnot v ⟨0, 1⟩; `cv ≥ 2` | `AutoReject(n_interpolate, consensus, cv, random_state=ica.random_state, picks=EEG bez bads).fit_transform(epochs)`. Mřížkové hledání lokálních prahů, oprava interpolací a zahazování epoch. | vypnuto, `[1,4,8]`, `[0.2,0.5,0.8]`, 5 |
| 4 | `csp` | `n_components > 0`; `reg`: `None`, float nebo string (např. `ledoit_wolf`, `oas`); `log`, `norm_trace`: bool | `CSP(...).fit_transform(epochs.get_data(), labels=events[:, -1])`. **Výstup je `ndarray` příznaků** `(n_epochs, n_components)` místo `Epochs`. | vypnuto, 4, null, true, false |

Při jakékoli výjimce se zaloguje index nahrávky a výjimka se pošle dál, takže pipeline skončí.

### Poznámky k epoch krokům (vyčteno z kódu)

- **Epoch ICA potřebuje EOG kanál.** `find_bads_eog` bez EOG kanálu vyhodí `RuntimeError` a celá pipeline spadne. ICA je přitom v `default.yaml` zapnutá. Na datech bez EOG ji musíte vypnout.
- **`n_components` u epoch ICA přijme jen int.** Komentář v YAML zmiňuje „0.95 variance“, ale validace float odmítne. Hodnota také nesmí překročit počet kanálů.
- **AutoReject bere `random_state` z bloku `ica`**, takže ICA config musí být validní i při vypnuté ICA. Pokud raw stage spočítala CSD, kanály jsou typu `csd`, `pick_types(eeg=True)` nic nenajde a AutoReject se s warningem přeskočí. Výchozí kombinace (CSD zapnuté) tedy AutoReject fakticky vyřadí. Interpolace navíc vyžaduje montáž a `cv` nesmí být větší než počet epoch.
- **CSP:**
  - **Únik informací (data leakage):** CSP se fituje na celé nahrávce s labely ještě před splitem, takže filtry „vidí“ i testovací data. Výsledky evaluace pak budou optimisticky zkreslené.
  - **Kompatibilita dál v pipeline:** splitter ([src/impl/split/moabb_splitter.py](src/impl/split/moabb_splitter.py), obdobně `basic_splitter`) u `ndarray` vstupu hledá labely v `metadata["labels"]`. Když tam nejsou, vyhodí `ValueError`. Modely a vizualizace, které čekají `Epochs` nebo 3D pole, s 2D příznaky také nemusí fungovat.
- **Alignment** jen přečísluje časy. Baseline už aplikoval paradigm, takže na hodnoty signálu nemá vliv.
- Všechny epoch configy kromě `tmin_offset` a `reg` nemají defaulty, takže `default.yaml` musí být kompletní.

---

## 6. Souhrn všech dostupných postupů

**Raw (kontinuální signál):** resampling → interpolace předem označených špatných kanálů → HP / LP / pásmový FIR filtr → notch (víc frekvencí) → ICA (bez odstranění komponent) → re-reference CSD nebo průměrová reference → automatická anotace pauz.

**Paradigm:** povinný pásmový filtr 8–35 Hz (nastavitelný) → výběr eventů → epochování s baseline → volitelný resampling epoch → vyřazení epoch podle `BAD_*` anotací.

**Epoch:** posun časové osy → ICA s automatickým odstraněním EOG komponent → AutoReject (lokální prahy, interpolace, zahazování) → CSP transformace na příznaky.

**Hlavní problémy, které stojí za zmínku v analýze:**

1. Raw ICA nic neodstraňuje.
2. Interpolace kanály sama nedetekuje.
3. Výchozí CSD vyřadí AutoReject.
4. Výchozí epoch ICA bez EOG kanálu shodí pipeline.
5. CSP způsobuje data leakage a nekompatibilní výstup.
6. Pásmový filtr v paradigmu nejde vypnout.
7. Pořadí kroků nejde konfigurovat.
