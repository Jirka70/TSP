# Refaktor DL a kontrola reprodukovatelnosti

Datum: 2026-11-07. Výchozí pracovní strom obsahoval uživatelské necommitnuté změny nad
`b89cc807773ee1963c0c773cb22c4e8721a368fb`; jejich obsah byl před úpravami zachován.

## Rozsah

Přepis zahrnuje FinalEEGNetTrainer, EEGNetTrainingLoop, EEGNetModel, agregaci metrik,
technickou validaci vstupu, továrnu a konstrukci sítě, seed, serializer, loader a typy checkpointu.

Trainer přijímá stávající FinalTrainingInputDTO s explicitními train/val. Test dál dostává samostatný
evaluator. Nezavádí se foldy, nový adaptér DTO ani odstraněné kontroly specifikace a původu signálu.
Splitter, augmentace, preprocessing, hyperparametry, konfigurace sítě a ML implementace se tímto
refaktorem nemění. Kontrola proti kopii původního pracovního stromu nenašla změnu produkčního Pythonu
mimo DL moduly a jejich funkci pro konstrukci EEGNet sítě.

Veřejné DTO a parametry traineru zůstávají zachované. Během dlouhého ověřování přibylo v jiném chatu
výslovné odstranění `fold_accuracy` a nová hranice exportu verze 2, popsaná v plánu DL.
Tyto souběžné změny se zachovávají: historie nyní zapisuje `validation_accuracy` a aktuální loader odmítá v1.
Porovnání proto explicitně mapuje starý název validační metriky na nový; porovnává přesné číselné hodnoty,
nikoli shodu názvů klíčů. Soubory historie ani původní checkpointy se nepřepisují.

## Implementace

- Trainer skládá ověření vstupu, seed, vytvoření modelu, trénování a výsledné DTO.
- Smyčka nastavuje metadata nejlepší epochy na jednom místě a obnovuje nejstarší nejlepší stav při shodě val accuracy.
- EEGNetModel odděluje výpočet skóre a batch loss od aktualizace parametrů.
- EpochMetrics sdílí výpočet průměrů train/eval. Pořadí součtů, vážení posledního batche a gradienty se zachovávají.
- Loader ověřuje obsah neznámých slovníků a tensorů bez Any či cast. Zachovává původní OrderedDict
  včetně interních metadat verzí modulů PyTorch a používá konkrétní výjimky s příčinou.
- Serializer používá existující Pydantic hranici a dál kontroluje shodu predikcí na celé val po uložení a načtení.
- Volitelná val v přímém API trénovací smyčky zůstává kvůli kompatibilitě. Hlavní trainer ji stále vyžaduje.

## Kontroly kódu

Původní pracovní strom: 133 testů prošlo; cílený mypy hlásil 21 chyb v šesti souborech.
Po přepisu včetně souběžné změny exportu: 154 testů prošlo, cílený mypy a Ruff včetně E501 prošly.
AST kontrola DL souborů a nových testů nenašla více než dvě úrovně řídicího zanoření ani řádky nad 120 znaků.

Nové testy ověřují obnovu nejlepší epochy včetně shody metrik, zachování vah, gradientů a batch normalization
během validace, kompatibilitu přímé smyčky bez val a odmítnutí chybných či poškozených checkpointů.
Existující testy dál pokrývají numeriku, normalizaci, váženou loss, vstup train/val, konfigurace a export.

Příkazy spuštěné v projektovém prostředí:

```sh
.venv/bin/python -m pytest -q --tb=short
.venv/bin/mypy --follow-imports=silent --ignore-missing-imports \
  src/impl/model/deep_learning src/impl/save_artifacts/deep_learning \
  src/impl/model/util/network/create_eegnet_network.py src/types/dto/model/deep_learning \
  tests/unit/test_eegnet_training_loop.py tests/unit/test_eegnet_checkpoint_version.py \
  reviews/dl-refactor-2026-11-07
.venv/bin/ruff check --select E,F,I,E501 \
  src/impl/model/deep_learning src/impl/save_artifacts/deep_learning \
  src/types/dto/model/deep_learning src/impl/model/util/network/create_eegnet_network.py
```

Mypy je cílená kontrola DL, nikoli prohlášení o čistotě celého repozitáře nebo typování závislostí.
Ruff byl spuštěn i nad změněnými testy a reprodukčními skripty; záznamy jsou v adresáři výstupů.

## Reálná data a reference

Použitý vstup je BNCI2014_001, osoby 1–9, stejně jako v posledních bězích 4. listopadu.
Znovu proběhlo předzpracování všech 108 nahrávek po osobách. Jejich pole se uložila do místní cache,
aby všechny nové běhy dostaly stejný vstup bez dalšího předzpracování.

U obou rozdělení se přesně shodují SHA-256 signálů a štítků train, val, test i augmentovaného train
s `outputs/dl-performance-2026-11-04/`. Zachované nastavení je v tehdejším `current_run.yaml`.
Finální kód je zmrazený v `outputs/dl-refactor-2026-11-07/final_source/`, aby další souběžné úpravy
neovlivnily běh. Běžný split se po zavedení v2 opakuje; starší úspěšný přepis v1 je zvlášť v
`refactored-v1/general` a nevstupuje do finálního porovnání.
Každý dokončený trénink používá 15 epoch, seed 67 a CPU; běžný split má 4 PyTorch vlákna a osoba 8 má 2.

Reference pro běžný split zahrnuje nový úplný běh původního pracovního stromu z kopie
`outputs/dl-refactor-2026-11-07/baseline_source/`. Jeho historie, vybraná epocha i všechny testovací metriky
se přesně shodují s 4. listopadem. Pro osobu 8 je referencí přímo původní uložená historie a checkpoint.
Původní i nově natrénované checkpointy se porovnávají po jednotlivých tensorech.
Referenční predikce původních v1 modelů počítá `reference_predictions.py` v odděleném procesu
přes skutečný původní loader z `baseline_source`. Aktuální v2 loader se neobchází a žádný v1
checkpoint se tiše neoznačuje jako v2. Pomocný proces navíc ověřuje hashe testovacích polí.
U osoby 8 po dokončení tréninku původní benchmarkový skript narazil na očekávané odmítnutí v1 loaderem v2;
uložený v2 model a report byly v pořádku. Predikce byly doplněné oddělenými procesy bez dalšího učení.

První souběžné pokusy byly kvůli paměti ukončené před dokončením. Logy mají příponu `-interrupted.log`
a nevstupují do závěrečného porovnání. Dokončené běhy probíhají postupně; jejich časy nejsou měřením rychlosti refaktoru.

## Výsledky

Oba finální běhy v2 se přesně shodují s posledními výsledky ze 4. listopadu:

| Test | Původní accuracy | Nová accuracy | Nové F1 macro | Vybraná epocha |
| --- | --- | --- | --- | --- |
| Běžný GeneralSplitter, 518 příkladů | 87,258687 % | 87,258687 % | 87,221350 % | 14. z 15 |
| Osoba 8, 288 příkladů | 96,875 % | 96,875 % | 96,873153 % | 14. z 15 |

Shodují se vstupní hashe, konfigurace, verze knihoven, číselné historie všech 15 epoch,
vybraná epocha, všechny tensory vybraných sítí, testovací metriky i confusion matrix.
Shodují se také všechny testovací štítky a pravděpodobnosti: maximální absolutní rozdíl je 0.
Historie má záměrně jiné jméno validační metriky a finální exporty jsou v2; tyto rozdíly nejsou
v porovnání skryté (`are_raw_history_keys_equal_to_november_4` je false pro oba běhy).

Strojové porovnání je v `outputs/dl-refactor-2026-11-07/comparison.json`.
Jde o shodu dvou konkrétních běhů se stejnými vstupy, seedem a CPU, nikoli o ověření dalších datasetů či zařízení.

## Reprodukce

Z kořene projektu, postupně:

```sh
.venv/bin/python reviews/dl-refactor-2026-11-07/run_comparison.py baseline general
.venv/bin/python reviews/dl-refactor-2026-11-07/run_comparison.py refactored general
.venv/bin/python reviews/dl-refactor-2026-11-07/run_comparison.py refactored subject-8
.venv/bin/python reviews/dl-refactor-2026-11-07/compare_results.py
```

Skripty vyžadují listopadové artefakty, konfiguraci, původní zdroje v baseline_source a finální zdroje ve final_source.
Každý cílový adresář má obsahovat právě jeden dokončený checkpoint; opakování patří do nového výstupu.
Soubor `source_fingerprints.json` ve výstupech obsahuje hashe původních a nových DL zdrojů.
