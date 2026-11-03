# Vstupní validace hlavního EEGNet trénování

## Místo volání

`FinalEEGNetTrainer.run()` volá přímo `validate_dl_input(input_dto)` s `FinalTrainingInputDTO`
před seedem, konstrukcí modelu a trénováním. Vstup obsahuje konfiguraci, train a val.
Test zůstává mimo trainer a dostává ho samostatný závěrečný evaluator.
Dočasný `DLInputDTO` ani adaptér už neexistují.

## Aktuální kontroly

- Konfigurace musí být `EEGNetConfig`; train i val musí existovat a obsahovat nahrávky a epochy.
- Signál má v každé nahrávce tvar `[N, C, T]`, všechny rozměry jsou kladné.
- Štítky mají tvar `[N]` a celočíselný typ. Počet štítků odpovídá epochám každé nahrávky zvlášť.
- Signál obsahuje reálná konečná čísla, také po převodu na `float32`.
- Při převodu do tensoru se kontrolují kladné rozměry, shoda kanálů a vzorků s modelem a konečnost hodnot.
  Zapnutá normalizace navíc ověřuje konečnost průměrů, směrodatných odchylek a výsledného signálu.
- Počet kanálů a časových vzorků se shoduje mezi nahrávkami i train/val.
- Skutečný počet tříd v train odpovídá `config.n_classes`.
- Val neobsahuje třídu neznámou v train. Štítky nemusí být souvislé ani začínat nulou.
- Počty příkladů tříd se reportují. Chybějící známá třída ve val se oznámí upozorněním.

NumPy vstup potřebuje štítky v `RecordingDTO.metadata["labels"]`.
U MNE se používají štítky událostí; pokud existují také štítky v metadatech, musí se shodovat.
Při chybě se uvede sada, nahrávka a důvod. Žádné tiché opravování nebo zahazování dat.

## Odstraněné kontroly metadat

Na výslovné zadání uživatele byly odstraněny požadavky na `signal_specification` a `epoch_origins`,
jejich Pydantic schémata i všechny kontroly, které na nich závisely. Odstraněný je také modul
`dataset_separation.py` a nevyužívané helpery. Případné staré položky se ignorují.

DL už podle metadat nekontroluje názvy a pořadí kanálů, jednotky, frekvenci vzorkování, časové okno,
kódování tříd, preprocessing, původ epoch, překryvy sad nebo přítomnost augmentace.
Tyto kontroly nebyly přenesené jinam. Správné přiřazení štítků a rozdělení dat zůstává
odpovědností jejich přípravy a splitteru. Technická validita polí nezaručuje nezávislost sad.

## Výstup a ověření

`ValidatedDeepLearningInput` obsahuje `training`, `validation`, `class_counts` a `warnings`.
Pole pro trainer vznikají spojením již ověřených nahrávek. Popis signálu ani původ se nevrací.
Kompatibilitu konkrétní architektury s tvarem vstupu následně ověřuje konstrukce sítě před učením.

`.venv/bin/python -m pytest -q --tb=short` — 74 testů prošlo.
Testy zahrnují běžná pole se štítky bez zvláštních metadat, MNE epochy, chybná data a předání do traineru.
Ruff ověřuje upravené Python soubory s limitem 120 znaků.

## Kontroly během učení a při predikci

`numerical_validation.py` obsahuje kontroly tensorů, které se neprovádějí ve vstupním validátoru:

- Skóre tříd musí mít tvar `[počet příkladů v batchi, počet tříd]` a konečné hodnoty.
- Loss musí být skalární a konečná, při učení před `backward()` i při validaci.
- Gradienty se kontrolují po `backward()` a před `optimizer.step()`; chybějící gradient se toleruje.
- Parametry sítě se ověřují také po aktualizaci optimizerem, aby neplatné váhy nevytvořily úspěšný výsledek.
- Pravděpodobnosti musejí mít správný tvar, být konečné, v rozsahu `[0, 1]` a mít součty řádků přibližně jedna.
- Predikované štítky mají tvar `[počet vstupních epoch]`.

Neplatné numerické hodnoty vyvolají `FloatingPointError`. Chyba obsahuje fázi, epochu/batch a u gradientů
jméno parametru. Vadná loss ani vadný gradient nevedou k aktualizaci vah. Neplatná epocha se nezapisuje
jako dokončená do historie. Žádné tiché nahrazování NaN nebo automatické přeskakování batchů.

Ověření po doplnění kontrol: celá sada — 107 testů prošlo. Nové testy používají malou skutečnou PyTorch síť,
vkládají neplatnou loss, gradienty, skóre i pravděpodobnosti a ověřují hranici před aktualizací optimizerem.
