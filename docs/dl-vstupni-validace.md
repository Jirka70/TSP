# Vstupní validace hlavního EEGNet trénování

## Místo volání

`FinalEEGNetTrainer.run()` nejprve zavolá interní `_to_dl_input()` a následně `validate_dl_input()`.
Validace tak proběhne před nastavením seedu, konstrukcí modelu a trénovací smyčkou.
Trainer používá vrácená ověřená pole, nikoli další extrakci původních nahrávek.

Implementace je v `src/impl/model/deep_learning/validation/`:

- `validate_dl_input.py`: pořadí kontrol, společný kontrakt, třídy, report a spojení ověřených nahrávek.
- `input_schemas.py`: Pydantic schémata signálu, původu a jedné nahrávky.
- `recording_validation.py`: načtení NumPy/MNE vstupu a ověření skutečných dostupných metadat.
- `dataset_separation.py`: konzistence identit, oddělení sad a časových intervalů, původ fitu preprocessingu.
- `array_validation.py`: pojmenované operace nad poli a numerické porovnávání.

## Povinná metadata každé nahrávky

`RecordingDTO.metadata` musí obsahovat `signal_specification` a `epoch_origins`.
Při NumPy vstupu obsahuje také `labels`. Při MNE vstupu se používají štítky z událostí;
pokud existují i `labels`, musí se shodovat.

`signal_specification` má následující položky:

| Položka | Význam |
| --- | --- |
| `class_identifiers` | Nejméně dvě různá celočíselná ID tříd, bez požadavku na souvislost nebo začátek nulou. |
| `label_encoding_identifier` | Identita společného kódování tříd; lidské názvy tříd nejsou nutné. |
| `channel_names` | Neprázdné, jedinečné názvy kanálů v pořadí odpovídajícím poli. |
| `sampling_frequency` | Kladná konečná vzorkovací frekvence. |
| `unit` | Jednotka hodnot pole; u MNE signálu je nutné deklarovat jednotku vracených dat. |
| `representation` | Typ reprezentace, například EEG signál. |
| `number_of_time_samples` | Kladný počet časových vzorků v každé epoše. |
| `window_start_seconds` | Konečný začátek okna vůči události. |
| `preprocessing_identifier` | Identita společného postupu předzpracování. |
| `split_protocol` | `trial`, `session` nebo `subject`. Časový protokol napříč měřeními není implementován. |
| `preprocessing_fit_trial_identifiers` | Původní pokusy použité k fitu sdílených parametrů preprocessingu; jinak prázdný seznam. |

Všechny nahrávky a sady musí mít shodnou specifikaci. U MNE se navíc porovnává skutečné pořadí kanálů,
vzorkovací frekvence a začátek okna. U NumPy se porovnají také starší `ch_names` a `sfreq`, pokud jsou dostupné.
Samotné NumPy pole nemůže doložit jednotky nebo názvy kanálů; za pravdivost specifikace odpovídá producent dat.

`epoch_origins` obsahuje pro každou epochu jeden záznam ve stejném pořadí jako signál a štítky:

```python
{
    "augmented": False,
    "roots": [{
        "subject_identifier": "dataset/person-1",
        "session_identifier": "dataset/person-1/session-1",
        "source_identifier": "dataset/person-1/session-1/run-1",
        "trial_identifier": "dataset/person-1/session-1/run-1/trial-42",
        "start_sample": 1000,
        "stop_sample": 1032,
    }],
}
```

Identifikátory musí být jednoznačné napříč zdroji. Opětovný výskyt pokusu, zdroje nebo měření nesmí
odkazovat na jiného vlastníka. Interval je polootevřený `[start_sample, stop_sample)` v souřadnicích
původního zdroje před resamplingem či augmentací. Sousední intervaly se nepřekrývají.
Augmentovaná kopie zachovává původ originálu; kombinace více pokusů uvede všechny kořeny v `roots`.

## Prováděné kontroly

- Neprázdné train/val/test a neprázdné rozměry každé nahrávky.
- Signál `[N, C, T]`, celočíselné štítky `[N]`, shodný počet epoch, štítků a původů před spojováním.
- Reálné konečné hodnoty, také po převodu na `float32`; bez tichého opravování nebo vyhazování epoch.
- Shoda deklarovaného a skutečného počtu kanálů a časových vzorků.
- Deklarované třídy odpovídají konfiguraci; train obsahuje všechny, val/test žádnou neznámou.
- Počty tříd ve všech sadách. Chybějící známá třída ve val/test je upozornění.
- Žádná trénovací augmentace ve val/test; augmentované epochy v train jsou povolené.
- Oddělení původních pokusů ve všech protokolech; navíc celých měření nebo osob podle protokolu.
- Žádné překrývající se části stejného zdroje mezi sadami, ani při rozdílných ID pokusů.
- Sdílené parametry preprocessingu byly podle deklarovaných ID fitované pouze na train.

Validace vrací samostatná pole pro train/val/test, společnou specifikaci, počty tříd a upozornění.
Chyby nahrávek uvádějí sadu, index nahrávky a příčinu bez výpisu signálových polí.
Konkrétní architekturu a vstupní tvar následně ověřuje konstrukce sítě před učením.

## Stav integrace a meze

Současný splitter a augmentace nová metadata ještě nepřenášejí podle tohoto kontraktu.
Hlavní DL trénování proto neúplný vstup odmítne. Validátor metadata nedoplňuje odhadem.
Je nutné doplnit vytváření a zachování původu už v přípravě dat, jeho dělení ve splitteru
a rozšíření při augmentaci. Ostatní modely a foldové trénování nový validátor nepoužívají.

Validátor ověřuje deklarovaný kontrakt, ne pravdivost ručně chybně zadaných údajů.
Příznak augmentace ani seznam fitovacích pokusů nedokazují chování předchozího kódu.
Správnost producentů dat vyžaduje vlastní ověření. Kontrola vstupů sama neověřuje průběh
učení, obnovení nejlepších vah ani export. Závěrečný evaluator pipeline stále používá val;
jeho přepojení na test je samostatný nedokončený úkol.

## Ověření

`.venv/bin/python -m pytest -q` — 96 testů prošlo.
Nové testy zahrnují chybná pole, metadata, třídy, původ a překryvy i skutečný MNE kontejner.
Integrační test ověřuje odmítnutí neplatného testu před vytvořením modelu; samotné učení je mockované.
Ruff ověřuje nové a upravené Python soubory, včetně limitu 120 znaků.
Jednorázová kontrola AST ověřila nejvýše dvě úrovně řídicího odsazení v upravených produkčních funkcích.
