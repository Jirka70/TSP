# Kontrakt a plán produkčního DL kroku

Datum zápisu: 2026-11-03. Stav: návrh a úkoly, částečná implementace interního převodu popsaná níže.

## Implementovaná vstupní validace (2026-11-03)

Navazující zadání implementuje `validate_dl_input()` hned po interním adaptéru v `FinalEEGNetTrainer.run()`,
před seedem, konstrukcí modelu a učením. Kontroly jsou rozdělené do malých pojmenovaných funkcí.
Pydantic schémata ověřují každou nahrávku a její metadata, další funkce kontrolují společnou specifikaci,
třídy, konzistenci původu, oddělení sad podle protokolu, časové překryvy a deklarované fitování preprocessingu.
Trainer používá vrácená ověřená pole; vypisuje počty tříd a upozornění.

Po upřesnění čitelnosti zůstávají jednoduché cykly a podmínky přímo ve validátoru.
Odstraněné jsou helpery, které pouze obalovaly jednoduché průchody sadami; samostatné kontroly
si vlastní funkce zachovávají. Limit znamená dvě vnořené řídicí struktury (`for → if`, `for → for`),
bez započítání základního odsazení těla funkce. Upřesnění je závazně zapsané v `AGENTS.md`.
Stejná úprava je provedená v celé dosud nepushnuté validaci a v upravené části traineru:
obecné `require_condition()` nahradily přímé podmínky a výjimky, jednoduché průchody jsou viditelné
na místě. NumPy operace a samostatné metodické kontroly si pojmenované helpery zachovávají.
Po této úpravě znovu prošlo všech 96 testů a Ruff s limitem 120 znaků.

Přesný kontrakt a meze popisuje [vstupní validace DL](dl-vstupni-validace.md).
Metadata používají plné názvy `signal_specification` a `epoch_origins`.
Současný splitter a augmentace tento kontrakt ještě nepřenášejí; neúplné vstupy nyní hlavní DL odmítá.
Chybějící metadata se neodhadují a nezávislost sad se nepovažuje za ověřenou bez jejich původu.
Nová validace nemění foldové trénování, ostatní modely ani stávající finální evaluaci na val.
Oddělené testovací vyhodnocení zůstává otevřeným úkolem.

Ověření: `.venv/bin/python -m pytest -q` — 96 testů prošlo; navíc Ruff kontrola upravených Python souborů
a kontrola nejvýše dvou úrovní řídicího odsazení v produkčních funkcích. Testy ověřují i chyby metadat,
překryvy a skutečný MNE vstup, nikoli kvalitu modelu na reálných EEG datech.

## Implementovaný dílčí krok (2026-11-03)

Na následné výslovné zadání uživatele je dočasný převod umístěný přímo v `FinalEEGNetTrainer._to_dl_input` a volaný na začátku `run()`. Pipeline a její veřejný `FinalTrainingInputDTO` se nemění. Toto zadání nahrazuje původní návrh umístit přebalovací adaptér mimo trainer.

- Přidán interní `DLInputDTO` s konfigurací EEGNetu a sadami `train`, `val`, `test` typu `EpochPreprocessedDTO`. Jde o první obal sad, nikoli dokončený kontrakt polí, schématu a původu úseků.
- Převod vyžaduje právě jeden fold a tři neprázdné sady včetně epoch. Přebaluje `fold.train_data`, společnou `validation_data` a `fold.test_data`; více foldů odmítá, nic neslučuje ani nově nedělí. Je určený pro BasicSplitter; samotný počet foldů nedokazuje původ ani nezávislost sad.
- Hlavní EEGNet trainer používá train a val z interního DTO. Původní nerozdělené `FinalTrainingInputDTO.training_data` ignoruje. Ostatní modely ani foldové trénování se nemění.
- Test je zatím pouze součástí interního vstupu. Oddělená závěrečná testovací evaluace, úplný vstupní validátor a kontroly překryvu nejsou tímto krokem implementovány. Stávající evaluace pipeline se nemění.
- Ověření: `.venv/bin/python -m pytest tests/unit/test_final_eegnet_input.py tests/unit/test_stage_factory.py -q` — 42 testů prošlo. Ověřeno odmítnutí nesprávného počtu foldů a chybějících/prázdných sad i předání fold train a společné val do trénovací smyčky místo původních dat. Trénovací smyčka je v tomto ověření nahrazena mockem; nejde o trénink na skutečných EEG datech.

## Uživatelský požadavek

DL má přijímat tři explicitní sady **train, val a test**. Nemá přijímat foldy ani si uvnitř rozdělovat data. Val jsou data, na kterých se model validuje po jednotlivých trénovacích epochách. Uživatel chce začít pracovat na DL, ale současný splitter dodává jiný kontrakt: foldy.

Cílem je metodicky správné učení, důvěryhodné vyhodnocení a model, který lze uložit, načíst a použít se stejným významem vstupů i výstupů. Předchozí diskuse se soustředila zejména na chyby před DL, které ovlivňují jeho vstupy a výsledky.

Požadavek train/val/test a význam val jsou výslovné rozhodnutí uživatele. Konkrétní názvy DTO, rozložení modulů, metrika výběru a další podrobnosti níže jsou návrh k realizaci podle následného zadání. Tento dokument není důkaz, že jsou již implementované.

## Současný nesoulad v kódu

Při zápisu kontextu platí:

- `src/types/dto/model/training_input_dto.py`: `TrainingInputDTO` obsahuje `folds` a je sdílený mezi EEGNet a sklearn modely.
- `src/types/dto/model/final_training_input_dto.py`: `FinalTrainingInputDTO` také obsahuje foldy vedle `training_data` a volitelných `validation_data`.
- `src/impl/model/deep_learning/eegnet/trainer/eegnet_model_trainer.py`: `train_fold` předává `fold.test_data` do trénovací smyčky jako validační data.
- `src/pipeline/training/training_pipeline.py`: nejprve trénuje foldy, potom předává final traineru původní `epoch_preprocessing_result.data`, tedy data před splitem a před epochovou augmentací. Po dílčím kroku výše hlavní EEGNet trainer tuto položku ignoruje a používá train z jediného foldu; pipeline ji nadále předává kvůli stávajícímu rozhraní.
- `src/impl/model/util/extract/extract_learning_data.py`: spojuje nahrávky do polí `x` a `y`; kontroluje celkové délky a tři rozměry `x`. To nenahrazuje úplnou kontrolu vstupního kontraktu.
- `src/types/dto/model/learning_dataset.py`: současný `LearningDataset` obsahuje pouze `x` a `y`.

Před implementací znovu ověř tyto body v aktuálním kódu. Změna DL nesmí bezdůvodně rozbít cestu pro sklearn nebo jiné uživatele foldů.

## Navržené rozhraní

```text
DLInput
  train: Dataset
  val: Dataset
  test: Dataset
  input_spec: popis signálu a tříd
  config: konfigurace modelu a trénování

Dataset
  X: numerické pole [N, C, T]
  y: štítky [N]
  provenance: identita a původ každého úseku
```

`N` je počet EEG úseků, `C` počet kanálů, `T` počet časových vzorků. Každá sada může mít jiný počet úseků. Kanály, jejich pořadí a časový rozměr musí odpovídat jednomu vstupnímu schématu. Pro predikci je tedy počet úseků proměnlivý, nikoli pevně svázaný s počtem při trénování.

Popis vstupu musí určovat názvy a pořadí kanálů, vzorkovací frekvenci, jednotky či typ reprezentace, časové okno vůči události, význam tříd a identitu/verzi předzpracování. Původ úseků má umožnit dohledat člověka, měření, původní pokus a vazbu augmentované kopie na originál. Konkrétní datové typy a místo uložení těchto informací je potřeba navrhnout.

Celý DL krok může přijmout tři sady. Samotná trénovací funkce má dostat jen train a val; test dostane až závěrečný evaluator. Tím se omezuje možnost použít test omylem během výběru modelu.

## Rozdělení odpovědností

| Část | Odpovědnost |
| --- | --- |
| Příprava dat a splitter | Vytvořit tři sady podle domluveného protokolu, zachovat štítky, popis signálu a původ dat. |
| Augmentace | Upravovat pouze train a zachovat vazbu kopií na originály. |
| Případný integrační převod | Mimo DL převést existující výstup na nový kontrakt, pouze pokud je význam všech tří sad jednoznačný. |
| Vstupní validátor DL | Ověřit kontrakt a odmítnout chybný nebo neúplný vstup. |
| Trainer | Učit na train, validovat po epochách na val, obnovit vybraný nejlepší stav. |
| Evaluator | Vyhodnotit vybraný model na test bez dalšího učení či výběru nastavení. |
| Export a loader | Uchovat a ověřit model i informace nutné pro shodné použití. |

Z foldu obsahujícího jen train a test nelze přejmenováním získat tři nezávislé sady. Převod nesmí potichu vybrat první fold, slít překrývající se foldy nebo použít stejná data jako val i test. Pokud třetí sada chybí, musí být explicitně stanoven postup jejího vytvoření mimo DL.

## Kontroly před trénováním

- Všechny tři sady jsou pro tento trénovací protokol přítomné a neprázdné.
- `X` má tvar `[N, C, T]`, `y` má tvar `[N]`; při spojování nahrávek se délky kontrolují i pro každou nahrávku zvlášť.
- Data jsou numerická, konečná a po převodu na podporovaný typ zůstávají konečná. Konfigurace sítě podporuje dané `C` a `T`.
- Skutečné údaje všech sad odpovídají vstupnímu schématu. Pouhé připojení společného popisu nesmí zakrýt rozdíly mezi zdroji.
- Mapování tříd má všude stejný význam; train obsahuje očekávané třídy, val a test neobsahují neznámé štítky. Zastoupení tříd se reportuje.
- Sady dodržují pravidla oddělení původních pokusů a případně osob, měření nebo času. Různá ID kopií nesmějí zakrýt stejný původní pokus.
- Podezřelá amplituda, konstantní kanály a další známky špatného signálu se posuzují podle deklarované reprezentace. Univerzální limit pro všechny jednotky není vhodný.

Zásadní porušení kontraktu má zastavit běh s konkrétní chybou. Nevyrábět chybějící štítky, tiše nevyhazovat problematické úseky a nepokračovat s NaN. Validátor nemůže dokázat správnost ručně chybně označené události; za její význam odpovídá zdroj a příprava dat.

## Metodika a průběh trénování

1. Ověřit vstupy a zaznamenat konfiguraci, identitu dat, rozdělení, seed a prostředí.
2. Vytvořit nový model a jednotné mapování tříd.
3. Provést trénovací epochu na train.
4. Vyhodnotit val v eval režimu, bez gradientů a bez aktualizace trénovacího stavu.
5. Podle předem zvolené metriky uložit nejlepší stav; případně použít předčasné zastavení.
6. Obnovit nejlepší stav podle val.
7. Závěrečně vyhodnotit tento stav na test.
8. Exportovat a ověřit právě tento model.

Test nesmí ovlivňovat výběr epochy, hyperparametrů, rozhodovacích prahů ani seedů. Opakované ladění podle testu ruší jeho roli nezávislé závěrečné zkoušky. Po testu model automaticky nepřetrénovávat na všech datech; tím by vznikl jiný, tímto testem neověřený model.

Split musí odpovídat zamýšlenému použití: noví lidé, nová měření stejného člověka nebo budoucí čas. Náhodné rozdělení úseků samo o sobě nedokazuje zobecnění na nové lidi. Překrývající se okna a kopie jednoho pokusu musí respektovat hranice sad.

Parametry předzpracování učené ze souboru dat se standardně odhadují pouze na train a aplikují na další sady. Případná adaptace na nové měření musí být výslovně součástí protokolu a odpovídat produkčnímu použití. Ne každá pevná filtrace automaticky představuje únik informací; rozhoduje konkrétní operace a dostupnost dat při použití.

## Výstup a serializace

Výstupem je vybraný model, nejlepší epocha a metrika, historie train/val, testovací report, identifikace běhu a jednoznačný stav dokončení. Report má obsahovat počty příkladů a výsledky podle tříd; přijatelné metriky a minimální výkon je nutné určit podle účelu. Dokončený trénink sám neznamená schválení k nasazení.

Predikce pro `N` úseků mají vracet štítky `[N]` a pravděpodobnosti `[N, K]` s jednoznačným pořadím `K` tříd. Ověřovat konečnost, rozsah pravděpodobností a součet každého řádku v toleranci.

Balíček pro nasazení má obsahovat váhy, architekturu a její konfiguraci, mapování tříd, vstupní schéma, popis i naučený stav preprocessingu, verzi formátu a informace o kompatibilitě. Preprocessing může být samostatný artefakt, ale musí být jednoznačně navázaný na model.

Checkpoint pro pokračování trénování je jiný účel: potřebuje navíc konzistentní stav optimizeru, případného scheduleru, náhodných generátorů a průběhu trénování. Nemíchat nejlepší váhy jedné epochy se stavem optimizeru z jiné epochy a netvrdit, že jde o přesné pokračování.

Ukládat nejprve do dočasného umístění, ověřit úplnost balíčku a načtení v novém procesu, porovnat predikce před a po uložení v toleranci a teprve potom označit výsledek za dokončený. Přerušené ukládání ani neplatné váhy nesmějí vytvořit zdánlivě úspěšný výsledek. Zařízení a limity prostředků musí být explicitní; chyba musí být rozpoznatelná i návratovým stavem procesu.

## Doporučené pořadí implementace

- [x] Definovat interní DTO, popis signálu a původ úseků; vymezit hranici DL bez změny kontraktu ostatních modelů.
- [x] Implementovat vstupní kontroly a malé syntetické train/val/test fixture. Kontrakt metadat a meze jsou popsané výše.
- [ ] Napojit jeden trainer na train/val, vybrat a obnovit nejlepší stav, přidat oddělenou testovací evaluaci.
- [ ] Domluvit se splitterem sémantiku a vytvoření tří sad. Případný převod realizovat mimo DL s explicitními pravidly, nikoli jako odhad uvnitř traineru.
- [ ] Upravit DL větev pipeline: odstranit závislost na foldech a automatické finální trénování na nerozdělených datech. Zachovat ostatní cesty projektu.
- [ ] Vyřešit správné předání train augmentace a konzistenci preprocessingu, kanálů, jednotek, časových oken a štítků.
- [ ] Dokončit výstupní kontrakt, export, loader a ověření po načtení.
- [ ] Provést integrační ověření na skutečných datech a vyhodnotit vhodnost k nasazení podle předem určených kritérií.

## Co musí prokázat ověření

- Chybné rozměry, štítky, neplatná čísla a nesoulad schématu se odmítnou ještě před trénováním.
- Překryv původních pokusů mezi sadami je odhalen podle jejich původu.
- Změna testovacích dat neovlivní trénování ani výběr nejlepší epochy.
- Validace neaktualizuje váhy ani trénovací stav; val se kontroluje po každé trénovací epoše.
- Test i export používají obnovený nejlepší stav, nikoli poslední epochu nebo jiný přetrénovaný model.
- Augmentace nezasahuje val/test; správná train data skutečně dorazí do traineru.
- Predikce mají správný tvar a význam tříd, také po serializaci a načtení v novém procesu.
- Nekompatibilní či poškozený checkpoint a selhání trénování či ukládání nevedou k hlášení úspěchu.
- Existující relevantní testy ostatních cest zůstávají funkční. Úspěch na syntetických datech ověřuje implementaci, nikoli kvalitu EEG klasifikace.

## Podklady z předchozího rozboru

- `reviews/deep-learning-2026-10-20/rozbor.md`: rozbor DL včetně serializace a diagnostických skriptů ve stejném adresáři.
- `reviews/upstream-2026-10-29/rozbor.md`: předchozí kroky a jejich vliv na DL, včetně sond a výstupů.

Pro tento úkol mají přednost zejména nálezy o nejednotném mapování událostí na štítky, časovém posunu při raw augmentaci, intenzitě šumu vůči jednotkám EEG, úniku dat mezi sadami a předání nerozdělených dat final traineru. Jejich platnost a konkrétní podmínky je potřeba ověřit před opravou. Chyba pouze v metadatech exportu nemusí měnit pole, které dostává DL v aktuálním běhu; neoznačovat automaticky každý upstream nález za přímé poškození učení.

## Rozhodnutí, která zbývá konkretizovat

- Pro jaké použití se dělí data: noví lidé, nová měření nebo budoucí čas?
- Kdo a jak dodá tři sady ze současného výstupu splitteru?
- Jaký bude přesný typ metadat a identifikátorů původu?
- Podle které val metriky se vybírá model a jak funguje předčasné zastavení?
- Jaká jsou kritéria přijatelného výkonu, podporovaná zařízení a požadavek na pokračování přerušeného tréninku?

Tyto otázky neblokují vytvoření cílového kontraktu, validátoru a syntetických fixture. Je potřeba je uzavřít před závislou integrací nebo tvrzením, že model splňuje podmínky nasazení.
