# Kontext pro agenty v projektu TSP

## Deep learning a kontrakt vstupních dat

Před prací na DL, rozdělení dat, augmentaci, evaluaci nebo serializaci modelu si přečti [plán DL kroku](docs/dl-kontrakt-a-plan.md). Obsahuje uživatelské požadavky, stav integrace, návrh rozhraní a pořadí práce.

- Po merge splitteru uživatel upřesnil hranici: hlavní DL trainer přijímá pouze explicitní **train a val**, nikoli foldy ani test. **Test** dostává samostatný závěrečný evaluator.
- **Train** slouží k učení. **Val** slouží k validaci po každé trénovací epoše a výběru modelu. **Test** slouží k závěrečnému nezávislému vyhodnocení.
- `GeneralSplitter` vrací přímo train/validation/test. MOABB splittery dál vracejí foldy pro samostatnou evaluační pipeline; nezaváděj je do hlavního DL traineru.
- Nepřejmenovávej test na val, neslévej foldy a nevybírej potichu první fold jako řešení tohoto nesouladu. Případný převod patří mimo DL a musí mít explicitní pravidla pro tři oddělené sady.
- `FinalEEGNetTrainer` používá přímo `FinalTrainingInputDTO`. `validate_dl_input()` ověřuje train/val a vrací `ValidatedDeepLearningInput`. Dočasný `DLInputDTO` a adaptér byly odstraněny; neobnovuj je.
- Foldové trénování bylo na výslovné zadání odstraněno pouze z EEGNetu. EEGNet nemá `EEGNetModelTrainer`
  ani konfigurační přepínač `fold_training`; používá `FinalEEGNetTrainer`. ML implementace a jejich foldy zachovej.
- Na následné zadání byly odstraněny kontroly specifikace signálu a původu epoch. DL nevyžaduje
  `signal_specification` ani `epoch_origins` a neověřuje podle nich překryvy, augmentaci nebo fit preprocessingu.
  Zůstávají technické kontroly polí a tříd. Správné rozdělení dat zajišťuje příprava dat a splitter.
- Prioritou jsou chyby, které ovlivňují skutečný vstup DL, jeho učení nebo důvěryhodnost výsledků. Rozlišuj potvrzenou chybu od rizika závislého na konfiguraci.
- Zachovej funkčnost ostatních modelů a použití splitteru. Změna požadavku na DL sama o sobě neznamená odstranit foldy z celého projektu.

## Stav a způsob pokračování

K 2026-11-03 byl zapsán kontext a plán; tento zápis neimplementuje nové rozhraní ani opravy. Poslední požadavek uživatele byl vytvořit dokumentaci pro další vlákna. Tento plán sám není pokyn automaticky provést celý refaktor: pokračuj podle aktuálního zadání uživatele. Při následném zadání implementace není potřeba znovu schvalovat již výslovně dohodnutý směr oddělených train/val/test; hranici traineru upřesňuje aktuální pravidlo výše.

Před změnami ověř aktuální kód a stav pracovního stromu. Starší rozbory v `reviews/` jsou podklady k ověření, ne záruka, že nález stále platí. Po implementaci udržuj stav úkolů v plánu aktuální a uveď provedené ověření. S uživatelem komunikuj česky a vysvětluj konkrétně a jednoduše.

## Závazné konvence psaní Pythonu

Platí pro nový a upravovaný ručně psaný kód, včetně testů. Nesouvisející starší kód kvůli nim plošně nepřepisuj.
Výslovná pravidla níže mají přednost před odlišným stylem staršího kódu. Zachovej existující veřejné kontrakty.

### Návrh před implementací

- Nejdřív uvažuj o systému jako o černé skříňce: účel, veřejné operace, vstupy, výstupy a chybové stavy.
- U větší změny stručně ukaž smysluplné API a příklad volání, potom pokračuj implementací.
  U malé změny to stačí promyslet; nevytvářej zbytečný návrhový dokument ani schvalovací krok.
- Teprve potom rozhodni o vnitřním rozdělení, datových strukturách a závislostech.
- KISS, DRY a SOLID jsou závazné. Vol nejjednodušší řešení skutečných požadavků.
- Každá funkce, třída a modul má jednu soudržnou odpovědnost.
- Sdílej stejnou znalost a pravidla. Nesjednocuj kód jen kvůli podobnému zápisu, pokud má jiný důvod ke změně.
- Preferuj kompozici a malé kontrakty. SOLID neznamená třídu, abstraktní bázi nebo továrnu pro každou funkci.
- Nevytvářej generické frameworky, vrstvy ani rozhraní pro hypotetické budoucí použití.

### Čitelnost a rozdělení funkcí

- Kód má plynout jako souvislý text. Hlavní funkce skládá smysluplně pojmenované kroky v přirozeném pořadí.
- Název jednoznačně říká, co funkce dělá, a nezamlčuje významný vedlejší účinek.
- Nespojuj samostatné odpovědnosti do názvů jako `validate_and_save_model` nebo `load_and_train`.
  Souhrnná operace může mít jeden věcný název, například `train_model`, pokud odpovídá jejímu účelu.
- Extrahuj helper pro samostatnou odpovědnost, složitější výpočet, doménové pravidlo nebo nejasnou knihovní operaci.
- Jednoduchý `for`, `if`, přiřazení a sestavení výsledku nech přímo ve funkci, pokud tím lépe ukazuješ tok dat.
- Atomická funkce znamená jednu odpovědnost, nikoli jeden příkaz. Nezaváděj mechanický limit počtu řádků funkce.
- Nevytvářej řetěz jednorádkových helperů, kvůli kterému musí čtenář neustále přeskakovat mezi implementacemi.
  I krátký helper je vhodný, pokud pojmenovává skutečné pravidlo nebo skrývá detail cizího API.
- Běžné `if → raise` ponech viditelné. Neskrývej je do obecného `require_condition(...)`.
  Samostatný validátor musí názvem říkat, jaký konkrétní kontrakt ověřuje.
- Krátké comprehensions s jednoduchým účelem jsou v pořádku. Více cyklů, filtrů nebo složitý výpočet rozepiš
  do čitelných kroků. Totéž platí pro generátorové výrazy, řetězce `map`/`filter` a složité `lambda` funkce.
- Důležitou operaci umísti před její lokální helpery, pokud to dovoluje organizace modulu.

### Maximální hloubka zanoření: 2

- Tělo funkce je úroveň 0. Vnořený řídicí nebo kontextový blok přidává úroveň:
  `if`, `for`, `while`, `try`, `with`, `async for`, `async with` a `match`.
- `for → if`, `for → for` nebo `with → for` jsou povolené. `for → if → if` už je zakázané.
- `elif` je alternativa na stejné úrovni jako příslušný `if`. `except`, `else` a `finally` jsou větve
  na úrovni příslušného `try` nebo cyklu. Jednotlivé větve `case` ber jako větve jednoho `match`.
- Odsazení třídy, definice metody, dekorátory ani zalomené argumenty se do hloubky nepočítají.
- Preferuj časné `return`, `continue`, konkrétní kontroly a smysluplné helpery.
- Nepřesouvej složitost do vnořené funkce, callbacku nebo comprehension jen kvůli obcházení limitu.
- Rozšířené zanoření JSX povolené pro frontend se na běžný Python kód nevztahuje.

### Pojmenování

- Identifikátory piš anglicky. Preferuj delší přesný název před krátkým nejednoznačným.
- Funkce, metody, proměnné a moduly používají `snake_case`; třídy a vlastní typy `PascalCase`.
- Běžné modulové konstanty mohou používat `UPPER_SNAKE_CASE`. Boolean konstanty dodržují prefixy níže.
- Nepoužívej vlastní zkratky: `signal_specification`, `configuration`, `parameter`, `expression`,
  `sampling_frequency`, `sample_index`; nikoli `spec`, `cfg`, `param`, `expr`, `sfreq` nebo `idx`.
- Také index pojmenuj podle významu, například `epoch_index`, nikoli `i`.
- Vyhýbej se neurčitým názvům jako `data`, `item`, `result`, `manager` nebo `process`, pokud skrývají význam.
  Krátký název je v pořádku, je-li význam jasný; například `epoch` ve smyčce přes epochy.
- Vlastní boolean proměnné, parametry, atributy, properties a predikáty začínají výhradně
  `is_`, `can_`, `was_`, `are_` nebo `has_`: `is_training`, `can_export_model`, `are_labels_valid`.
- Boolean konstanta je například `is_validation_enabled`, nikoli `VALIDATION_ENABLED`.
- Preferuj kladnou formulaci; vyhýbej se dvojité negaci. Příkaz `set_is_training` není boolean predikát.
- Zachovej existující veřejná rozhraní, serializovaná pole, konfigurace a smluvené vstupní body,
  například `validate_dl_input`, `FinalTrainingInputDTO`, `fit`, `predict`, `forward`, `__len__` a `__eq__`.
  Totéž platí pro názvy parametrů vyžadované knihovnou, například `requires_grad` nebo `shuffle`.
- Na hranici cizího kontraktu používej srozumitelné místní názvy, například
  `is_shuffle_enabled`, předané knihovně jako `shuffle=is_shuffle_enabled`.
- Nové vlastní názvy ani importní aliasy nezkracuj. Stávající knihovní aliasy neměň plošně jen kvůli stylu.
  Ustálenou doménovou terminologii jako EEGNet nebo MOABB zachovej.

### Podmínky

- Podmínku se dvěma a více logickými operátory pojmenuj boolean proměnnou nebo helper predikátem.
- Počítej `and`, `or` a logickou negaci `not`; nepočítej `!=`, `is not` ani `not in` jako samostatnou negaci.
  `is_training and not has_validation` tedy patří do pojmenované podmínky.
- Proměnnou použij pro místní rozhodnutí; helper pro samostatné pravidlo, složitější logiku nebo opakované použití.
- Pojmenuj význam, například `can_start_training`, nikoli `is_condition_met`.
- Uvnitř pojmenovaného predikátu může být složený výraz přímo v `return`.
  Nevytvářej další helper pro každý logický operátor.
- Zachovej krátké vyhodnocování a pořadí volání. Extrakce nesmí předčasně číst atribut hodnoty `None`
  ani provést vedlejší účinek, který původní podmínka přeskočila.
- Pro `None` používej `is None` a `is not None`. Nezaměňuj `None` s prázdnou kolekcí nebo nulou.
- NumPy pole a tenzory nepoužívej jako implicitní boolean podmínku. Pojmenuj konkrétní kontrolu,
  například prázdnost, tvar, konečnost hodnot nebo shodu všech prvků.
- Zanořené podmíněné výrazy `value_if_true if condition else value_if_false` nepoužívej.

### Knihovny a hranice systému

- Operaci, jejíž význam vyžaduje znalost konkrétního knihovního API, skryj za helper s názvem popisujícím účel.
  Například `numpy.isclose` používej uvnitř `are_numbers_close_to_each_other`.
- Název helperu musí odpovídat skutečnému výsledku: porovnání polí potřebuje explicitní pravidlo,
  zda rozhoduje shoda všech prvků, některého prvku nebo jiná doménová podmínka.
- Běžné operace Pythonu jako `len`, `append`, `enumerate`, `zip`, `sorted` nebo `strip` automaticky neobaluj.
- Odděl doménová rozhodnutí a orchestrace od frameworkových detailů, práce se soubory, sítí a serializace.
  Pokud čistá logika potřebuje vnější operaci, předej úzkou funkci nebo smysluplný malý kontrakt.
- NumPy, PyTorch, MNE a další knihovní detaily soustřeď do konkrétních výpočetních nebo integračních funkcí
  pojmenovaných podle účelu. Nemíchej jejich nejasná volání přímo do hlavní orchestrace pipeline.
- Zachovej efektivní vektorové výpočty a dávkové zpracování. Kvůli abstrakci nepřeváděj velká pole a tenzory
  na Python seznamy ani nenahrazuj vhodnou vektorizaci pomalými smyčkami.
- Na hranicích hlídej význam os, jednotky, tvar polí a očekávané typy podle existujícího kontraktu.
  Neobnovuj zrušené kontroly vstupního DL kontraktu popsané výše.
- Neobaluj každou standardní metodu frameworku novou vrstvou. Helper má chránit hranici nebo vysvětlovat účel.

### Idiomatický Python, typování a chyby

- Pro jednoduchý výpočet použij funkci. Třída má smysl pro stav, invarianty, životní cyklus nebo skutečný kontrakt.
- Pro pojmenovaná data zvaž `dataclass`; `Protocol` nebo abstraktní bázi zaváděj jen při skutečné potřebě.
  Nevytvářej náhrady existujících DTO pouze kvůli nové vrstvě.
- Nové a upravované funkce typuj včetně návratové hodnoty. Používej syntaxi dostupnou v Pythonu 3.10,
  který projekt deklaruje; například `list[float]` a `str | None`.
- Nepotlačuj chybějící kontrakt pomocí `Any`, neověřeného `cast` nebo nezdůvodněného `type: ignore`.
  Vnější vstupy ověř na jejich hranici. Pokud mypy neumí knihovní typ přesně vyjádřit, izoluj konkrétní omezení.
- Pro vstupy zvaž `Sequence`, `Iterable` nebo `Mapping`, pokud odpovídají skutečně potřebným operacím.
  Nevyžaduj `list`, pokud funkce nepotřebuje jeho změny; u velkých numerických dat zachovej vhodné pole či tenzor.
- Vstupy neměň bez zjevného kontraktu a nepoužívej měnitelné výchozí argumenty jako `[]` nebo `{}`.
- Používej context managery pro zdroje. `with` ale zůstává součástí limitu zanoření.
- Zachytávej konkrétní výjimky tam, kde je umíš vyřešit nebo doplnit smysluplný kontext.
  Chybu nepohlcuj a nevracej úspěch při selhání. Při převodu chyby zachovej příčinu pomocí `raise ... from error`.
- Nepoužívej výjimky jako náhradu běžného rozhodování, pokud je kontrola přímo a čitelně dostupná.
- Docstring vysvětluje veřejný kontrakt, jednotky, tvary, vedlejší účinky a chyby, pokud nejsou jasné z typů a názvu.
  Komentář vysvětluje důvod nebo invariant; neopisuje příkaz.

### Formátování a ověření

- Každý ručně psaný řádek kódu má nejvýše 120 znaků, včetně komentářů, docstringů a importů.
- Odsazuj čtyřmi mezerami. Delší řádek rozděl; kratší nech pohromadě, pokud je to čitelnější.
- Víceřádkové výrazy zalamuj pomocí závorek. Nevkládej do řetězců zalomení, která mění jejich hodnotu.
- Používej existující Ruff konfiguraci. Formátování není samo o sobě zárukou pevného limitu;
  délku změněného kódu zkontroluj také pomocí `ruff check --select E501` a ručně u výjimek nástroje.
- Přečti změněný kód jako text: jsou kroky a tok dat jasné bez soustavného otevírání helperů?
- Ověř jednu odpovědnost, názvy bez vlastních zkratek, boolean prefixy a pojmenované složité podmínky.
- Ověř maximálně dvě úrovně zanoření, délku řádků a smysluplné oddělení knihovních detailů.
- Při změně chování testuj veřejný kontrakt, hraniční vstupy a chyby. Nekopíruj implementaci do testu.
- Spusť relevantní existující kontroly změněného kódu pomocí projektového prostředí: Ruff, mypy a pytest
  podle rozsahu změny a dostupných závislostí. Neinstaluj nové nástroje jen kvůli těmto instrukcím.
- Nalezená porušení oprav před odevzdáním. Pokud dodržení brání neměnný kontrakt, popiš konkrétní odchylku;
  nevypínej pravidla plošně. Uváděj, co bylo ověřeno a co zůstalo neověřené.
