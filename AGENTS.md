# Kontext pro agenty v projektu TSP

## Deep learning a kontrakt vstupních dat

Před prací na DL, rozdělení dat, augmentaci, evaluaci nebo serializaci modelu si přečti [plán DL kroku](docs/dl-kontrakt-a-plan.md). Obsahuje uživatelské požadavky, stav integrace, návrh rozhraní a pořadí práce.

- Po merge splitteru uživatel upřesnil hranici: hlavní DL trainer přijímá pouze explicitní **train a val**, nikoli foldy ani test. **Test** dostává samostatný závěrečný evaluator.
- **Train** slouží k učení. **Val** slouží k validaci po každé trénovací epoše a výběru modelu. **Test** slouží k závěrečnému nezávislému vyhodnocení.
- `GeneralSplitter` vrací přímo train/validation/test. MOABB splittery dál vracejí foldy pro samostatnou evaluační pipeline; nezaváděj je do hlavního DL traineru.
- Nepřejmenovávej test na val, neslévej foldy a nevybírej potichu první fold jako řešení tohoto nesouladu. Případný převod patří mimo DL a musí mít explicitní pravidla pro tři oddělené sady.
- `FinalEEGNetTrainer` používá přímo `FinalTrainingInputDTO`. `validate_dl_input()` ověřuje train/val a vrací `ValidatedDeepLearningInput`. Dočasný `DLInputDTO` a adaptér byly odstraněny; neobnovuj je.
- Na následné zadání byly odstraněny kontroly specifikace signálu a původu epoch. DL nevyžaduje
  `signal_specification` ani `epoch_origins` a neověřuje podle nich překryvy, augmentaci nebo fit preprocessingu.
  Zůstávají technické kontroly polí a tříd. Správné rozdělení dat zajišťuje příprava dat a splitter.
- Prioritou jsou chyby, které ovlivňují skutečný vstup DL, jeho učení nebo důvěryhodnost výsledků. Rozlišuj potvrzenou chybu od rizika závislého na konfiguraci.
- Zachovej funkčnost ostatních modelů a použití splitteru. Změna požadavku na DL sama o sobě neznamená odstranit foldy z celého projektu.

## Stav a způsob pokračování

K 2026-11-03 byl zapsán kontext a plán; tento zápis neimplementuje nové rozhraní ani opravy. Poslední požadavek uživatele byl vytvořit dokumentaci pro další vlákna. Tento plán sám není pokyn automaticky provést celý refaktor: pokračuj podle aktuálního zadání uživatele. Při následném zadání implementace není potřeba znovu schvalovat již výslovně dohodnutý směr oddělených train/val/test; hranici traineru upřesňuje aktuální pravidlo výše.

Před změnami ověř aktuální kód a stav pracovního stromu. Starší rozbory v `reviews/` jsou podklady k ověření, ne záruka, že nález stále platí. Po implementaci udržuj stav úkolů v plánu aktuální a uveď provedené ověření. S uživatelem komunikuj česky a vysvětluj konkrétně a jednoduše.

## Závazné konvence čitelnosti kódu

Platí pro nový a upravovaný kód. Nesouvisející starší kód kvůli nim plošně nepřepisuj.

1. Funkce vracející boolean pojmenovávej s prefixem `has`, `is`, `are`, `was` nebo `can`.
2. Složitější podmínky přesuň do pojmenovaných boolean funkcí. Z podmínky musí být jasné,
   co rozhoduje, bez čtení matematických nebo knihovních detailů.
3. Kód má plynout jako text. Každá funkce či metoda musí názvem jednoznačně říkat, co dělá.
   Helpery odděluj podle samostatné odpovědnosti nebo složitější logiky, ne mechanicky po jednotlivých
   příkazech. Jednoduché `for`, `if`, přiřazení a sestavování výsledků nech přímo ve funkci, pokud tím
   čtenář lépe vidí tok dat. Nevytvářej řetěz jednorádkových helperů, kvůli kterému musí čtenář neustále
   přeskakovat do dalších implementací. Běžné `if → raise` neskrývej za obecné volání typu
   `require_condition(...)`; helper má pojmenovávat konkrétní kontrolu nebo výpočet.
   Neskrývej složitost v dlouhých comprehensions.
4. Nové názvy piš celými slovy, bez zkratek: například `signal_specification`, nikoli `spec`,
   `sampling_frequency`, nikoli `sfreq`. Zachovej názvy existujících veřejných rozhraní,
   knihovních polí a smluvených vstupních bodů jako `validate_dl_input`; na jejich hranici je
   převeď na srozumitelné interní názvy. Nové zkratky nezaváděj.
5. Knihovní operace, jejichž význam vyžaduje znalost API, obal pomocnou funkcí s názvem
   popisujícím účel. Například `numpy.isclose` používej uvnitř
   `are_numbers_close_to_each_other`. Platí i pro jiné operace než NumPy.
6. Uvnitř jedné funkce používej nejvýše dvě vnořené řídicí struktury, například `for → if` nebo
   `for → for`. Základní odsazení těla funkce ani odsazení třídy se do tohoto počtu nezapočítává;
   zalomení argumentů také není řídicí blok. Cílem je omezit hluboké vnoření, nikoli odstranit cykly
   a podmínky. Jednoduché cykly a podmínky jsou žádoucí, pokud zlepšují srozumitelnost. Teprve další
   vnoření řeš extrakcí smysluplného helperu nebo předčasným návratem.
7. Řádek má nejvýše 120 znaků. Delší řádky rozděl; kratší nech na jednom řádku, pokud je to
   čitelnější. Formátuj pomocí Ruff s limitem 120 znaků.
