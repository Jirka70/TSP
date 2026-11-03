# Kontext pro agenty v projektu TSP

## Deep learning a kontrakt vstupních dat

Před prací na DL, rozdělení dat, augmentaci, evaluaci nebo serializaci modelu si přečti [plán DL kroku](docs/dl-kontrakt-a-plan.md). Obsahuje uživatelské požadavky, stav integrace, návrh rozhraní a pořadí práce.

- Uživatel požaduje, aby DL přijímal explicitní **train, val a test**, nikoli foldy.
- **Train** slouží k učení. **Val** slouží k validaci po každé trénovací epoše a výběru modelu. **Test** slouží k závěrečnému nezávislému vyhodnocení.
- Současný splitter posílá foldy. To je nesoulad kontraktů, nikoli požadovaný návrh DL. Nezaváděj kvůli tomu foldy do nového rozhraní DL.
- Nepřejmenovávej test na val, neslévej foldy a nevybírej potichu první fold jako řešení tohoto nesouladu. Případný převod patří mimo DL a musí mít explicitní pravidla pro tři oddělené sady.
- Vývoj DL může začít proti cílovému kontraktu s malými syntetickými daty; není nutné čekat na úpravu splitteru.
- Prioritou jsou chyby, které ovlivňují skutečný vstup DL, jeho učení nebo důvěryhodnost výsledků. Rozlišuj potvrzenou chybu od rizika závislého na konfiguraci.
- Zachovej funkčnost ostatních modelů a použití splitteru. Změna požadavku na DL sama o sobě neznamená odstranit foldy z celého projektu.

## Stav a způsob pokračování

K 2026-11-03 byl zapsán kontext a plán; tento zápis neimplementuje nové rozhraní ani opravy. Poslední požadavek uživatele byl vytvořit dokumentaci pro další vlákna. Tento plán sám není pokyn automaticky provést celý refaktor: pokračuj podle aktuálního zadání uživatele. Při následném zadání implementace není potřeba znovu schvalovat již výslovně dohodnutý směr train/val/test.

Před změnami ověř aktuální kód a stav pracovního stromu. Starší rozbory v `reviews/` jsou podklady k ověření, ne záruka, že nález stále platí. Po implementaci udržuj stav úkolů v plánu aktuální a uveď provedené ověření. S uživatelem komunikuj česky a vysvětluj konkrétně a jednoduše.
