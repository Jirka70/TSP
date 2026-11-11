# EEGNet: ověření proti běhu z 15. května 2026

Datum ověření: 2026-11-04. Použitý kód: aktuální pracovní strom nad commitem
`b89cc807773ee1963c0c773cb22c4e8721a368fb`, včetně již existujících necommitnutých změn.
Při ověřování nebyl měněn produkční kód ani běžné konfigurace projektu.

## Výsledek

Na stejné osobě vyhodnocované v původním běhu dosáhl současný EEGNet stejné accuracy:
**96,875 % = 279/288 správných klasifikací**. Na tomto konkrétním srovnání se zhoršení neprojevilo.
Shodná accuracy neznamená shodné váhy nebo chyby na stejných jednotlivých pokusech.

| Běh | Vyhodnocovaná data | Accuracy | F1 macro |
| --- | --- | --- | --- |
| 2026-05-15 17:05:29 | Osoba 8, val používaná po každé epoše, 288 epoch | 96,875 % | 96,87 % (zaokrouhlený log) |
| Aktuální, stejná osoba | Osoba 8, oddělený test, 288 epoch | 96,875 % | 96,8732 % |
| Aktuální běžný GeneralSplitter | Náhodný test napříč 9 osobami, 518 epoch | 87,2587 % | 87,2214 % |

Nižší výsledek posledního řádku není přímá regrese vůči prvnímu: měří jinou směs osob a pokusů.
Srovnání na osobě 8 ukazuje, proč nelze původní číslo převzít jako výkon na všech osobách.
Jde o dva jednorázové běhy s předem daným seedem, nikoli odhad variability napříč seedy nebo osobami.

## Zachovaný vstup a parametry

- BNCI2014_001, osoby 1 až 9, všechny jejich session a běhy, úlohy left_hand/right_hand.
- 108 nahrávek, 2 592 epoch. Jedna epocha má tvar `[26, 577]` při 128 Hz.
- Z uložené konfigurace byly převzaté celé sekce source, raw_preprocessing, paradigm,
  epoch_preprocessing a evaluation. Včetně CSD, pásma 8–35 Hz, okna −0,5 až 4 s, baseline,
  posunu časových značek −0,2 s, ICA a zapnutého AutoRejectu.
- Stejné parametry augmentace: jedna kopie, šum 0,01, maskování, posun, změna znaménka a škálování,
  seed 42. Staré ploché položky byly převedené do dnešního vnořeného schématu.
- EEGNet: 15 epoch, batch 16, Adam, learning rate 0,0005, dropout 0,25, kernel 16,
  temporal filters 8, depth multiplier 2, pointwise filters 16, dvě třídy.
- Bez raw augmentace. Vizualizace a export datasetu vypnuté; model, metriky a historie uložené.

Shodu všech historických číselných vzorků nelze doložit: starý běh neukládá vstupní pole ani jejich hashe
a v logu není přesný commit běžícího kódu. Toto je opakování stejného zdroje a uloženého nastavení,
nikoli prokázaná bitová reprodukce historického běhu. Současné vstupy mají uložené SHA-256 hashe.

## Rozdělení dat a výběr modelu

Historický log výslovně uvádí vyčlenění osoby 8 pro společnou validaci a 2 304 epoch ve zbývající části.
Historická konfigurace používá WithinSessionSplitter, 5 foldů; log uvádí 80 foldů.
Výsledek 96,875 % je na validační sadě opakovaně vyhodnocované během učení.

V přímějším současném srovnání je osoba 8 rezervovaná pro závěrečný test. Ostatní osoby se rozdělily
pomocí aktuálního GeneralSplitteru s test fraction 0 a validation fraction 0,2:

- Train: 1 844 originálních epoch, po augmentaci 3 688.
- Val: 460 epoch z ostatních osob, žádná epocha osoby 8.
- Test: všech 288 epoch osoby 8; zastoupení tříd 144/144.
- Vybraná epocha: 14. z 15 podle val accuracy 86,7391 %.
- Testovací confusion matrix v pořadí štítků `[2, 3]`: `[[136, 8], [1, 143]]`.

Běžný současný GeneralSplitter se seedem 42 a konfigurací 70/10/20 poskytl:

- Train: 1 867 originálních epoch, po augmentaci 3 734.
- Val: 207 epoch.
- Test: 518 epoch; třídy 249/269, confusion matrix `[[212, 37], [29, 240]]`.
- Vybraná epocha: 14. z 15 podle val accuracy 88,8889 %.

Test v obou současných bězích sloužil až k závěrečnému reportu a nevybíral epochu ani nastavení.
Celá val sada také posloužila k již implementovanému ověření serializace.

## Co je jinak a co je problém

1. **Potvrzený problém interpretace poměrů ve splitteru.** Nastavení 70/10/20 vytvoří přibližně
   72/8/20: test se odebere z celku a validation fraction se aplikuje až na zbývající část.
   Hodnota train fraction se při určení hranic přímo nepoužije, ačkoli validátor požaduje součet 1.
   Ověřeno kódem GeneralSplitteru i skutečnými počty 1 867/207/518. V rámci tohoto ověření neopraveno.
2. **Normalizace a seed nejsou stejné jako historicky.** Současné nastavení používá
   `per_epoch_channel`, training seed 67 a deterministic mode. Historická konfigurace tyto položky
   neobsahuje; git historie zavádí normalizaci v commitu `a4d25cb` dne 16. května.
   Z těchto dvou běhů nelze izolovat její přínos nebo škodu. Na osobě 8 se celková accuracy nezhoršila.
3. **Vstup obsahuje i ne-EEG kanály.** Do sítě jde 22 EEG/CSD, 3 EOG a STI. Na osobě 1 byl STI
   v každé ověřované epoše nulový; diagnostika nepotvrdila přímé prozrazení třídy tímto kanálem.
   Bez samostatné ablace nelze vyčíslit vliv EOG. Není to doložená nově vzniklá regrese.
4. **Náročnost na paměť/volné místo.** Standardní společný běh skončil během předzpracování
   chybou `OSError: [Errno 28] No space left on device`, ještě před učením. Příčinu zaplnění
   z dostupného tracebacku nelze přesně určit. Proto se stejné předzpracovací implementace spustily
   postupně po osobách a původní kontinuální záznamy se uvolňovaly. Oba následné běhy doběhly.
5. **Rychlost nelze férově označit za regresi.** Historické trénování podle logu trvalo 451,95 s.
   Současné měření dalo přibližně 524,58 s (běžný split) a 473,71 s (osoba 8), ale současné běhy
   se částečně překrývaly a měly 4/2 CPU vlákna. Nejde o kontrolovaný výkonnostní benchmark.

AutoReject byl v obou současných bězích u všech 108 nahrávek přeskočen, protože po CSD nebyly
nalezené kanály typu EEG. Totéž uvádí historický log. Zapnutá konfigurace tedy neznamená,
že tento konkrétní vstup skutečně prošel AutoRejectem.

Učení předchází ICA na jednotlivých celých měřeních. Běžný náhodný split proto nelze vydávat za
ověření zobecnění na nové lidi nebo za důkaz absence vlivu předzpracování přes hranice sad.
Ověření nepřidává ani neobchází kontroly původu, které byly na dřívější pokyn odstraněny.

## Artefakty a reprodukce

- `historical_config.yaml`, `historical_results.json`: kopie původní konfigurace a čísla vyčtená z logu.
- `current_run.yaml`: úplná převedená konfigurace současného běhu.
- `source_fingerprints.json`: commit, stav pracovního stromu a hashe produkčních Python souborů.
- `run_benchmark.py`: běžný současný GeneralSplitter.
- `run_subject_8_benchmark.py`: explicitní rezervace osoby 8, train/val z ostatních osob.
- `inspect_channels.py`: diagnostika původního kanálového vstupu na osobě 1.
- `outputs/dl-performance-2026-11-04/results.json`: úplné výsledky běžného splitu.
- `outputs/dl-performance-2026-11-04/subject-8-held-out/results.json`: úplné výsledky osoby 8.
- V obou výstupních adresářích: `artifacts/eegnet-*/models/eegnet.pt`, metriky, historie a konfigurace.
- `run.log`: první neúspěšný společný běh; `streamed_run.log` a `subject_8_run.log`: dokončené běhy.

Spuštění z kořene projektu:

```sh
PYTHONPATH="$PWD" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv/bin/python reviews/dl-performance-2026-11-04/run_benchmark.py
PYTHONPATH="$PWD" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python reviews/dl-performance-2026-11-04/run_subject_8_benchmark.py
```

Použité verze: torch 2.11.0, MNE 1.11.0, MOABB 1.5.0, braindecode 1.3.2, NumPy 2.4.3, TorchEEG 1.1.3.
Zařízení CPU. Oba běhy skončily návratovým kódem 0 a prošly současným ověřením uloženého modelu na val.
Nové benchmarkové skripty prošly Ruff kontrolou s limitem 120 znaků.
