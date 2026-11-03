**Problémy v přípravě dat před deep learningem**

Ověřeno 29. 10. 2026 na revizi `e9c1efe`, stejné jako při původním rozboru. Rozsah končí před učením sítě: načítání → raw preprocessing → raw augmentace → epochování → epoch preprocessing → split → epochová augmentace → předání trenérovi. Evaluator a serializace modelu do tohoto navazujícího rozboru nepatří. Export datasetu je zahrnut, protože pracuje s výsledkem přípravy dat před tréninkem.

**Před DL mohou vzniknout špatné štítky, chybné časování, ztracené záznamy a překryv tréninku s validací. Oprava samotné sítě tyto problémy nevyřeší.** Některé chyby jsou podmíněné konkrétní konfigurací; níže je vždy uvedeno kterou.

Bylo spuštěno 16 lokálních diagnostických scénářů na malých syntetických datech s reálnými implementacemi kroků. Sondy netrénují DL a nestahují datasety. Zdrojová implementace nebyla upravena. Kde jde pouze o závěr z kódu, je to výslovně uvedeno.

**1. Načtené soubory mohou při splitu ztratit všechny vzorky**

[FilesystemDatasetLoader](/Users/jirka/TSP/src/impl/data_loader/FilesystemDatasetLoader.py:120) nastavuje `session_id=None` a `run_id=None`. [BasicSplitter](/Users/jirka/TSP/src/impl/split/basic_splitter.py:112) a [MoabbSplitter](/Users/jirka/TSP/src/impl/split/moabb_splitter.py:112) pak podle těchto hodnot seskupují data přes pandas `groupby`, které zde řádky s chybějícími klíči vynechá.

Ověřeno: při rekonstrukci 10 vybraných vzorků s těmito identifikátory vrátily oba splittery **0 záznamů**. BasicSplitter následně může hlásit obecné selhání rozdělení, přestože příčina vznikla už v loaderu. Při smíchání úplných a neúplných metadat může vypadnout jen část dat.

Lidsky: soubory jsme načetli, ale při ukládání do přihrádek pro sessions je zahodíme, protože nemají jmenovku.

Oprava: vytvořit stabilní identifikátory podle skutečné struktury datasetu a jednoznačné `recording_id` podle souboru. Pokud je session neznámá, zacházet s tím explicitně. Pouhé doplnění stejného `run_id` všem souborům by mohlo sloučit nezávislé záznamy. Ověřovat zachování počtu a identity vzorků při rekonstrukci.

**2. Loader může zahodit platné události nebo chybně interpretovat jejich čas**

[Globální TSV anotace](/Users/jirka/TSP/src/impl/data_loader/FilesystemDatasetLoader.py:150) se vždy dělí 1000. To je správně pro očekávaný dataset s milisekundami, ale konfigurace neobsahuje jednotku ani kontrolu tohoto předpokladu. Soubor s onsetem `2` v sekundách byl načten jako `0.002` sekundy.

[Přizpůsobení anotací délce záznamu](/Users/jirka/TSP/src/impl/data_loader/FilesystemDatasetLoader.py:168) vyžaduje kladnou délku události. Platné okamžité značky s `duration=0` proto vyřadí. Ověřeno: dvě události uvnitř záznamu s nulovou délkou vedly k výsledku `None`.

Globální anotace se také podle kódu připojují ke všem souborům, které nemají vlastní anotace, bez ověření jejich vazby ke konkrétnímu měření. To je použitelné jen tehdy, pokud skutečně všechny soubory sdílejí stejný časový protokol. Přítomnost libovolné vestavěné anotace naopak zablokuje fallback, i když jde pouze o poznámku o vadném úseku a chybí cílové události.

Oprava: explicitní jednotky a vazba anotací na záznam; zachovat okamžité události v platném časovém rozsahu. Chybějící požadované události hlásit přímo při načítání nebo epochování.

**3. Epochování ignoruje nakonfigurovaná čísla tříd**

[ParadigmPreprocessor](/Users/jirka/TSP/src/impl/paradigm/paradigm_preprocessing.py:60) ze slovníku událostí použije jen názvy. Čísla přidělí MNE zvlášť pro každý záznam, ještě před filtrováním vybraných událostí.

Ověřeno se skutečným krokem: konfigurace `{left_hand: 10, right_hand: 20}` dala výstup `{left_hand: 1, right_hand: 2}`. Přidání anotace `cue` změnilo výstup na `{left_hand: 2, right_hand: 3}`.

Lidsky: slovník říká „10 znamená levou ruku“, ale krok si vytváří vlastní slovník a mezi soubory ho mění. Číslo 2 tak může znamenat různé pohyby.

Oprava: jedna explicitní mapa názvů a čísel platná pro všechny záznamy; zkontrolovat unikátnost a uložit ji s daty. Normalizované názvy nesmějí vytvářet neodhalené kolize.

**4. Po změně signálu zůstávají neaktuální nebo neúplná metadata**

[Raw preprocessing](/Users/jirka/TSP/src/impl/raw_preprocessing/raw_preprocessing.py:118) a [epochování](/Users/jirka/TSP/src/impl/paradigm/paradigm_preprocessing.py:26) nahrazují `data`, ale neaktualizují doprovodný slovník metadat. Loader používá klíč `channel_names`, zatímco [splitter](/Users/jirka/TSP/src/impl/split/basic_splitter.py:74) zachovává seznam pod názvem `ch_names`. Původní názvy kanálů proto po převodu na pole zmizí.

Ověřený průchod epochováním, splitterem a [FIF exportérem](/Users/jirka/TSP/src/impl/dataset_export/fif_dataset_exporter.py:55):

| Informace | Skutečný signál | Exportovaný soubor |
| --- | --- | --- |
| Sampling rate | 128 Hz po převzorkování | 200 Hz ze starých metadat |
| Kanály | C3, C4, F3, F4 | EEG 000, EEG 001, EEG 002, EEG 003 |

Lidsky: zvukovou nahrávku upravíme, ale na obalu necháme starou rychlost přehrávání. Při novém načtení má jiný časový význam. Export navíc vytváří umělé onsets a neobnovuje původní časovou osu epoch.

Oprava: jednotné schéma metadat a jeho aktualizace po každé transformaci. Zachovat pořadí i typy kanálů, jednotky, skutečnou frekvenci, tmin a mapu událostí. Neodhadovat neznámé údaje v exporteru výchozí hodnotou. Neaktuální metadata sama nemění čísla v přímém vstupu DL, ale znemožňují jejich spolehlivou interpretaci a kazí následný export.

**5. Požadovaný preprocessing se může přeskočit a běh pokračuje**

[CSD transformace](/Users/jirka/TSP/src/impl/raw_preprocessing/raw_preprocessing.py:90) při chybě pouze zaloguje varování a vrátí původní signál. V testu bez pozic elektrod bylo CSD zapnuté, ale data zůstala přesně stejná a typ kanálů zůstal EEG.

Pokud jeden záznam podmínky splňuje a druhý ne, může pipeline spojit transformované a netransformované signály se stejným rozměrem, ale rozdílným významem a jednotkami. Tento smíšený průchod je riziko odvozené z kódu; samostatně byl ověřen fallback bez transformace.

Oprava: povinnou transformaci při nesplněných podmínkách odmítnout. Pokud má být fallback podporovaný, musí být explicitní, zaznamenaný a kompatibilní s ostatními vzorky; pouhé varování nestačí.

**6. Nastavení alignment nezmění vzorky, které uvidí DL**

[EpochPreprocessor](/Users/jirka/TSP/src/impl/epoch_preprocessing/epoch_preprocessing.py:71) volá `epochs.shift_time`. To změní popisky časové osy, nikoli hodnoty signálu. Při následném převodu na NumPy pole časová osa zmizí.

Ověřeno: po posunu `-0.2 s` byl nový `tmin=-0.2`, ale všechna čísla ve vstupu zůstala bitově shodná. Pokud účelem nastavení je kompenzovat zpoždění ve vstupních vzorcích DL, nemá očekávaný účinek. Samotné přeznačení osy může být legitimní operace, ale musí být takto popsáno.

Oprava: určit, zda chceme změnit interpretaci času, nebo skutečně posunout výběr signálu. Kompenzaci zpoždění pro DL řešit posunem událostí/epochovacího okna či definovaným posunem vzorků a testem změny výsledného pole.

**7. ICA nemá spolehlivý kontrakt odstraňování artefaktů**

[Raw ICA](/Users/jirka/TSP/src/impl/raw_preprocessing/raw_preprocessing.py:74) se fituje a aplikuje, ale žádné komponenty se neoznačí k odstranění. Samotné rozložení a složení signálu není detekce a odstranění mrkání. V sondě se čtyřmi kanály a čtyřmi komponentami byl maximální rozdíl vstupu a výstupu přibližně `1.9e-19`; vstup měl směrodatnou odchylku přibližně `1e-5`. V tomto běhu FastICA také vydalo varování o nekonvergenci, které nezastavilo krok.

[Epochová ICA](/Users/jirka/TSP/src/impl/epoch_preprocessing/epoch_preprocessing.py:89) už volá detekci očních artefaktů, ale bez nastaveného náhradního EOG kanálu. Na jinak platném EEG záznamu bez EOG kanálu skončila `No EOG channel(s) found`. Kontrola potřebného počtu kanálů, ranku a existence EOG neprobíhá před fitem. Výchozí epochová konfigurace ICA zapíná a požaduje 15 komponent.

Oprava: raw ICA musí mít skutečnou politiku výběru komponent, nebo nesmí slibovat odstranění artefaktů. U epochové ICA předem ověřit předpoklady a explicitně stanovit chování bez EOG; neoznačovat automaticky jakýkoli EEG kanál za EOG.

**8. Naučené úpravy se fitují ještě před dělením dat**

[TrainingPipeline](/Users/jirka/TSP/src/pipeline/training/training_pipeline.py:82) provádí preprocessing před splitem. ICA a AutoReject tedy při rozdělení jednoho záznamu po epochách čerpají i z pozdější testovací části. [CSP](/Users/jirka/TSP/src/impl/epoch_preprocessing/epoch_preprocessing.py:113) se dokonce učí ze štítků všech těchto epoch.

Jde o závěr z pořadí volání a vstupů fitovacích metod. Dopad je podmíněný zvoleným splittem a evaluačním protokolem: předem definovaný filtr není totéž jako transformace naučená na testových datech. Neoznačuji každý preprocessing před splitem automaticky za chybu. Kalibrace na celém novém záznamu může být povolená, ale musí odpovídat deklarovanému způsobu používání.

Oprava: hranice train/validation/test určit podle původu dat před učením transformací. Fit na train, transform ostatních; případnou adaptaci na nový záznam definovat a hodnotit zvlášť.

**9. Raw augmentace mění časovou referenci a může propojit train s testem**

[TorchEEGRawAugmentor](/Users/jirka/TSP/src/impl/raw_augmentation/torcheeg_raw_augmentor.py:89) vytvoří nový RawArray bez zachování `first_samp`, pak na něj přenese původní anotace. Pro záznam začínající nenulovým samplem se tím mohou posunout události vůči datům.

Ověřeno na záznamu oříznutém o 1 sekundu při 200 Hz, s augmentací nastavenou na násobení přesně 1: originál měl události na relativních samplech `[200,400]`, kopie na `[400,600]`. Ačkoli jsme signál neměli měnit, epochy z kopie se vyříznou jinde.

Navíc raw augmentace běží před splitem a vytváří nové `session_id`. Podle kódu může originál a jeho blízká kopie při dělení po epochách nebo podle umělých sessions skončit na opačných stranách train/test. Dělení podle skutečného subjektu může tento konkrétní únik zabránit, pokud skutečně zachová všechny kopie společně.

Oprava: zachovat časovou referenci, anotace a identitu zdroje. Uchovávat `origin_recording_id` a `origin_trial_id`; všechny příbuzné kopie držet ve stejné partition. Nejjednodušší bezpečná varianta je augmentovat až trénovací část po rozdělení.

**10. Výchozí velikost přidávaného šumu může přehlušit signál**

[Basic augmentor](/Users/jirka/TSP/src/impl/augmentation/basic_augmentor.py:118) přičítá šum v jednotkách vstupních čísel. Konfigurace Basic i TorchEEG mají `std=0.01`, ale nevážou tuto hodnotu na jednotky nebo sílu signálu.

Pro vstup ve voltech se směrodatnou odchylkou přibližně `10 µV` sonda přidala šum s odchylkou `0.0098 V`: zhruba **982krát větší než signál**. Toto číslo platí pro ověřenou kombinaci; není univerzálním hodnocením všech datasetů, normalizovaných vstupů nebo CSD dat.

Lidsky: místo lehkého zašumění řeči pustíme přes ni velmi hlasitý šum. Pozdější normalizace neobnoví ztracený poměr signálu a šumu.

Oprava: explicitní jednotky, kontrola rozsahu a ideálně šum nastavený relativně k signálu nebo požadovanému SNR. Ověřit, že upravený vzorek ještě nese stejnou úlohu a štítek.

**11. MOABB globální validace není při režimu po splitu nezávislá na foldech**

[MoabbSplitter](/Users/jirka/TSP/src/impl/split/moabb_splitter.py:278) z každého foldu odebere vlastní validační vzorky a pak je sloučí do jednoho globálního poolu. Co odebral v jednom foldu, může být stále trénovací v jiném.

Ověřeno na 60 vzorcích a třech foldech: globální validace měla 23 vzorků a s tréninkem jednotlivých foldů se překrývala v **6, 6 a 10 vzorcích**. S jejich testy se překrývala v 9, 9 a 5 vzorcích. Samotný překryv s testem neznamená únik do tréninku, ale potvrzuje, že pool není zvláštní nezávislá partition.

Oprava: buď jednu globální validaci vyčlenit před vytvořením foldů, nebo uchovávat vlastní validaci u každého foldu. Neslučovat je a následně nevydávat za holdout pro libovolný foldový model.

**12. BasicSplitter nedodržuje ve všech režimech význam nastavených poměrů**

Při `exclude_validation_data_before_split=false` se [validační podíl](/Users/jirka/TSP/src/impl/split/basic_splitter.py:207) bere z již vypočtené trénovací části, zatímco train/test se nejdřív přepočítají bez validačního podílu.

Pro 900 vzorků a poměry 70 % / 10 % / 20 % vzniklo **630 train / 70 validation / 200 test** místo očekávaných 630 / 90 / 180. Změna volby pořadí splitu tedy zároveň změní význam čísel. Dále se podle kódu náhodně míchají epochy bez stratifikace a bez ochrany skupin původu; to je vhodné jen pro odpovídající cíl hodnocení, nikoli pro tvrzení o nových osobách či nezávislých nahrávkách.

Oprava: jednoznačně definovat jmenovatel poměrů a testovat výsledné počty. Podle cíle volit stratifikovaný/group split a před tréninkem ověřit zastoupení tříd. Zohlednit příbuzné či překrývající se časové úseky.

**13. Při převodu MNE dat na pole se mohou ztratit štítky**

[CSP větev](/Users/jirka/TSP/src/impl/epoch_preprocessing/epoch_preprocessing.py:115) použije štítky při fitu, ale neuloží je do výsledných metadat. Ověřeno: výstup `(12,2)` bez `labels`, poté BasicSplitter selhal na chybějících štítcích. Navíc jde o 2D příznaky, nikoli 3D EEG signál očekávaný EEGNet. CSP je tedy potřeba pro tuto DL cestu zakázat, nebo navrhnout jiný modelový vstup.

[Basic augmentor](/Users/jirka/TSP/src/impl/augmentation/basic_augmentor.py:149) a [TorchEEG augmentor](/Users/jirka/TSP/src/impl/augmentation/torcheeg_augmentor.py:214) zapíší nové štítky jen tehdy, pokud klíč `labels` existoval už předem. Při přímém vstupu MNE Epochs jsou ale štítky uvnitř objektu MNE. Ověřeno pro Basic: ze dvou epoch vznikly čtyři vzorky, ale žádné štítky v metadatech. V obvyklé cestě přes zapnutý splitter se klíč doplní a tato konkrétní chyba se neprojeví.

Oprava: při převodu reprezentace vždy explicitně přenést `y`, ne pouze podle přítomnosti původního klíče. Chybějící štítky odmítnout; nevytvářet náhradní nuly.

**14. Chybí souvislá kontrola předpokladů a výstupního datasetu**

Rozhraní [IRawDataValidator](/Users/jirka/TSP/src/types/interfaces/raw_data_validator.py:8) existuje, ale v přípravné pipeline není zapojené. Jednotlivé knihovny provádějí dílčí kontroly, nikde však není kompletně vynuceno: konečná čísla, shodné kanály a jednotky, délky vzorků a štítků po každém záznamu, jednoznačný původ a očekávané zastoupení tříd.

Další ověřený příklad: `paradigm.preload=false` projde configem, ale krok zavolá `len(epochs)` ještě před rozhodnutím o vyřazených epochách a skončí `RuntimeError`. Mezní frekvence filtrů se nekontrolují proti skutečné sampling rate před voláním MNE. Chybějící události a prázdné záznamy se mohou přeskočit bez požadovaného souhrnu toho, která část datasetu zůstala.

Oprava: kontroly hned na vstupu kroku a po transformaci. Před DL vracet dataset s garantovaným schématem a souhrnem počtů za subjekt/session/třídu. Každý úbytek vzorků musí mít dohledatelnou příčinu. Délku `X` a `y` kontrolovat po jednotlivých záznamech; až celkový součet může skrýt vzájemně se vyrušující chyby.

**15. Orchestrace při předání finálnímu trenérovi obejde připravené partition**

[TrainingPipeline, řádek 127](/Users/jirka/TSP/src/pipeline/training/training_pipeline.py:127) předává `epoch_preprocessing_result.data`, tedy dataset před splitem a epochovou augmentací. To současně vrací validační/testovací vzorky do tréninku a vynechává epochovou augmentaci finálního modelu. Jde o již reprodukovaný nález z původního rozboru; při této kontrole byl potvrzen nezměněný zdrojový kód.

I perfektně opravený splitter by tak finální model stále nedostával správná data. Oprava musí zahrnout i místo, které výstupy kroků propojuje.

**Doporučené pořadí oprav**

1. **Význam dat:** pevná mapa štítků, časování anotací a raw kopií, správné identifikátory záznamů, zachování frekvence/kanálů/jednotek. Jinak mohou být vstupy už od začátku chybně označené.
2. **Nezávislost množin:** jednoznačný split podle cíle hodnocení, původ všech kopií, oddělená validace a správná data pro finální trénink. Bez toho nelze věřit výsledným metrikám.
3. **Skutečný účinek preprocessingu:** vyřešit alignment, raw ICA, chování bez EOG, CSD fallback a neslučitelné CSP. Fit naučených transformací umístit podle evaluačního protokolu.
4. **Přiměřenost augmentace a kontrakty:** jednotky šumu, zachování štítků, počet vzorků, finite hodnoty a kontrola každého záznamu.

Dočasné vypnutí problematické volitelné transformace může zjednodušit ověřování, ale samo neopraví mapování událostí, datové partition ani finální předání dat. Pro výběr prvního opravného balíku dává smysl spojit stabilní identitu vzorků, mapu tříd a kontrolu rozdělení dat.

**Reprodukční podklady**

- [upstream_probe.py](/Users/jirka/TSP/reviews/upstream-2026-10-29/upstream_probe.py) obsahuje 16 scénářů. Sondy vypisují i zachycené výjimky; exit kód 0 neznamená, že zkoumaná implementace funguje správně.
- [vysledky-sond.txt](/Users/jirka/TSP/reviews/upstream-2026-10-29/vysledky-sond.txt) obsahuje skutečný výstup včetně varování.
- [Původní rozbor DL](/Users/jirka/TSP/reviews/deep-learning-2026-10-20/rozbor.md) zahrnuje navazující chyby modelu, evaluace a serializace.

Spuštění z kořene repozitáře:

```bash
PYTHONPATH=/Users/jirka/TSP OMP_NUM_THREADS=1 .venv/bin/python reviews/upstream-2026-10-29/upstream_probe.py
```
