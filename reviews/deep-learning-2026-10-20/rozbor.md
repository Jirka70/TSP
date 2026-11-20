**Rozbor připravenosti deep learning kroku na produkci**

Stav k 20. 10. 2026, revize `e9c1efe`. Rozsah: EEGNet, foldový i finální trénink, predikce, evaluace, serializace a načítání, včetně navazujícího dělení dat, augmentace a preprocessingu.

**Verdikt: současný DL krok bych do produkce nepustil.** Hlavní překážkou jsou nedůvěryhodné výsledky evaluace, možnost úspěšně dokončit trénink s poškozenými vahami a nedostatečný kontrakt vstupů a checkpointu. Samotný běžný přenos vah uložením a načtením funguje; produkční spolehlivost ale vyžaduje více než úspěšné `torch.save` a `torch.load`.

P1 označuje chybu, kterou je potřeba odstranit před použitím dotčené funkce v produkci. P2 označuje další konkrétní chybu nebo provozní mezeru; její naléhavost závisí na způsobu nasazení. U podmíněných scénářů je podmínka uvedena přímo u nálezu.

Ověření proběhlo na CPU s Pythonem 3.12.0, PyTorch 2.11.0, Braindecode 1.3.2, NumPy 2.4.3 a Pydantic 2.12.5. Stávající testy: **40 passed**. Syntetický smoke test s foldovým i finálním DL tréninkem a uložením artefaktů doběhl. Níže uvedené cílené sondy přesto reprodukovaly chyby. GPU, skutečný výpadek disku a výkon při produkční zátěži nebyly dynamicky ověřovány; příslušná rizika vycházejí z implementace.

| Priorita | Nález | Hlavní důsledek |
| --- | --- | --- |
| P1 | 1. Finální trénink zahrnuje validaci i test | Výsledná metrika neměří generalizaci |
| P1 | 2. Foldový test slouží k výběru epochy, evaluator ignoruje ostatní modely | Neexistuje správně provedené nezávislé hodnocení foldů |
| P1 | 3. Finální model ignoruje epochovou augmentaci | Nasazený model vzniká jiným postupem než foldové modely |
| P1 | 4. NaN a Inf nejsou zastaveny | Poškozený model může vracet běžné třídy a být uložen |
| P1 | 5. Není vynucen tvar vstupu a výstupu | Jeden vzorek může vrátit několik predikcí |
| P1 | 6. Chybí kontrakt kanálů, jednotek a preprocessingu | Správný rozměr pole neznamená správný signál |
| P1 | 7. Význam číselných štítků není stabilní mezi záznamy | Stejné číslo může označovat jinou událost |
| P1 | 8. Checkpoint projde s neplatnými třídami a vahami | Tichá změna výsledků nebo pozdní pád |
| P1 při použití těchto metrik | 9. Pravděpodobnostní metriky dostávají štítky | Např. nesprávné ROC AUC |
| P2 | 10. Kontrola počtu štítků probíhá až po spojení záznamů | Chyby se mohou vzájemně vyrušit a posunout štítky |
| P2 | 11. Zápis artefaktů není atomický ani unikátní pro běh | Kolize, přepsání nebo neúplný artefakt |
| P1 pro automatické spouštění | 12. Neplatná konfigurace končí úspěšným exit kódem | CI nebo plánovač může vykázat úspěch bez modelu |
| P2 | 13. Validace konfigurace a hraničních vstupů je neúplná | Pozdní a nekonzistentní chyby |
| P2 | 14. Obnovený model nezachovává celé chování objektu | Nefunguje přímé `evaluate`, není podporováno přesné resume |
| P2 | 15. Průměrování a evidence metrik jsou nekonzistentní | Zkreslený loss a nejednoznačné výsledky |
| P2 | 16. Determinismus není izolovaný ani plně vynucený | Chování závisí i na předchozích bězích |
| P2 | 17. Zařízení a paměť nemají řízený životní cyklus | Riziko OOM a nečekaného použití GPU |
| P2 | 18. Testování a distribuce nepokrývají produkční kontrakt | Úspěšné CI neodhalí zásadní problémy |

**1. P1 — Finální model se učí i z dat, na kterých je následně hodnocen**

[TrainingPipeline, řádek 127](/Users/jirka/TSP/src/pipeline/training/training_pipeline.py:127) předává finálnímu trenérovi `epoch_preprocessing_result.data`. Jde o celý dataset před splitem. Současně předává validační podmnožinu téhož datasetu. [FinalEEGNetTrainer](/Users/jirka/TSP/src/impl/model/deep_learning/eegnet/trainer/final_eegnet_trainer.py:17) tuto vstupní trénovací množinu přímo použije.

Ověřeno průchodem skutečnou orchestrace s označenými vzorky a náhradami výpočetně náročných kroků: finální trénink dostal ID `0–9`, přestože `6–7` byla validace a `8–9` test. Syntetický smoke test také trénoval finální model na všech 24 vzorcích a validoval na 2 vzorcích z této množiny.

Navíc se validační množina používá pro výběr nejlepší epochy a potom znovu pro závěrečnou metriku. Pouhé vyloučení validace z tréninku tedy nevyřeší nezávislost závěrečného testu.

**Oprava:** explicitně oddělit trénovací data, validační data pro volbu epochy a dosud nepoužitý test. Finální DTO má dostávat správnou trénovací partition. Uchovávat původní ID vzorků a automaticky kontrolovat průniky. Pokud se po nezávislém vyhodnocení provede refit na širší množině, musí být zřejmé, kterému modelu patří publikovaná metrika.

**2. P1 — Foldový test je použit k ladění a následná evaluace neodpovídá deklarovanému chování**

[EEGNetModelTrainer](/Users/jirka/TSP/src/impl/model/deep_learning/eegnet/trainer/eegnet_model_trainer.py:57) předává `fold.test_data` jako validaci. [TrainingLoop](/Users/jirka/TSP/src/impl/model/deep_learning/eegnet/trainer/eegnet_training_loop.py:45) podle ní vybírá nejlepší checkpoint. Taková množina už není nezávislý test.

Pipeline následně loguje hodnocení foldových modelů na jejich testech, ale [StandardEvaluator](/Users/jirka/TSP/src/impl/evaluator/standard_evaluator.py:54) vezme pouze `trained_models[0]` a globální `validation_data`. Ve zkoušce se dvěma modely nebyl druhý model vůbec zavolán. Výsledek tohoto kroku pipeline dále nezachovává.

[MetricsAggregator](/Users/jirka/TSP/src/impl/model/metrics_aggregator.py:105) agreguje nejlepší validační accuracy, tedy maxima použitá při výběru epochy. To není nezávislá cross-validation metrika.

**Oprava:** v každém foldu oddělit vnitřní validaci od vnějšího testu; evaluator musí párovat model s jeho `fold_idx`, hodnotit všechny foldy a ukládat jejich výsledky. Agregovat výsledky nezávislého testu.

**3. P1 — Epochová augmentace se nepromítne do finálního uloženého modelu**

[Augmentor](/Users/jirka/TSP/src/impl/augmentation/basic_augmentor.py:70) vrací augmentovaná trénovací data ve foldech. Finální trenér ale dostává data před splitem a augmentací, viz [předání DTO](/Users/jirka/TSP/src/pipeline/training/training_pipeline.py:127). Seznam `folds` používá finální DL trenér pouze pro metadata, nikoli jako zdroj trénovacích vzorků.

Označené augmentované vzorky `100–111` se ve finálním tréninku neobjevily. S `fold_training=false` se tedy epochová augmentace může spočítat a exportovat, ale na uložený model nemá vliv. Tento nález se týká augmentace po splitu; raw augmentace před splitem je jiný krok.

**Oprava:** augmentovat správnou finální trénovací partition. Nespojovat naivně všechny foldy, protože by vznikly opakované vzorky a další leakage.

**4. P1 — NaN a Inf mohou vytvořit zdánlivě použitelný výstup**

[Převod vstupu](/Users/jirka/TSP/src/impl/model/deep_learning/eegnet/model/eegnet_model.py:252) kontroluje pouze počet os. Neověřuje konečnost hodnot před ani po převodu na `float32` a normalizaci. [Trénink](/Users/jirka/TSP/src/impl/model/deep_learning/eegnet/model/eegnet_model.py:141) nekontroluje konečnost lossu, gradientů ani vah; [predikce](/Users/jirka/TSP/src/impl/model/deep_learning/eegnet/model/eegnet_model.py:160) nekontroluje pravděpodobnosti.

Jeden NaN ve vstupu vedl k `[NaN, NaN]`, ale `predict()` vrátil běžný štítek první třídy. Trénink s NaN doběhl, loss i váhy obsahovaly NaN a přesto byla zaznamenána nejlepší accuracy `0.5`. Serializer takový model neodmítne. JSON writer navíc používá výchozí povolení NaN, takže poškození může přejít i do nestandardního JSON s historií.

**Oprava:** kontrolovat `isfinite` na hranicích vstupu, po normalizaci, pro loss a výsledné pravděpodobnosti; při numerickém selhání ukončit běh jako neúspěšný a nezveřejnit model. Před uložením zkontrolovat váhy. Pro JSON použít zákaz NaN.

**5. P1 — Změna délky signálu může potichu změnit tvar predikce**

Model ukládá `input_shape`, ale [při inferenci](/Users/jirka/TSP/src/impl/model/deep_learning/eegnet/model/eegnet_model.py:252) nekontroluje soulad počtu kanálů a časových bodů s tréninkem. Současně nekontroluje výstup sítě před softmaxem.

Model vytvořený pro `(N, 2, 128)` přijal vstup `(2, 2, 256)` a vrátil pravděpodobnosti `(2, 2, 5)` a predikce `(2, 5)`. Kontrakt [IModel](/Users/jirka/TSP/src/types/interfaces/model/model.py:49) očekává pravděpodobnosti `(N, n_classes)`. Délka 129 byla přijata také; délka 16 skončila nízkoúrovňovou chybou poolingu. Jiný počet kanálů skončil `EinopsError`.

**Oprava:** validovat `(C, T)` proti uloženému schématu. Počet vzorků `N` smí být jiný než při tréninku. Pokud mají být různé délky podporovány, musí mít definovanou agregaci a otestovaný výstup. Kontrolovat `logits.shape == (N, K)`, konečnost, interval pravděpodobností a součet každého řádku přibližně 1.

**6. P1 — Checkpoint nepopisuje dostatečně význam vstupního signálu**

[Extrakce](/Users/jirka/TSP/src/impl/model/util/extract/extract_learning_data.py:13) spojuje pole a štítky; neporovnává názvy a pořadí kanálů, sampling rate, jednotky ani referenci. [Stav modelu](/Users/jirka/TSP/src/impl/model/deep_learning/eegnet/model/eegnet_model.py:188) ukládá rozměry a DL konfiguraci, nikoli úplné vstupní schéma.

Signály `[C3, C4]` a `[C4, C3]` mají stejný tvar, ale odlišný význam. Stejný počet časových bodů při jiné frekvenci znamená jinak dlouhé okno. Normalizace s konstantou `1e-6` navíc sama nezaručuje identické chování pro volty a mikrovolty.

Konfigurace celé pipeline může být uložena v `config.json`, ale [loader](/Users/jirka/TSP/src/impl/model/deep_learning/eegnet/load/eegnet_model_loader.py:181) ji neobnovuje ani neporovnává s novou konfigurací experimentu. [ExperimentPipeline](/Users/jirka/TSP/src/pipeline/experiment/experiment_pipeline.py:72) provádí preprocessing podle aktuálního configu. Naučené objekty ICA a AutoReject se neukládají; při jejich zapnutí se v experimentu znovu fitují.

**Oprava:** připojit k modelu vynucované vstupní schéma: pořadí kanálů, jednotky, frekvence, okno, reference, preprocessing a jeho verze. Naučené transformace buď uložit, nebo explicitně definovat a vyhodnotit podporovanou adaptaci na nový záznam. Nesoulad odmítnout ještě před inferencí.

**7. P1 — Číselné štítky událostí nejsou stabilně mapovány**

[ParadigmPreprocessor](/Users/jirka/TSP/src/impl/paradigm/paradigm_preprocessing.py:60) z konfiguračního slovníku událostí použije pouze klíče. Číselné identifikátory nechá vytvořit `mne.events_from_annotations(raw)` zvlášť pro každý záznam a až potom filtruje vybrané události.

Ověřený příklad: anotace `left_hand, right_hand` vytvoří mapu `{left_hand: 1, right_hand: 2}`. Přítomnost další anotace `cue` změní mapu na `{cue: 1, left_hand: 2, right_hand: 3}`. Stejné číslo `2` tak označuje pravou ruku v jednom záznamu a levou v druhém. Spojení může narazit na nesprávný počet tříd; při odděleném tréninku a experimentu může dojít k nesprávnému vyhodnocení bez kontroly významu.

Model si pamatuje pouze číselné hodnoty tříd. Výsledek evaluace neobsahuje povinné pořadí tříd pro sloupce pravděpodobností a osy confusion matrix.

**Oprava:** používat jednu explicitní mapu `název události → štítek` pro celý dataset a experiment. Uložit ji do checkpointu a vracet seřazené `classes` s pravděpodobnostmi i maticí záměn. Kontrolovat neznámé a chybějící třídy.

**8. P1 — Loader přijme sémanticky poškozený checkpoint**

[Parsování tříd](/Users/jirka/TSP/src/impl/model/deep_learning/eegnet/load/eegnet_model_loader.py:154) kontroluje jen to, zda jde o seznam nebo `None`. [Obnova mapy](/Users/jirka/TSP/src/impl/model/deep_learning/eegnet/model/eegnet_model.py:199) odmítne prázdný seznam, ale nekontroluje unikátnost, jednorozměrnost, typy ani shodu s `n_classes`.

Ověřené varianty checkpointu pro dvoutřídní síť:

| Změna checkpointu | Skutečné chování |
| --- | --- |
| `classes=None` | Načtení projde; `predict` následně selže |
| `classes=[1]` | Načtení projde; při predikci druhé třídy nastane `IndexError` |
| `classes=[7,7]` | Načtení projde; všechny predikce mají štítek 7 |
| `classes=[1,2,3]` | Načtení projde, přestože síť má dva výstupy |
| NaN v plovoucích vahách | Načtení projde; pravděpodobnosti nejsou konečné |
| `format_version=True` | Je přijato jako verze 1 |
| `input_shape=[6.9,2.8,128.9]` | Je tiše převedeno na `(6,2,128)` |

Chybí také vynucení konzistence názvů modelu v jednotlivých částech checkpointu a validace rozsahu `best_epoch` a `best_validation_accuracy`. Ne všechny chyby obnovy jsou sjednoceny na `InvalidEEGNetCheckpointError`.

**Oprava:** striktní schéma checkpointu, přesně `K` unikátních platných tříd, konečné váhy, přesné celočíselné rozměry a verze bez přijetí `bool`, kontrola souvisejících metadat. Před vrácením modelu provést kontrolní inferenci a ověřit výstupní kontrakt.

**9. P1 při použití pravděpodobnostních metrik — Evaluator počítá jiné veličiny, než config slibuje**

[EvaluationConfig](/Users/jirka/TSP/src/types/dto/config/evaluation_config.py:16) povoluje názvy ze všech sklearn scorerů. [StandardEvaluator](/Users/jirka/TSP/src/impl/evaluator/standard_evaluator.py:83) ale vždy volá interní `_score_func(y_true, y_pred, ...)`, přestože pravděpodobnosti už spočítal.

Pro `y=[0,0,1,1]` a pravděpodobnosti pozitivní třídy `[0.1,0.2,0.3,0.4]` vyšlo **ROC AUC 0.5 místo 1.0**, protože evaluator použil tvrdé predikce. Multiclass AUC může selhat kvůli dimenzi; u scorerů `neg_*` se neaplikuje znaménko scoreru. Povoleny jsou i metriky, které pro tento klasifikační kontrakt nedávají smysl.

Výchozí `accuracy`, `f1_macro`, `precision_macro` a `recall_macro` tento konkrétní problém se vstupem pravděpodobností nemají.

**Oprava:** explicitně podporovat vhodné metriky, pro každou určit požadovaný vstup a způsob agregace, nepoužívat interní API scorerů jako náhradu jejich veřejného kontraktu.

**10. P2 — Poškozené přiřazení štítků se může skrýt při konkatenaci**

[extract_learning_data](/Users/jirka/TSP/src/impl/model/util/extract/extract_learning_data.py:26) ověřuje počet vzorků a štítků až po spojení všech záznamů. Dva záznamy s 2+2 vzorky a 3+1 štítky projdou, protože celkové délky jsou stejné. To bylo reprodukováno. Není ověřeno ani jednorozměrné `y` po jednotlivých záznamech.

**Oprava:** každý záznam samostatně zkontrolovat před spojením: `(N,C,T)`, `y.shape == (N,)`, datový typ, hodnoty, kanály a identifikátor zdroje. Chybová zpráva musí identifikovat vadný záznam.

**11. P2 — Uložení není bezpečné při souběhu a přerušení**

[ArtifactSaver](/Users/jirka/TSP/src/impl/artifacts_saver/artifacts_saver.py:60) vytváří cestu pouze z modelu a času zahájení s přesností na sekundy. Dvě různá `run_id` se stejným časem dostala v testu stejnou cestu. `exist_ok=True` kolizi nezastaví.

[Serializer](/Users/jirka/TSP/src/impl/save_artifacts/deep_learning/eegnet/eegnet_model_serializer.py:49) zapisuje přímo do cílového `.pt`. JSON soubory i manifest se rovněž zapisují postupně. Přerušení zápisu může ponechat neúplný artefakt; souběh může smíchat model, config a metriky různých běhů. Loader neověřuje manifest ani kontrolní součty.

**Oprava:** unikátní cesta obsahující `run_id`, zápis celé sady do dočasného adresáře, validace načtením a teprve potom atomické zveřejnění. Manifest s kontrolními součty a stavem dokončení. Reálný výpadek během zápisu nebyl simulován.

**12. P1 pro automatické spouštění — Konfigurační chyba je hlášena jako úspěšný proces**

[main.py](/Users/jirka/TSP/src/main.py:25) při neplatné konfiguraci pouze zaloguje chybu a provede `return`. Spuštění syntetického DL configu s `model.training.epochs=0` vypisovalo `Validation failed`, ale skončilo **exit kódem 0**.

To je problém pro dávkové zpracování i [CI smoke testy](/Users/jirka/TSP/.github/workflows/ci.yml:65), které sledují úspěch procesu. Neplatný běh může být zelený, aniž by model vznikl.

**Oprava:** konfigurační i runtime selhání vracet jako nenulový exit kód. Smoke test musí ověřit i existenci nového artefaktu, jeho načtení a kontrolní predikci.

**13. P2 — Konfigurace a hraniční vstupy neprocházejí jednotným předběžným ověřením**

[EEGNetConfig](/Users/jirka/TSP/src/types/dto/config/model/model_config.py:14) má `model_name: str` a dědí povolení ignorovat neznámé klíče. Název `typo` i neznámý klíč prošly. Naproti tomu vnořený `TrainingConfig` správně používá `extra="forbid"`. [Random seed](/Users/jirka/TSP/src/types/dto/config/model/training_config.py:17) nemá horní mez; hodnota `2**32` projde konfigurací a následně selže v NumPy. Přičítání `fold_idx` může limit překročit také.

Další ověřené nebo přímo dosažitelné hraniční stavy:

- Prázdný vstup `predict` skončí na `np.concatenate`; prázdné `evaluate` na dělení nulou.
- Dvourozměrný vstup je správně odmítnut, ale nulové osy a minimální délka pro pooling nemají vlastní předběžnou validaci.
- Nezatrénovaný model vrací konečné pravděpodobnosti z náhodných vah; `predict` přitom selže kvůli chybějící mapě tříd.
- Chybějící třída v tréninku je správně odmítnuta. BasicSplitter ale nerozděluje stratifikovaně, takže u malých či nevyvážených dat může vyrobit nepoužitelný fold.
- `CSP` v epochovém preprocessingu vrací 2D příznaky, což je neslučitelné s EEGNet. Kombinace není zakázána v konfiguraci.
- Volba `model_path.backend` není svázána s DL modelem; výchozí obecný loader umí pouze joblib. Pro EEGNet je nutné zvolit registrovaný backend `eegnet` a kořen adresáře artefaktu.
- `split.enabled=false` vrátí data bez validace, ale následný evaluator validaci vyžaduje.

**Oprava:** striktní validace configu i kombinací kroků, kontrola rozdělení před vytvořením sítě a jasný stav připravenosti modelu. Stanovit, zda prázdná inference vrací prázdný výstup, nebo doménovou chybu, a dodržovat to všude.

**14. P2 — Serializace zachová inferenci v běžném případě, ale ne celý životní cyklus**

U platného, krátce natrénovaného modelu byl po uložení a načtení na CPU rozdíl pravděpodobností **0.0** a štítky byly shodné. To je pozitivní výsledek a neukazuje na základní chybu přenosu vah.

Zůstávají tyto konkrétní problémy:

- [evaluate()](/Users/jirka/TSP/src/impl/model/deep_learning/eegnet/model/eegnet_model.py:100) vyžaduje `history`, ale loader historii neobnovuje. Přímé `loaded_model.evaluate(x,y)` po načtení selže. StandardEvaluator používající `predict` tímto konkrétním problémem netrpí.
- [Metadata serializeru](/Users/jirka/TSP/src/impl/save_artifacts/deep_learning/eegnet/eegnet_model_serializer.py:34) jsou libovolné objekty. `metadata={"n": np.int64(6)}` se úspěšně uloží, ale vlastní loader je s `weights_only=True` nedokáže načíst. Současné běžné metadata trenéra používají jednoduché typy; jde o porušení obecného DTO kontraktu.
- Nejsou uloženy optimizer state, RNG state, průběžná pozice tréninku ani historie v checkpointu. Pokračování tréninku po restartu proto není přesné resume.
- TrainingLoop obnovuje nejlepší váhy, ale optimizer v objektu zůstává ze závěrečné epochy. Další trénování stejného objektu tak kombinuje stavy různých epoch.
- Checkpoint obsahuje verzi formátu, ale ne verzi implementace architektury, PyTorch a Braindecode. Manifest může obsahovat Git commit, loader jej však nepoužívá pro kontrolu kompatibility.
- Opakované `fit()` resetuje historii, ale nereinitializuje síť, optimizer a existující mapu tříd. Je potřeba explicitně určit, zda API znamená nové trénování, nebo pokračování.

**Oprava:** oddělit artefakt určený pro inferenci a checkpoint určený pro resume. Pro inferenci nevázat evaluaci na historii, standardizovat metadata na bezpečné primitivní typy a zaznamenat kompatibilitu. Resume podporovat kompletním stavem nebo jej explicitně odmítnout. Absence optimizer state sama o sobě není překážkou čistě inferenčního nasazení.

**15. P2 — Loss a doprovodné metriky nejsou konzistentní**

[Trénovací i evaluační smyčka](/Users/jirka/TSP/src/impl/model/deep_learning/eegnet/model/eegnet_model.py:126) sčítají průměrný loss batchů a dělí počtem batchů. Menší poslední batch má stejnou váhu jako plný. Pro 3 vzorky a batch size 2 bylo reprodukováno **5.346596 místo správného průměru 3.795447**. Accuracy se naopak počítá podle počtu vzorků.

Historie zapisuje validaci do `fold_accuracy` i u finálního modelu; `validation_accuracy` zůstává prázdné. Aggregator hledá klíč `accuracy`, takže jeho fallback neodpovídá těmto názvům. Když chybí validační metrika, může místo ní použít trénovací accuracy nebo nulu. Trenér neukládá `subject_id` do metadat modelu, proto výchozí seskupení aggregatoru skončí pod `unknown`.

Confusion matrix se vytváří bez explicitního seznamu tříd, takže při chybějící třídě může měnit rozměr. DTO neukládá popis os. Volba nejlepší epochy je vždy podle accuracy; konfigurace jiných evaluačních metrik ji nemění.

**Oprava:** vážit loss velikostí batchů; sjednotit názvy, zdroje a význam metrik; chybějící metriku vracet jako chybějící, nikoli zaměňovat za trénovací výsledek nebo nulu. Ukládat třídy a potřebná metadata. Monitorovací metriku pro výběr epochy explicitně definovat.

**16. P2 — Nastavení determinismu závisí na historii procesu**

[set_torch_seed](/Users/jirka/TSP/src/impl/model/deep_learning/reproducibility/set_torch_seed.py:7) zapne deterministické algoritmy jen při `deterministic=True`; volba `False` je zase nevypne. Ověřeno voláním True a následně False: deterministický režim zůstal aktivní. Navíc používá `warn_only=True`.

Při `random_state=None` trenéři funkci vůbec nezavolají, takže ani `deterministic=True` není aplikováno. Globální RNG a globální nastavení PyTorch se mění pro celý proces. To komplikuje paralelní běhy ve stejném procesu a porovnání opakovaných experimentů.

**Oprava:** aplikovat nastavení determinismu nezávisle na přítomnosti seedu a pro oba booleovské stavy. Rozhodnout, zda nepodporovaná operace znamená chybu, nebo pouze varování. Pro souběžné trénování použít izolované procesy nebo vlastní generátory a evidovat parametry prostředí. Bitovou shodu mezi různými zařízeními a verzemi nelze z tohoto CPU testu vyvozovat.

**17. P2 — Zařízení a paměť nejsou řízené explicitně**

[Konstruktor modelu](/Users/jirka/TSP/src/impl/model/deep_learning/eegnet/model/eegnet_model.py:28) automaticky použije CUDA, pokud je dostupná. `EEGNetModelLoader(map_location="cpu")` ovlivňuje načtení tensorů, ale následné sestavení modelu jej může opět přesunout na GPU. Není možné spolehlivě požadovat CPU pouze tímto parametrem loaderu.

Foldové modely včetně optimizerů zůstávají [v seznamu výsledků](/Users/jirka/TSP/src/impl/model/deep_learning/eegnet/trainer/eegnet_model_trainer.py:38); poté vzniká další finální model. Paměť pro modely tak roste s počtem foldů. Každá epocha znovu normalizuje a převádí celý dataset na tensor. Inference také nejprve materializuje celý vstup a StandardEvaluator provede inferenci dvakrát, přes `predict` a znovu pro pravděpodobnosti.

**Oprava:** explicitní `device`, oddělený inferenční objekt bez optimizeru, přesun dokončených foldů na CPU nebo jejich uvolnění, příprava dat mimo epochovou smyčku a dávkové zpracování s limity. Před produkcí změřit RAM/VRAM a latenci na cílovém objemu. Konkrétní GPU OOM zde nebylo reprodukováno.

**18. P2 — Testy a distribuční balíček nezaručují použitelnost v produkci**

Stávajících 40 testů pokrývá především konfiguraci a StageFactory. Chybí automatické testy skutečných predikcí, izolace partition, numerických chyb, načtení po uložení a poškozených checkpointů.

[CI](/Users/jirka/TSP/.github/workflows/ci.yml:51) spouští pouze `pytest tests/unit`, takže soubor [test_deep_learning_config_validation.py](/Users/jirka/TSP/tests/test_deep_learning_config_validation.py:1) mimo tento adresář se do tohoto pytest kroku nezahrne. Smoke konfigurace ověřují průchod tréninku; nemají kontroly, které by odhalily výše popsané úniky dat nebo vadné metriky.

[pyproject.toml](/Users/jirka/TSP/pyproject.toml:9) má prázdné runtime dependencies. Instalace samotného wheelu proto nezajistí potřebné knihovny. Balíček také nezahrnuje adresář Hydra konfigurací podle stávajícího nastavení `include=["src*"]`. Docker používající checkout a requirements je jiný distribuční postup. Soubor nazvaný `requirements.lock.txt` navíc obsahuje neuzamčené položky, např. `skorch`, a rozsah verze Torch, takže nejde o plně reprodukovatelné prostředí.

**Oprava:** rozšířit CI na všechny testy a skutečné kontrakty. Ověřovat DL načtení a predikci i v cílovém distribučním prostředí. Doplnit závislosti a konfigurace do balíčku nebo jednoznačně určit Docker jako podporovaný způsob nasazení. Uzamknout a evidovat testované verze.

**Další scénáře na hranici DL kroku**

Tyto podmínky mohou znehodnotit výsledek i po opravě samotné EEGNet třídy:

- **Raw augmentace před splitem:** [TorchEEGRawAugmentor](/Users/jirka/TSP/src/impl/raw_augmentation/torcheeg_raw_augmentor.py:69) ponechá originál a vytvoří kopie s jiným `session_id`. Náhodné dělení po epochách či dělení podle těchto umělých sessions může rozdělit příbuzné signály mezi trénink a test. Nutná je skupina podle původního záznamu; samotné přejmenování session nezajišťuje nezávislost.
- **Fit preprocessingu před splitem:** [ICA a AutoReject](/Users/jirka/TSP/src/impl/epoch_preprocessing/epoch_preprocessing.py:76) se fitují nad celým záznamem před rozdělením. U splitu po epochách to využívá informaci z pozdějšího testu. Je potřeba rozlišit povolenou kalibraci na záznam a skutečně nezávislou inferenci; případný fit na testových datech musí odpovídat definovanému protokolu.
- **MOABB validace po splitu:** [MoabbSplitter](/Users/jirka/TSP/src/impl/split/moabb_splitter.py:278) sloučí validační vzorky odebrané z různých foldů do globálního poolu. Pool není zaručeně nezávislý na každém foldovém modelu; vzorek může být trénovací v jiném foldu.
- **Malá či nevyvážená data:** při dvou validačních vzorcích není stabilní odhad kvality možný a některé metriky nemají definovanou hodnotu. Kontrolovat zastoupení tříd a dostatečnost množiny, nedovolit zaměnit automaticky dosazenou nulu za validní odhad.
- **Souběžný trénink a inference na stejném objektu:** model přepíná společný modul mezi `.train()` a `.eval()` a mění jeho váhy. Takové sdílení musí být zakázáno nebo řízeno; čistě inferenční instance má mít jasný životní cyklus.
- **Nenatrénovaný nebo špatně fungující model:** náhodný model už dnes vrací pravděpodobnosti. Export proto musí vyžadovat stav úspěšného dokončení a produkční zveřejnění výsledek nezávislého hodnocení vůči domluvenému baseline. Konkrétní akceptovatelnou kvalitu nelze určit pouze ze zdrojového kódu.

**Co už funguje**

Konfigurace tréninku kontroluje kladné epochy a batch size, learning rate, optimizer a záporný seed. EEGNet ověřuje vazbu `pointwise_filters = temporal_filters * depth_multiplier` a počet tříd v tréninku. Neznámé štítky při volání modelového `evaluate` nebo `validate` jsou odmítnuty. Predikce používá eval režim a vypnuté gradienty; nejlepší váhy se kopírují do samostatného snapshotu.

Serializer ukládá `state_dict`, config, mapu tříd a verzi formátu. Loader používá `weights_only=True`, kontroluje základní formát a verzi a nepřepíná automaticky na neomezené načítání pickle. Běžný CPU round trip zachoval výstup přesně. Oprava metadat nesmí tento bezpečnější režim načítání oslabit.

**Minimální podmínky pro nasazení**

1. Trénink, validační výběr a závěrečný test mají doložitelně oddělené vzorky i skupiny původu; finální model používá zamýšlenou augmentaci.
2. Vstup má vynucované schéma `(N,C,T)`, konečné hodnoty, správné jednotky, kanály a preprocessing. Štítky mají stabilní význam.
3. Výstup vždy obsahuje `N` predikcí a matici `(N,K)` s konečnými pravděpodobnostmi a explicitním pořadím tříd.
4. Numerické selhání a neplatný vstup znamenají řízenou chybu. Nenatrénovaný nebo poškozený model se nezveřejní.
5. Evaluace používá správná data i typ vstupu pro každou metriku a pokrývá všechny foldy. Kvalita je ověřena na nezávislých reprezentativních datech.
6. Checkpoint je striktně validovaný, zapisovaný atomicky a svázaný s configem, implementací a vstupním schématem. Po uložení se ověří načtení a shoda predikcí.
7. Neplatný běh má nenulový exit kód; CI ověřuje vytvoření nového použitelného artefaktu.
8. Zvolený způsob distribuce, cílové zařízení, latence a paměťový limit jsou otestované. Resume je buď řádně podporované, nebo oddělené od inferenčního API.

**Ověřovací scénáře pro regresní testy**

| Oblast | Scénáře | Očekávané chování |
| --- | --- | --- |
| Rozměry | 2D, 4D, N=0, C=0, T=0, jiné C, kratší i delší T, N=1 | Doménová chyba nebo explicitně definovaný prázdný výstup; vždy stabilní kontrakt |
| Hodnoty | NaN, ±Inf, přetečení float32, konstantní a nulové kanály, komplexní a nečíselná data | Jasná validace a politika kvality signálu |
| Štítky | Chybějící, různá délka v záznamu, 2D, neznámá třída, chybějící třída, textové štítky | Žádný posun štítků; známé a popsané třídy |
| Signál | Přeházené kanály, jiná sfreq, jednotky, reference nebo okno | Odmítnutí nebo řízená explicitní transformace |
| Split | Duplikáty, augmentované kopie, překryv subject/session, několik foldů | Nulový nepovolený průnik původu dat |
| Trénink | Fold training on/off, augmentace on/off, různé seedy, lichý poslední batch, numerická divergence | Zamýšlená data a správné metriky; poškození přeruší běh |
| Predikce | Opakované volání, různá velikost batchů, před fit a po load | Stejný význam i tvar, definované numerické tolerance |
| Checkpoint | Chybějící/duplicitní třídy, jiné rozměry, neplatné váhy, verze, nekompatibilní architektura, useknutý soubor | Včasné odmítnutí s vysvětlitelnou chybou |
| Uložení | Round trip, souběžné běhy, přerušení zápisu, nedostatek místa, nepodporovaná metadata | Předchozí artefakty zůstávají konzistentní; nový je buď kompletní, nebo nezveřejněný |
| Prostředí | CPU, cílová CUDA, explicitní CPU na GPU hostu, restart procesu, jiné verze | Podporované kombinace procházejí, nepodporované jsou jasně odmítnuty |
| Evaluace | Binární/multiclass, chybějící třída, AUC/log loss, více modelů, axes confusion matrix | Správné vstupy metrik a srovnatelný význam výsledků |
| Provoz | Neplatný config, wheel/Docker, velký dataset, mnoho foldů, restart tréninku | Správný stav běhu, funkční distribuce a předvídatelné zdroje |

**Reprodukce**

U rozboru jsou tři samostatné diagnostické skripty. Jde o sondy dokumentující současné chyby; jejich úspěšný exit kód neznamená, že implementace je správná. Produkční zdrojové soubory nebyly upraveny.

- [model_probe.py](/Users/jirka/TSP/reviews/deep-learning-2026-10-20/model_probe.py) — skutečná EEGNet síť, rozměry, NaN, ukládání a načítání, poškozené checkpointy, konfigurace.
- [integration_probe.py](/Users/jirka/TSP/reviews/deep-learning-2026-10-20/integration_probe.py) — orchestrace s označenými daty, výběr modelů v evaluatoru, AUC, loss, štítky a kolize cest.
- [supporting_probe.py](/Users/jirka/TSP/reviews/deep-learning-2026-10-20/supporting_probe.py) — mapování MNE událostí a přetrvávající nastavení determinismu.

Spuštění z `/Users/jirka/TSP`:

```bash
PYTHONPATH=/Users/jirka/TSP .venv/bin/python reviews/deep-learning-2026-10-20/model_probe.py
PYTHONPATH=/Users/jirka/TSP .venv/bin/python reviews/deep-learning-2026-10-20/integration_probe.py
PYTHONPATH=/Users/jirka/TSP .venv/bin/python reviews/deep-learning-2026-10-20/supporting_probe.py
.venv/bin/python -m pytest -q
```

Ověřený syntetický smoke běh používal `--config-name test/smoke/synthetic/training_dl_fold` a izolovaný pracovní adresář v `/tmp`. Stejný config s `model.training.epochs=0` reprodukoval nesprávný úspěšný exit kód.
