# TSP 1/2

## Project structure
```
repo/
├─ configs/ (konfigurační soubory pro hydru)
│  ├─ config.yaml
│  ├─ preprocessing/
│  │  ├─ mne.yaml
│  │  └─ moabb.yaml
│  ├─ augmentation/
│  │  ├─ torch.yaml
│  │  └─ none.yaml
│  ├─ model/
│  │  ├─ csp_svm.yaml
│  │  └─ eegnet.yaml
│  └─ visualization/
│     ├─ mne_plots.yaml
│     └─ matplotlib.yaml
│
└─ src/
   ├─ types/
   │  ├─ dto/ (vstupní datové struktury pro každý krok pipeline)
   │  │  ├─ dto_preprocessing.py
   │  │  ├─ dto_augmentation.py
   │  │  ├─ dto_model.py
   │  │  └─ dto_visualization.py
   │  │
   │  └─ interfaces/
   │     ├─ preprocessing.py
   │     ├─ augmentation.py
   │     ├─ model.py
   │     ├─ visualization.py
   │     └─ validator.py
   │
   ├─ impl/ (implementace pro různé knihovny daných bloků)
   │  ├─ preprocessing/
   │  │  ├─ mne_preprocessor.py (implementuje types/interfaces/preprocessing)
   │  │  └─ moabb_preprocessor.py
   │  │
   │  ├─ augmentation/
   │  │  ├─ none_augmentor.py
   │  │  └─ torch_augmentor.py
   │  │
   │  ├─ model/
   │  │  ├─ eegnet_model.py
   │  │  └─ sklearn_csp_svm.py
   │  │
   │  └─ visualization/
   │     ├─ matplotlib_viz.py
   │     └─ mne_visualizer.py
   │
   ├─ validation/ (validace jednotivých config souborů hydry)
   │  ├─ preprocessing/
   │  │  ├─ mne.py (validátor pro mne config.yaml)
   │  │  └─ moabb.py
   │  │
   │  ├─ augmentation/
   │  │  ├─ none.py
   │  │  └─ torch.py
   │  │
   │  ├─ model/
   │  │  ├─ eegnet.py
   │  │  └─ csp.py
   │  │
   │  └─ visualization/
   │     ├─ matplotlib.py
   │     └─ mne.py
   │
   └─ main.py
```
### src/validation
Obsahuje validátor pro každý hydra config, kontroluje:
- existenci parametrů
- typy
- strukturu configu
- logickou konzistenci parametrů
- hodnotové rozsahy
- existence vstupních souborů

### src/types/dto
DTO definují datové struktury, které jsou vstupem pro každý krok pipeline (viz `src/impl`). Umožní jednodušší testování, případnou serializaci do .jsonu / Pickle. Každá struktura obsahuje:
- parametry
- seed
- dataset

### src/types/interfaces
Obecná rozhraní, která definují chování implementace v daném kroku pipeline. Pro každou knihovnu existuje adaptér, který implementuje dané rozhraní. Pipeline pracuje s rozhraním jako závislost a je kompletně oddělena od vnějších knihoven.

## Pipeline
```
Hydra Config (config.yaml)
     │
     ▼
Config Schema Validation (validation for each config input)
     │
     ▼
Preprocessing
     │
     ▼
Augmentation (optional)
     │
     ▼
Model / Detection
     │
     ▼
Visualization
```