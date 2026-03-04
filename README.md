# TSP-1-2---Modul-rn-software-pro-klasifikaci-elektrofyziologick-ch-dat
# some chang

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
