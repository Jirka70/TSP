# Připnuté závislosti

Hlavní prostředí používá přesné verze v `requirements.lock.txt`, včetně nepřímých závislostí.
`requirements.txt` tento soubor zahrnuje a přidává volitelné Qt pro desktopové grafy.
Instalační a build nástroje jsou připnuté v `requirements.tools.txt` a build backend v `pyproject.toml`.

## Instalace jako v CI

V čistém virtuálním prostředí s Pythonem 3.11:

```sh
python -m pip install -r requirements.tools.txt
python -m pip install --build-constraint requirements.tools.txt -r requirements.lock.txt
python -m pip check
python -m pip install --build-constraint requirements.tools.txt -r requirements.torcheeg.txt --no-deps
python -c "import torch, torchaudio; from braindecode.models import EEGNet"
python -m pytest tests
```

CI i Docker používají oficiální linuxové CPU wheels `torch==2.11.0+cpu` a `torchaudio==2.11.0+cpu`.
Tím se sjednotí binární varianta a neinstalují se nepotřebné CUDA knihovny na CPU runner.
Na macOS se používají běžné wheels `torch==2.11.0` a `torchaudio==2.11.0`.
Braindecode zůstává na verzi 1.3.2 a skorch na 1.3.1.

Oficiální [instalační dokumentace TorchAudio](https://docs.pytorch.org/audio/stable/installation.html)
potvrzuje kompatibilitu TorchAudio 2.11 s PyTorch 2.11.
Po změně této dvojice je nutné společně ověřit import Braindecode, testy, trénování a export/načtení EEGNetu.

## Výjimka TorchEEG

TorchEEG 1.1.3 se nadále instaluje samostatně přes `--no-deps`.
Projekt používá jeho transformace; potřebné knihovny pro tuto cestu jsou připnuté v hlavním lock souboru.
Plná instalace podle metadat TorchEEG není slučitelná s aktuálním prostředím:
TorchEEG vyžaduje SciPy nejvýše 1.10.1, které nepodporuje používané NumPy 2.4.3.
Navíc zde nejsou instalované nepoužívané závislosti `lmdb`, `mne-connectivity`, `pytorch-lightning`,
`torchmetrics` a `xlrd`. `pip check` se proto provádí před instalací TorchEEG;
po ní hlásí tuto známou výjimku. Nejde o plně vyřešenou kompatibilitu celého balíčku TorchEEG.

Při aktualizaci verzí nejprve řeš hlavní prostředí bez TorchEEG, připni také nové nepřímé závislosti
a spusť `pip check`. Potom přidej TorchEEG přes `--no-deps` a ověř všechny testy a smoke konfigurace.
Nepoužívej samostatné instalační příkazy s nepřipnutými názvy balíčků.

## Ověření 2026-11-11

- Při původním ověření prošlo 154 pytest testů na macOS/Pythonu 3.12 i v čistém Linux amd64/Pythonu 3.11.
- Na obou platformách prošly syntetický DL a MOABB DL smoke scénář.
  MOABB používá BNCI2014_001, osobu 1; DL běh zahrnuje uložení a načtení EEGNetu.
- V čistém Linuxu prošel `pip check` hlavního prostředí, import EEGNetu a build wheelu i sdist.
  Všechny skutečně nainstalované runtime závislosti mají explicitní pin.
- Ruff prošel. Celoplošný mypy zůstává neúspěšný.

Změny konfigurací splitterů, evaluačního DTO a jejich dva nové regresní testy byly na zadání
uživatele vráceny. Dřívější úspěšné ML smoke běhy zahrnovaly tyto dočasné opravy;
nevztahují se tedy na výsledný stav. Zastaralé cesty splitterů a nesoulad foldové evaluace
zůstávají k samostatnému řešení mimo opravu verzí knihoven.

Opakované spuštění pytest po vrácení těchto změn zastavil import v `stage_factory.py`:
`ModuleNotFoundError` pro `eegnet_model_trainer`. Aktuální testovací sada tedy neprošla sběrem testů;
tato další chyba nebyla v rámci vrácení změn opravována. Ruff a kontrola diffu prošly.
