import json
import numpy as np
import mne
import torch
from src.impl.model.deep_learning.reproducibility.set_torch_seed import set_torch_seed

def event_map(desc):
    raw=mne.io.RawArray(np.zeros((1,400)),mne.create_info(['Cz'],100,['eeg']),verbose=False)
    raw.set_annotations(mne.Annotations(np.arange(len(desc),dtype=float),np.zeros(len(desc)),desc))
    return mne.events_from_annotations(raw,verbose=False)[1]
print(json.dumps({'case':'event_id_mapping','base':event_map(['left_hand','right_hand']),'extra_annotation':event_map(['cue','left_hand','right_hand'])}))
set_torch_seed(67,True); before=torch.are_deterministic_algorithms_enabled()
set_torch_seed(67,False); after=torch.are_deterministic_algorithms_enabled()
print(json.dumps({'case':'determinism_sticky','after_true':before,'after_false':after,'warn_only':torch.is_deterministic_algorithms_warn_only_enabled()}))
