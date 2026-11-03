"""Diagnostic probes for upstream steps; prints observed behavior, not pass/fail."""
import copy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import warnings
import numpy as np
import mne
import yaml
from src.impl.data_loader.FilesystemDatasetLoader import FilesystemDatasetLoader
from src.impl.split.basic_splitter import BasicSplitter
from src.impl.split.moabb_splitter import MoabbSplitter
from src.impl.paradigm.paradigm_preprocessing import ParadigmPreprocessor
from src.impl.epoch_preprocessing.epoch_preprocessing import EpochPreprocessor
from src.impl.raw_preprocessing.raw_preprocessing import RawPreprocessor
from src.impl.augmentation.basic_augmentor import BasicAugmentor
from src.impl.raw_augmentation.torcheeg_raw_augmentor import TorchEEGRawAugmentor
from src.impl.dataset_export.fif_dataset_exporter import FifDatasetExporter
from src.types.dto.config.paradigm_config import ParadigmConfig
from src.types.dto.config.epoch_preprocessing_config import EpochPreprocessingConfig
from src.types.dto.config.raw_preprocessing_config import RawPreprocessingConfig
from src.types.dto.config.augmentation_config import AugmentationConfigBasic
from src.types.dto.config.raw_augmentation_config import RawAugmentationConfigTorchEEG
from src.types.dto.config.split_config import SplitBasicConfig, SplitMoabbWithinSessionConfig
from src.types.dto.load.recording import RecordingDTO
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.split.split_input_dto import SplitInputDTO

ROOT=Path('/Users/jirka/TSP')
mne.set_log_level('ERROR')
warnings.filterwarnings('ignore', category=RuntimeWarning)
class Log:
    def for_step(self,*a): return self
    def __getattr__(self,name): return lambda *a,**kw:None
ctx=NS(logger=Log())
def emit(case, fn):
    try: print(json.dumps({'case':case,'observed':fn()},default=str),flush=True)
    except Exception as e: print(json.dumps({'case':case,'error':type(e).__name__,'message':str(e)[:300]}),flush=True)
def rec(data, session='s',run='r',metadata=None):
    return RecordingDTO(data=data,dataset_name='probe',subject_id=1,session_id=session,run_id=run,metadata=metadata or {})
def pack(r): return EpochPreprocessedDTO(data=r)
def raw_data(desc=('left_hand','right_hand'), sf=200, first=0):
    raw=mne.io.RawArray(np.random.default_rng(42).normal(0,1e-5,(4,2000)),mne.create_info(['C3','C4','F3','F4'],sf,'eeg'),first_samp=first,verbose=False)
    raw.set_annotations(mne.Annotations([2+i for i in range(len(desc))],[0]*len(desc),list(desc)))
    return raw
pcfg=ParadigmConfig(events={'left_hand':10,'right_hand':20},window=dict(tmin=0,tmax=.5,baseline=None),resampling=dict(enabled=True,sfreq=128))
def paradigm(raw,conf=pcfg):
    return ParadigmPreprocessor().run(NS(paradigm_preprocessing_config=conf,data=NS(data=[rec(raw,metadata={'sfreq':200,'channel_names':raw.ch_names})])),ctx).data.data[0]
def epoch_cfg():
    c=yaml.safe_load((ROOT/'configs/epoch_preprocessing/default.yaml').read_text())
    for key in ['alignment','ica','autoreject','csp']: c[key]['enabled']=False
    return c

def none_groups():
    x=np.zeros((10,2,64));y=np.array([1,2]*5)
    r=rec(x,session=None,run=None,metadata={'labels':y})
    out={}
    for splitter in [BasicSplitter(),MoabbSplitter()]:
        a,b,meta=splitter.extract_data([r],ctx) if isinstance(splitter,BasicSplitter) else splitter.extract_data([r])
        out[type(splitter).__name__]=len(splitter.create_dto(np.arange(10),a,b,meta,'probe').data)
    return {'input_samples':10,'reconstructed_recordings':out}
emit('filesystem_missing_session_run',none_groups)
loader=FilesystemDatasetLoader()
emit('instant_events_removed',lambda: str(loader._fit_annotations_to_recording(raw_data(),mne.Annotations([2,3],[0,0],['left_hand','right_hand']))))
def seconds():
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'events.tsv';p.write_text('onset\tduration\ttrial_type\n2\t1\tleft_hand\n')
        a=loader._load_global_annotations(NS(global_events_tsv_path=str(p)))
        return {'input_onset':2,'loaded_onset':a.onset.tolist(),'loaded_duration':a.duration.tolist()}
emit('tsv_time_unit_fixed_to_ms',seconds)
emit('explicit_event_mapping_ignored',lambda: {'requested':pcfg.events,'actual':paradigm(raw_data()).data.event_id,'with_cue':paradigm(raw_data(('cue','left_hand','right_hand'))).data.event_id})
def metadata():
    r=paradigm(raw_data()); x,y,meta=BasicSplitter().extract_data([r],ctx)
    out=BasicSplitter().create_dto(np.arange(len(y)),x,y,meta,'probe')
    with tempfile.TemporaryDirectory() as td:
        FifDatasetExporter()._export_epoch_dto(out,Path(td),'probe')
        saved=mne.read_epochs(next(Path(td).glob('*.fif')),preload=True,verbose=False)
        return {'actual_sfreq':r.data.info['sfreq'],'recording_metadata_sfreq':r.metadata['sfreq'],'split_metadata_keys':list(out.data[0].metadata),'exported_sfreq':saved.info['sfreq'],'exported_channels':saved.ch_names,'original_channels':r.data.ch_names}
emit('stale_signal_metadata_and_export',metadata)
def alignment():
    r=paradigm(raw_data());c=epoch_cfg();c['alignment']={'enabled':True,'tmin_offset':-.2}
    out=EpochPreprocessor().run(NS(epoch_preprocessing_config=EpochPreprocessingConfig.model_validate(c),data=pack([r])),ctx).data.data[0].data
    return {'samples_identical':bool(np.array_equal(r.data.get_data(),out.get_data())),'tmin_before':r.data.tmin,'tmin_after':out.tmin}
emit('alignment_changes_only_axis',alignment)
emit('preload_false',lambda:paradigm(raw_data(),pcfg.model_copy(update={'preload':False})).data.event_id)
def csp_labels():
    info=mne.create_info(['C3','C4','F3','F4'],128,'eeg')
    ep=mne.EpochsArray(np.random.default_rng(4).normal(0,1e-5,(12,4,128)),info,events=np.column_stack([np.arange(12)*200,np.zeros(12,int),[1,2]*6]),verbose=False)
    c=epoch_cfg();c['csp']['enabled']=True;c['csp']['n_components']=2
    out=EpochPreprocessor().run(NS(epoch_preprocessing_config=EpochPreprocessingConfig.model_validate(c),data=pack([rec(ep)])),ctx).data.data[0]
    result={'output_shape':out.data.shape,'labels_present':'labels' in out.metadata}
    try: BasicSplitter().extract_data([out],ctx)
    except Exception as e:result['splitter_error']=str(e)
    return result
emit('csp_loses_labels_and_time_axis',csp_labels)
def aug_labels():
    r=paradigm(raw_data());cfg=AugmentationConfigBasic(backend='basic',enabled=True,copies_per_sample=1)
    out=BasicAugmentor()._augment_single_fold(pack([r]),cfg,ctx.logger).data[0]
    return {'input_events':r.data.events[:,-1].tolist(),'output_samples':len(out.data),'output_metadata':out.metadata}
emit('augmentation_direct_mne_loses_labels',aug_labels)
def noise():
    x=np.random.default_rng(42).normal(0,1e-5,(8,4,128));r=rec(x,metadata={'labels':[1,2]*4})
    cfg=AugmentationConfigBasic(backend='basic',enabled=True,copies_per_sample=1,gaussian_noise={'std':.01})
    out=BasicAugmentor()._augment_single_fold(pack([r]),cfg,ctx.logger).data[0].data
    return {'signal_std':float(x.std()),'added_noise_std':float((out[8:]-x).std()),'noise_to_signal':float((out[8:]-x).std()/x.std())}
emit('noise_scale_for_volt_inputs',noise)
def raw_aug_timing():
    raw=raw_data().crop(tmin=1)
    cfg=RawAugmentationConfigTorchEEG(backend='raw_torcheeg',enabled=True,random_seed=42,copies_per_sample=1,gaussian_noise={'std':0},mask={'prob':0,'ratio':0},sign_flip={'prob':0},scale={'prob':1,'min':1,'max':1})
    result=TorchEEGRawAugmentor().run(NS(config=cfg,data=NS(data=[rec(raw)])),ctx).data.data
    return [{'session':r.session_id,'first_samp':r.data.first_samp,'relative_events':(mne.events_from_annotations(r.data,verbose=False)[0][:,0]-r.data.first_samp).tolist()} for r in result]
emit('raw_aug_nonzero_first_sample_timing',raw_aug_timing)
def ratios():
    x=np.arange(900).reshape(900,1,1);r=rec(x,metadata={'labels':[1,2]*450})
    cfg=SplitBasicConfig(backend='basic',enabled=True,ratios={'train':.7,'validation':.1,'test':.2},shuffle=True,random_seed=42,exclude_validation_data_before_split=False)
    out=BasicSplitter().run(SplitInputDTO(cfg,pack([r])),ctx).data
    count=lambda dto:sum(len(r.data) for r in dto.data)
    return {'requested':[630,90,180],'actual':[count(out.folds[0].train_data),count(out.validation_data),count(out.folds[0].test_data)]}
emit('postsplit_ratio_semantics',ratios)
def moabb_pool():
    cfg=yaml.safe_load((ROOT/'configs/split/moabb_within_session.yaml').read_text());cfg['evaluator']['n_folds']=3
    cfg=SplitMoabbWithinSessionConfig.model_validate(cfg)
    r=rec(np.arange(60).reshape(60,1,1),metadata={'labels':[1,2]*30})
    out=MoabbSplitter().run(SplitInputDTO(cfg,pack([r])),ctx).data
    ids=lambda dto:set(np.concatenate([r.data[:,0,0] for r in dto.data]).tolist())
    val=ids(out.validation_data)
    return {'validation_count':len(val),'train_overlap_by_fold':[len(val&ids(f.train_data)) for f in out.folds],'test_overlap_by_fold':[len(val&ids(f.test_data)) for f in out.folds]}
emit('moabb_global_validation_overlap',moabb_pool)
def csd_ica():
    names=['Fp1','Fp2','F3','F4','C3','C4','P3','P4'];info=mne.create_info(names,128,'eeg');info.set_montage('standard_1020')
    raw=mne.io.RawArray(np.random.default_rng(42).normal(0,1e-5,(8,2560)),info,verbose=False)
    ep=mne.Epochs(raw,np.column_stack([np.arange(5)*300+128,np.zeros(5,int),[1,2,1,2,1]]),tmin=0,tmax=1,baseline=None,preload=True,verbose=False)
    c=epoch_cfg();c['ica']['enabled']=True;c['ica']['n_components']=2
    out=EpochPreprocessor().run(NS(epoch_preprocessing_config=EpochPreprocessingConfig.model_validate(c),data=pack([rec(ep)])),ctx)
    return {'types':raw.get_channel_types(),'output':type(out).__name__}
emit('epoch_ica_without_eog',csd_ica)
def csd_skip():
    cfg=yaml.safe_load((ROOT/'configs/raw_preprocessing/default.yaml').read_text())
    for k,v in cfg.items():
        if isinstance(v,dict) and 'enabled' in v:v['enabled']=False
    cfg['re_referencing']['enabled']=True
    r=raw_data();result=RawPreprocessor().run(NS(raw_preprocessing_config=RawPreprocessingConfig.model_validate(cfg),data=NS(data=[rec(r)])),ctx).data.data[0]
    return {'requested':'CSD','output_channel_types':result.data.get_channel_types(),'samples_identical':bool(np.array_equal(r.get_data(),result.data.get_data()))}
emit('csd_skipped_without_montage',csd_skip)

def raw_ica_identity():
    cfg=yaml.safe_load((ROOT/'configs/raw_preprocessing/default.yaml').read_text())
    for k,v in cfg.items():
        if isinstance(v,dict) and 'enabled' in v:v['enabled']=False
    cfg['ica']={'enabled':True,'n_components':4,'method':'fastica'}
    raw=raw_data()
    out=RawPreprocessor().run(NS(raw_preprocessing_config=RawPreprocessingConfig.model_validate(cfg),data=NS(data=[rec(raw)])),ctx).data.data[0].data
    return {'input_std':float(raw.get_data().std()),'max_difference':float(np.abs(raw.get_data()-out.get_data()).max()),'unchanged_at_1e_12':bool(np.allclose(raw.get_data(),out.get_data(),rtol=0,atol=1e-12))}
emit('raw_ica_excludes_no_components',raw_ica_identity)
