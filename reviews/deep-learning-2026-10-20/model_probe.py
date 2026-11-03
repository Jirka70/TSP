import copy, json, tempfile, warnings
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from src.types.dto.config.model.model_config import EEGNetConfig
from src.impl.model.deep_learning.eegnet.builder.eegnet_builder import build_eegnet_model
from src.impl.model.deep_learning.eegnet.load.eegnet_model_loader import EEGNetModelLoader
from src.impl.model.deep_learning.eegnet.trainer.eegnet_training_loop import EEGNetTrainingLoop
from src.impl.model.deep_learning.reproducibility.set_torch_seed import set_torch_seed
from src.impl.save_artifacts.deep_learning.eegnet.eegnet_model_serializer import EEGNetModelSerializer
from src.types.dto.model.trained_model_dto import TrainedModelDTO
from src.types.dto.model.learning_dataset import LearningDataset

torch.set_num_threads(1)
torch.manual_seed(42)
class Log:
    def for_step(self,*a): return self
    def __getattr__(self,name): return lambda *a,**kw: None
ctx=SimpleNamespace(logger=Log(),run_id='audit')
raw=dict(backend='eegnet',model_name='eegnet',input_normalization='per_epoch_channel',fold_training=True,n_classes=2,dropout=.25,kernel_length=16,temporal_filters=2,depth_multiplier=2,pointwise_filters=4,training=dict(epochs=1,batch_size=4,learning_rate=.0005,optimizer='adam',random_state=67,deterministic=True))
cfg=EEGNetConfig.model_validate(raw)
x=np.random.default_rng(42).normal(size=(6,2,128)).astype('float32'); y=np.array([1,2,1,2,1,2])
def show(name, fn):
    try:
        value=fn()
        print(json.dumps({'case':name,'result':value},default=str),flush=True)
    except Exception as e:
        print(json.dumps({'case':name,'exception':type(e).__name__,'message':str(e)[:300],'cause':str(e.__cause__)[:250]}),flush=True)
model=build_eegnet_model(cfg,x.shape)
show('unfit_probabilities',lambda: {'shape':model.predict_class_probability(x).shape,'finite':bool(np.isfinite(model.predict_class_probability(x)).all())})
model.fit(x,y)
show('baseline',lambda: {'shape':model.predict_class_probability(x).shape,'labels':model.predict(x).tolist()})
for shape in [(0,2,128),(2,3,128),(2,2,129),(2,2,256),(2,2,16)]:
    data=np.random.default_rng(43).normal(size=shape).astype('float32')
    show('predict_shape_'+str(shape),lambda data=data: {'prob_shape':model.predict_class_probability(data).shape,'label_shape':model.predict(data).shape})
x_nan=x.copy(); x_nan[0,0,0]=np.nan
show('NaN_inference',lambda: {'probabilities':model.predict_class_probability(x_nan).tolist(),'labels':model.predict(x_nan).tolist()})
show('empty_evaluate',lambda:model.evaluate(x[:0],y[:0]))
with tempfile.TemporaryDirectory(prefix='tsp-audit-') as td:
    root=Path(td)
    EEGNetModelSerializer().save(TrainedModelDTO(model=model,model_name='eegnet'),root/'models')
    loader=EEGNetModelLoader(); restored=loader.load(root,ctx)
    show('roundtrip',lambda:{'max_probability_diff':float(np.abs(model.predict_class_probability(x)-restored.predict_class_probability(x)).max()),'same_labels':bool(np.array_equal(model.predict(x),restored.predict(x)))})
    show('restored_evaluate',lambda:restored.evaluate(x,y))
    ck=torch.load(root/'models/eegnet.pt',weights_only=True)
    for classes in [None,[1],[7,7],[1,2,3]]:
        bad=copy.deepcopy(ck); bad['model_state']['classes']=classes
        torch.save(bad,root/'models/eegnet.pt')
        def run_bad():
            m=loader.load(root,ctx)
            return {'load':'accepted','labels':m.predict(x).tolist(),'prob_shape':m.predict_class_probability(x).shape}
        show('checkpoint_classes_'+str(classes),run_bad)
    bad=copy.deepcopy(ck)
    first=next(k for k,v in bad['model_state']['network_state_dict'].items() if v.is_floating_point())
    bad['model_state']['network_state_dict'][first].fill_(float('nan'))
    torch.save(bad,root/'models/eegnet.pt')
    show('checkpoint_nan_weights',lambda: {'finite_probs':bool(np.isfinite(loader.load(root,ctx).predict_class_probability(x)).all())})
    bad=copy.deepcopy(ck);bad['format_version']=True;bad['model_state']['input_shape']=[6.9,2.8,128.9]
    torch.save(bad,root/'models/eegnet.pt')
    show('checkpoint_bool_version_float_shape',lambda:loader.load(root,ctx).input_shape)
    EEGNetModelSerializer().save(TrainedModelDTO(model=model,model_name='eegnet',metadata={'n':np.int64(6)}),root/'models')
    show('serializer_numpy_metadata',lambda:loader.load(root,ctx).name())
model_nan=build_eegnet_model(cfg,x.shape)
def run_nan_training():
    r=EEGNetTrainingLoop.train(model_nan,LearningDataset(x_nan,y),LearningDataset(x,y),ctx.logger,1)
    return {'best_accuracy':r.best_validation_accuracy,'train_loss':model_nan.history.train_loss,'validation_loss':model_nan.history.validation_loss,'finite_probabilities':bool(np.isfinite(model_nan.predict_class_probability(x)).all())}
show('NaN_training',run_nan_training)
for key,val in [('model_name','typo'),('unknown_key',1)]:
    conf=copy.deepcopy(raw);conf[key]=val
    show('config_'+key,lambda conf=conf: EEGNetConfig.model_validate(conf).model_dump())
conf=copy.deepcopy(raw);conf['training']['random_state']=2**32
show('large_seed',lambda:set_torch_seed(EEGNetConfig.model_validate(conf).training.random_state))
