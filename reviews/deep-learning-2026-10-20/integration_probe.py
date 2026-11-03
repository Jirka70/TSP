from types import SimpleNamespace
from datetime import datetime,timezone
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
from src.impl.evaluator.standard_evaluator import StandardEvaluator
from src.types.dto.evaluation.evaluation_input_dto import EvaluationInputDTO
from src.types.dto.config.evaluation_config import EvaluationConfig
from src.types.dto.model.trained_model_dto import TrainedModelDTO
from src.types.dto.load.recording import RecordingDTO
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.split.dataset_split_dto import DatasetSplitDTO,FoldDTO
from src.impl.model.util.extract.extract_learning_data import extract_learning_data
from src.impl.model.deep_learning.eegnet.model.eegnet_model import EEGNetModel
from src.types.dto.config.model.model_config import EEGNetConfig
from src.impl.artifacts_saver.artifacts_saver import ArtifactSaver
from src.pipeline.training.training_pipeline import TrainingPipeline
from src.pipeline.contracts.step_result import StepResult
class Log:
    def for_step(self,*a):return self
    def __getattr__(self,name):return lambda *a,**kw:None
ctx=SimpleNamespace(logger=Log(),run_id='audit',started_at=datetime(2026,10,20,tzinfo=timezone.utc))
def emit(name,value): print(json.dumps({'case':name,'result':value},default=str))
def rec(x,y): return RecordingDTO(data=x,dataset_name='audit',subject_id=1,session_id='s1',run_id='r1',metadata={'labels':y})
x=np.zeros((4,2,128)); y=np.array([0,0,1,1]); data=EpochPreprocessedDTO(data=[rec(x,y)])
class FakeModel:
    def __init__(self,p): self.p=np.array(p); self.calls=0
    def predict(self,x): self.calls+=1; return self.p.argmax(1)
    def predict_class_probability(self,x): return self.p
m1=FakeModel([[.9,.1],[.8,.2],[.7,.3],[.6,.4]]);m2=FakeModel([[.9,.1],[.9,.1],[.1,.9],[.1,.9]])
eval_input=EvaluationInputDTO(config=EvaluationConfig(backend='default',metrics=['roc_auc']),trained_models=[TrainedModelDTO(model=m1,model_name='eegnet',fold_idx=0),TrainedModelDTO(model=m2,model_name='eegnet',fold_idx=1)],dataset_split=DatasetSplitDTO(folds=[FoldDTO(0,data,None),FoldDTO(1,data,None)],validation_data=data))
r=StandardEvaluator().run(eval_input,ctx)
emit('auc_uses_labels_and_only_first_model',{'actual_auc':r.data.metrics['roc_auc'],'correct_probability_auc':roc_auc_score(y,m1.p[:,1]),'model1_calls':m1.calls,'model2_calls':m2.calls})
bad_data=EpochPreprocessedDTO(data=[rec(x[:2],np.array([0,1,0])),rec(x[:2],np.array([1]))])
ex=extract_learning_data(bad_data)
emit('recording_label_count_mismatch',{'accepted_x':ex.x.shape,'accepted_y':ex.y.tolist()})
class Net(nn.Module):
    def __init__(self): super().__init__(); self.dummy=nn.Parameter(torch.zeros(()))
    def forward(self,x): return torch.stack([x[:,0,0],torch.zeros(len(x))],dim=1)+self.dummy
cfg=EEGNetConfig(backend='eegnet',model_name='eegnet',input_normalization='none',fold_training=True,n_classes=2,dropout=.25,kernel_length=16,temporal_filters=2,depth_multiplier=2,pointwise_filters=4,training=dict(epochs=1,batch_size=2,learning_rate=.0005,optimizer='adam'))
loss_model=EEGNetModel(Net(),'eegnet',cfg,(3,1,1));loss_model.initialize_training(np.array([0,1]))
lx=np.array([[[0.]],[[0.]],[[-10.]]]);ly=np.array([0,0,0]);reported=loss_model.evaluate(lx,ly)[0]
expected=nn.CrossEntropyLoss()(loss_model._network(torch.tensor(lx,dtype=torch.float32)),torch.tensor(ly)).item()
emit('batch_loss_averaging',{'reported_loss':reported,'correct_sample_mean':expected})
args=SimpleNamespace(output_path=Path('/tmp/audit'),trained_model=TrainedModelDTO(model=m1,model_name='eegnet'))
saver=ArtifactSaver();p1=saver._create_artifact_output_path(args,ctx);ctx.run_id='another';p2=saver._create_artifact_output_path(args,ctx)
emit('artifact_path_collision',{'same_path_different_run_ids':p1==p2,'path':str(p1)})
# Exercise orchestration with marked data; isolate expensive stages with test doubles.
full=EpochPreprocessedDTO(data=[rec(np.arange(10).reshape(10,1,1),np.array([0,1]*5))])
train=EpochPreprocessedDTO(data=[rec(np.arange(6).reshape(6,1,1),np.array([0,1]*3))])
val=EpochPreprocessedDTO(data=[rec(np.arange(6,8).reshape(2,1,1),np.array([0,1]))])
test=EpochPreprocessedDTO(data=[rec(np.arange(8,10).reshape(2,1,1),np.array([0,1]))])
aug=EpochPreprocessedDTO(data=[rec(np.arange(100,112).reshape(12,1,1),np.array([0,1]*6))])
split=DatasetSplitDTO(folds=[FoldDTO(0,train,test)],validation_data=val)
aug_split=DatasetSplitDTO(folds=[FoldDTO(0,aug,test)],validation_data=val)
class Stage:
    def __init__(self,data): self.data=data;self.input=None
    def run(self,*args):self.input=args[0];return StepResult(self.data)
class Visual:
    def __getattr__(self,name):return lambda *a:None
final=Stage(SimpleNamespace(trained_model=TrainedModelDTO(model=m1,model_name='eegnet')))
fold=Stage(SimpleNamespace(trained_models=[]))
pipe=TrainingPipeline(Stage(full),Stage(full),Stage(full),Stage(full),Stage(full),Stage(split),Stage(aug_split),Stage(None),fold,Stage(None),final,Stage(None),Visual(),Stage(None),None)
conf=SimpleNamespace(**{k:None for k in ['source','raw_preprocessing','raw_augmentation','paradigm','epoch_preprocessing','split','augmentation','dataset_export','model','evaluation','save_artifacts']})
pipe.run(conf,ctx)
fin=extract_learning_data(final.input.training_data).x[:,0,0].tolist()
emit('pipeline_final_training_partition',{'final_ids':fin,'validation_ids':[6,7],'test_ids':[8,9],'fold_augmented_ids':list(range(100,112)),'validation_overlap':len(set(fin)&{6,7}),'test_overlap':len(set(fin)&{8,9})})
