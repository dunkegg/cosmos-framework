import argparse,torch
from torch.utils.data import DataLoader
from wzj_tools.reward_model.conditioned import ConditionedRankingDataset,InstructionTemporalRewardModel
from wzj_tools.reward_model.train_v2 import evaluate
p=argparse.ArgumentParser()
p.add_argument('--checkpoint',required=True); p.add_argument('--manifest',required=True); p.add_argument('--embeddings',required=True)
p.add_argument('--batch-size',type=int,default=8); p.add_argument('--num-workers',type=int,default=4)
a=p.parse_args(); ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False); config=ckpt['args']
dataset=ConditionedRankingDataset(embedding_path=a.embeddings,manifest_path=a.manifest,split='val',val_ratio=config['val_ratio'],split_seed=config['split_seed'])
model=InstructionTemporalRewardModel(instruction_dim=ckpt['instruction_dim'],hidden_dim=config['hidden_dim']).cuda().eval()
model.load_state_dict(ckpt['model'],strict=True)
loader=DataLoader(dataset,batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers,pin_memory=True)
print(f'[checkpoint] epoch={ckpt["epoch"]} saved={ckpt["val_metrics"]}')
print(f'[eval] {evaluate(model,loader,torch.device("cuda"))}')
