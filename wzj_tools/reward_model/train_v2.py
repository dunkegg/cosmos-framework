from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from wzj_tools.reward_model.conditioned import ConditionedRankingDataset,InstructionTemporalRewardModel

def score(model,batch,device):
    emb=batch['instruction_embedding'].to(device,non_blocking=True)
    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
        return {k:model(batch[k].to(device,non_blocking=True),emb) for k in ('gt','mild','medium','bad')}

@torch.no_grad()
def evaluate(model,loader,device):
    model.eval(); n=0; counts={k:0 for k in ('mild','medium','bad')}; top1=0
    for batch in loader:
        r=score(model,batch,device); bs=len(r['gt']); n+=bs
        for k in counts: counts[k]+=(r['gt']>r[k]).sum().item()
        top1+=(torch.stack([r[k] for k in ('gt','mild','medium','bad')],dim=1).argmax(dim=1)==0).sum().item()
    return {'acc_overall':sum(counts.values())/(3*n),'gt_top1':top1/n,**{f'acc_{k}':v/n for k,v in counts.items()},'num_groups':n}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True); p.add_argument('--embeddings',required=True)
    p.add_argument('--output-dir',required=True); p.add_argument('--init-v1',default='')
    p.add_argument('--hidden-dim',type=int,default=256); p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--epochs',type=int,default=30); p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--weight-decay',type=float,default=1e-4); p.add_argument('--num-workers',type=int,default=4)
    p.add_argument('--val-ratio',type=float,default=0.1); p.add_argument('--split-seed',type=int,default=42)
    p.add_argument('--mild-weight',type=float,default=0.5); p.add_argument('--medium-weight',type=float,default=1.0); p.add_argument('--bad-weight',type=float,default=1.0)
    args=p.parse_args(); torch.manual_seed(args.split_seed)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    def make_dataset(split): return ConditionedRankingDataset(embedding_path=args.embeddings,manifest_path=args.manifest,split=split,val_ratio=args.val_ratio,split_seed=args.split_seed)
    train_ds=make_dataset('train'); val_ds=make_dataset('val')
    train=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,pin_memory=True,persistent_workers=args.num_workers>0)
    val=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=True,persistent_workers=args.num_workers>0)
    device=torch.device('cuda'); model=InstructionTemporalRewardModel(instruction_dim=train_ds.embedding_dim,hidden_dim=args.hidden_dim).to(device)
    if args.init_v1:
        ckpt=torch.load(args.init_v1,map_location='cpu',weights_only=False)
        missing,unexpected=model.load_state_dict(ckpt['model'],strict=False)
        assert set(missing)=={'condition_norm.weight','condition_norm.bias','film.weight','film.bias'},(missing,unexpected)
        assert not unexpected,unexpected
        print(f'[init] loaded V1 epoch={ckpt.get("epoch")} missing={missing}')
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    weights={'mild':args.mild_weight,'medium':args.medium_weight,'bad':args.bad_weight}; best=-1
    for epoch in range(1,args.epochs+1):
        model.train(); loss_sum=0; count=0
        for batch in train:
            optimizer.zero_grad(set_to_none=True); r=score(model,batch,device)
            loss=sum(w*(-F.logsigmoid(r['gt'].float()-r[k].float()).mean()) for k,w in weights.items())/sum(weights.values())
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
            bs=len(r['gt']); loss_sum+=loss.item()*bs; count+=bs
        metrics=evaluate(model,val,device)
        print(f'[epoch] {epoch} train_loss={loss_sum/count:.6f} '+ ' '.join(f'{k}={v:.4f}' for k,v in metrics.items()),flush=True)
        checkpoint={'epoch':epoch,'model':model.state_dict(),'args':vars(args),'instruction_dim':train_ds.embedding_dim,'val_metrics':metrics}
        torch.save(checkpoint,out/'latest.pt')
        if metrics['acc_overall']>best:
            best=metrics['acc_overall']; torch.save(checkpoint,out/'best.pt'); print(f'[best] epoch={epoch} val_acc={best:.4f}',flush=True)
        with (out/'metrics.jsonl').open('a') as f: f.write(json.dumps({'epoch':epoch,'train_loss':loss_sum/count,**metrics})+'\n')
if __name__=='__main__': main()
