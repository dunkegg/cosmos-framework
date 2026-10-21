from __future__ import annotations
import hashlib
import torch
from torch import nn
from wzj_tools.reward_model.latent_dataset import RankingLatentDataset
from wzj_tools.reward_model.reward_model import TemporalRewardModel

def text_for_row(row,field):
    if field=='both': return f"Full navigation instruction: {row.get('instruction','')}\nCurrent subtask: {row.get('subtask_text','')}".strip()
    return str(row.get(field,'')).strip()

class ConditionedRankingDataset(RankingLatentDataset):
    def __init__(self,embedding_path,**kwargs):
        super().__init__(**kwargs)
        d=torch.load(embedding_path,map_location='cpu',weights_only=False)
        self.embedding_dim=d['dim']; self.text_field=d['text_field']
        self.embeddings=d['embeddings']
        for row in self.rows:
            key=hashlib.sha256(text_for_row(row,self.text_field).encode('utf-8')).hexdigest()
            if key not in self.embeddings: raise KeyError(f'Missing instruction embedding for {row["group_id"]}')
            row['_instruction_key']=key
    def __getitem__(self,idx):
        sample=super().__getitem__(idx)
        sample['instruction_embedding']=self.embeddings[self.rows[idx]['_instruction_key']].float()
        return sample

class InstructionTemporalRewardModel(TemporalRewardModel):
    def __init__(self,instruction_dim,**kwargs):
        super().__init__(**kwargs)
        hidden_dim=kwargs.get('hidden_dim',256)
        self.condition_norm=nn.LayerNorm(instruction_dim)
        self.film=nn.Linear(instruction_dim,2*hidden_dim)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
    def forward(self,z,instruction_embedding):
        if self.drop_t0: z=z[:,:,1:]
        z=z.mean(dim=(-1,-2)).transpose(1,2)
        z=self.temporal_encoder(self.proj(self.input_norm(z)))
        gamma,beta=self.film(self.condition_norm(instruction_embedding.float())).chunk(2,dim=-1)
        z=z*(1+gamma.unsqueeze(1))+beta.unsqueeze(1)
        attn=torch.softmax(self.attn_pool(z),dim=1)
        return self.head((z*attn).sum(dim=1)).squeeze(-1)
