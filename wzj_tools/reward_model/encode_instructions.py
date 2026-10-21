from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import torch
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

def text_for_row(row, field):
    if field == 'both': return f"Full navigation instruction: {row.get('instruction','')}\nCurrent subtask: {row.get('subtask_text','')}".strip()
    return str(row.get(field,'')).strip()

def key_for_text(text): return hashlib.sha256(text.encode('utf-8')).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True); p.add_argument('--qwen-path',required=True)
    p.add_argument('--output',required=True); p.add_argument('--text-field',choices=['instruction','subtask_text','both'],default='both')
    p.add_argument('--max-length',type=int,default=256); p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    texts={}
    with open(args.manifest) as f:
        for line in f:
            if not line.strip(): continue
            row=json.loads(line); text=text_for_row(row,args.text_field)
            if not text: raise ValueError(f"Empty instruction: {row.get('group_id')}")
            texts[key_for_text(text)]=text
    print(f'[text] unique={len(texts)} field={args.text_field}',flush=True)
    tokenizer=AutoTokenizer.from_pretrained(args.qwen_path,local_files_only=True)
    model=Qwen3VLForConditionalGeneration.from_pretrained(args.qwen_path,torch_dtype=torch.bfloat16,device_map=args.device,local_files_only=True).eval()
    embeddings={}
    with torch.inference_mode():
        for i,(key,text) in enumerate(texts.items(),1):
            inputs=tokenizer(text,return_tensors='pt',truncation=True,max_length=args.max_length)
            inputs={k:v.to(args.device) for k,v in inputs.items()}
            outputs=model(**inputs,output_hidden_states=True,return_dict=True,use_cache=False)
            h=outputs.hidden_states[-1].float()
            mask=inputs['attention_mask'].unsqueeze(-1)
            pooled=(h*mask).sum(dim=1)/mask.sum(dim=1).clamp_min(1)
            embeddings[key]=pooled[0].cpu().to(torch.float16)
            if i%50==0 or i==len(texts): print(f'[encode] {i}/{len(texts)}',flush=True)
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'embeddings':embeddings,'text_field':args.text_field,'qwen_path':args.qwen_path,'dim':next(iter(embeddings.values())).numel()},path)
    print(f'[saved] {path} dim={next(iter(embeddings.values())).numel()}')
if __name__=='__main__': main()
