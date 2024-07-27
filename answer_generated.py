import os
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, MiniGPT
import json
import re
# -----------------------------------------------------------------------------

out_dir = 'sft_10w' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 256 # number of tokens generated in each sample
temperature = 0.4 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 100 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1234
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file

# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# init from a model saved in a specific directory
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
config = GPTConfig(**checkpoint['model_args'])
model = MiniGPT(config)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)


questions=[]
answers=[] 
with open('2022010583.jsonl','w',encoding='utf-8') as file:
    pass
    file.close()
with open('测试集-day2.jsonl', 'r', encoding='utf-8') as f:
    try:
        for line in f:
            data = json.loads(line)
            ques=data['question']
            questions.append(ques)
            start_ids = encode(ques)
            x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
            with torch.no_grad():
                with ctx:
                    for k in range(num_samples):
                        output=""
                        t=0
                        while not output and t<=10:
                            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                            output_tokens = y[0].tolist()
                            try:
                                end_idx = output_tokens.index(50256)
                                output_tokens = output_tokens[:end_idx]
                            except:
                                pass
                            output = decode(output_tokens)
                            t+=1
            output=output[len(ques):]
            output=re.sub('[\uFFFD\n]', '', output)
            print(output)
            answers.append(output)
            dict = {'question':ques,'answer':output}
            with open('2022010583.jsonl','a',encoding='utf-8') as file:
                json.dump(dict, file, ensure_ascii=False)
                file.write('\n')
    except Exception as e:
        print(f"发生错误: {e}")
print("回答完毕！")
