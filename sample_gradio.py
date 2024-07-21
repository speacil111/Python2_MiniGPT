### TODO: build gradio app based on sample.py
###
import gradio as gr
import os
from contextlib import nullcontext
import torch
import tiktoken
import time
from model import GPTConfig, MiniGPT
import requests

# -----------------------------------------------------------------------------

out_dir = 'pretrained2' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 216 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1234
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file

# -----------------------------------------------------------------------------
save_path = os.path.join(out_dir, 'samples')

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

def generate_text(start):
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    # run generation
    with torch.no_grad():
        with ctx:
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            output_tokens = y[0].tolist()
            try:
                end_idx = output_tokens.index(50256)
                output_tokens = output_tokens[:end_idx]
            except:
                pass
            output = decode(output_tokens)
    
    output=output[len(start):]
    for i in range(len(output)):
        time.sleep(0.05)
        yield output[: i+1]


def generate_text_arena(start,max_new_tokens, temperature, top_k):
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    # run generation
    with torch.no_grad():
        with ctx:
            y = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
            output_tokens = y[0].tolist()
            try:
                end_idx = output_tokens.index(50256)
                output_tokens = output_tokens[:end_idx]
            except:
                pass
            output = decode(output_tokens)
    
    output=output[len(start):]
    return output

chatbot=gr.Interface(fn=generate_text,

                    inputs= gr.Textbox(label="开始文本",lines=5,
                                    placeholder="为什么不问问神奇海螺呢？"),
                    theme='soft',
                    title="神奇海螺",
                    outputs=gr.Textbox(label="回答文本",lines=7),
                    allow_flagging='never'
                    )


if __name__ == '__main__':
    chatbot.launch()



