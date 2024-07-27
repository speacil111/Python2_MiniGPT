import os
import sys
import tiktoken
import numpy as np
import json
from sklearn.model_selection import train_test_split

enc = tiktoken.get_encoding("gpt2")

names = sys.argv[1:]

### TODO: read data from ([name].jsonl for name in names)
### TODO: combine multiple files(if needed) into one single data file
### TODO: split data for train(0.9) and valid (0.1)
texts=[]
script_dir = os.path.dirname(__file__)

for name in names:
    file_path=os.path.join(script_dir, "dataset", f"{name}.jsonl")
    with open(file_path,'r',encoding='utf-8') as file:
        for line in file:
            texts.append(json.loads(line)["text"])
for text in texts:
    text = text.encode('utf-8', 'ignore').decode('utf-8')
    text = text.replace("\n", "")
train_data, val_data = train_test_split(texts,test_size=0.1)
###
train_data, val_data = "<|endoftext|>".join(train_data), "<|endoftext|>".join(val_data)
train_data, val_data =enc.encode(train_data,allowed_special={'<|endoftext|>'}),enc.encode(val_data,allowed_special={'<|endoftext|>'})
### TODO: tokenize raw data with tiktoken encoder
### TODO: transform to numpy array
train_ids, val_ids = np.array(train_data,dtype=np.uint16),np.array(val_data,dtype=np.uint16)
###

# save numpy array to file [name]/train.bin and [name]/val.bin
train_ids.tofile(os.path.join("processed_pretrain", "train.bin"))
val_ids.tofile(os.path.join("processed_pretrain", 'val.bin'))
print("successfully saved!")