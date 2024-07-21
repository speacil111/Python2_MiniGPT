### TODO: prepare SFT data similar to `prepare.py`
###

import os
import sys
import tiktoken
import numpy as np
import json
from sklearn.model_selection import train_test_split


q_max,a_max=64,256
enc = tiktoken.get_encoding("gpt2")

### TODO: read data from ([name].jsonl for name in names)
### TODO: combine multiple files(if needed) into one single data file
### TODO: split data for train(0.9) and valid (0.1)
combined_texts=[]
script_dir = os.path.dirname(__file__)
file_path=os.path.join(script_dir, "dataset", f"sft_data.jsonl")
with open(file_path,'r',encoding='utf-8') as file:
    for line in file:
        data=json.loads(line)
        ques=enc.encode(data["question"])
        ans=enc.encode(data["answer"])
        if len(ques)>q_max:
            ques=ques[:q_max]
        else:
            ques+=[50256]*(q_max-len(ques))
        if len(ans)>a_max:
            ans=ans[:a_max]
        else:  
            ans+=[50256]*(a_max-len(ans))
        combined_texts.append(ques+ans)
train_data, val_data = train_test_split(combined_texts,test_size=0.1)
###

### TODO: tokenize raw data with tiktoken encoder
### TODO: transform to numpy array
train_ids, val_ids = np.array(train_data,dtype=np.uint16),np.array(val_data,dtype=np.uint16)
###

# save numpy array to file [name]/train.bin and [name]/val.bin
train_ids.tofile(os.path.join("processed_pretrain_sft", "train.bin"))
val_ids.tofile(os.path.join("processed_pretrain_sft", 'val.bin'))
print("successfully saved!")