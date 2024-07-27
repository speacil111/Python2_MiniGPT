import os

import torch
import numpy as np

train_data = None
val_data = None

def init_data_pretrain(dataset):
    global train_data, val_data
    
    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def init_data_sft(dataset):
    global train_data, val_data
    
    ### TODO: 读取+初始化sft数据
    data_path=os.path.join('data', dataset)
    train_data=np.memmap(os.path.join(data_path,'train.bin'),dtype=np.uint16,mode='r')
    val_data=np.memmap(os.path.join(data_path,'val.bin'),dtype=np.uint16,mode='r')
    ###

def get_batch_pretrain(split, batch_size, block_size, device):
    global train_data, val_data
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    loss_mask = torch.ones_like(x, dtype=torch.float64)
    
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y, loss_mask = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True), loss_mask.pin_memory().to(device, non_blocking=True)
    else:
        x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)
    return x, y, loss_mask
    
def get_batch_sft(split,block_size,batch_size,device): 
    ### TODO: 获取sft数据的批次（batch）+ 构建损失函数掩码（loss_mask）
    q_max,a_max=64,256
    length=q_max+a_max
    global train_data, val_data
    data=train_data if split=='train' else val_data
    ix = torch.randint(int(len(data) /length) - 1, (batch_size,))
    x=torch.zeros(len(ix),block_size,dtype=torch.int64)
    y=torch.zeros(len(ix),block_size,dtype=torch.int64)
    loss_mask=torch.zeros(len(ix),block_size,dtype=torch.float64)
    for i in range(len(ix)):
        data_line=data[ix[i]*length:(ix[i]+1)*length]
        ques=data_line[:q_max]
        ans=data_line[q_max:]
        #去除所有填充符[50256]
        ques=[x for x in ques if x!=50256]
        ans=[x for x in ans if x!=50256]
        len_ques,len_ans=len(ques),len(ans)
        #合并，填充
        combined=ques+ans+[50256]*(length-len(ques)-len(ans))
        x[i]=torch.tensor(combined[:block_size],dtype=torch.int64)
        y[i]=torch.tensor(combined[1:block_size+1],dtype=torch.int64)
        #ans处1
        for j in range(block_size):
            if j>=len_ques and j<(len_ques+len_ans) :
                loss_mask[i][j]=1
    device_type='cuda' if 'cuda' in device else 'cpu'
    if device_type == 'cuda':
        x, y, loss_mask = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True), loss_mask.pin_memory().to(device, non_blocking=True)
    else:
        x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)
    return x, y, loss_mask