### TODO: Implement metrics Perplexity, Rouge-L, etc.
###

from config import train_config
import numpy as np
import torch
beta2=train_config.beta2

def Perplexity(sentence):
    m=len(sentence)


    
#查看cuda是否可用
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available")