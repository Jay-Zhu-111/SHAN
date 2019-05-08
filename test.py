import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import heapq

shan = torch.load('SHAN.pkl')
print(shan)
torch.save(shan.state_dict(), 'SHAN_dict.pkl')
