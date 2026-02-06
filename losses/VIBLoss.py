import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VIBLoss(nn.Module):
    def __init__(self, ):
        super().__init__()
    
    def forward(self, model_ouput, label):
        score, mu, logvar = model_ouput
        # 重构损失（预测任务）
        recon_loss = F.mse_loss(score, label, reduction='mean')
        # KL 正则项（压缩约束）
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss , kl_loss

# 线性 annealing（更平滑）
def get_beta(epoch, max_beta=5e-3, anneal_epochs=30):
  """线性增长：早停友好"""
  return min(epoch / anneal_epochs, 1.0) * max_beta
     
