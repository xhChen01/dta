import torch
import torch.nn as nn
from torch.nn.functional import dropout

class DeepDTA(nn.Module):
  def __init__(self):
    