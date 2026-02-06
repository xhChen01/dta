import torch.nn as nn
from .VIBLoss import VIBLoss
def create_criterion(params):
  """
  创建模型实例
  :param params: 模型超参数
  :return: 模型实例
  """
  if(params['model_name'] == 'VIB'):
    return VIBLoss()
  else:
    return nn.MSELoss()
  