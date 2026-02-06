from pyexpat import model
from models import MatrixFactorization, DeepDTA


def create_model(model_name, data, params):
  """
  创建模型实例
  :param model_name: 模型名称
  :param params: 模型超参数
  :return: 模型实例
  """
  if(model_name == 'MF'):
    params['n_drugs'] = data['n_drugs']
    params['n_targets'] = data['n_targets']
    return MatrixFactorization(**params)
  elif(model_name == 'DeepDTA'):
    return create_deepdta(data, params)
  else:
    raise ValueError(f"Unknown model name: {model_name}")
  
def create_deepdta(data, params):
  """
  创建DeepDTA模型实例
  :param data: 数据集相关信息
  :param params: 模型超参数
  :return: DeepDTA模型实例
  """
  params['ligands_features'] = data['ligands_features']
  params['proteins_features'] = data['proteins_features']
  params['max_smi_len'] = data['max_smi_len']
  params['max_seq_len'] = data['max_seq_len']
  params['charsmiset_size'] = data['charsmiset_size']
  params['charseqset_size'] = data['charseqset_size']
  return DeepDTA(**params)