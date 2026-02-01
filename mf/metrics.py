from matplotlib.pylab import f
import torch
import numpy as np



def evaluate_mse(model, loader, device):
  model.eval()
  total_mse, n = 0, 0
  with torch.no_grad():
    for drugs, targets, affinities in loader:
      drugs, targets, affinities = drugs.to(device), targets.to(device), affinities.to(device)
      preds = model(drugs, targets)
      total_mse += torch.sum((preds - affinities)**2).item()
      n += len(affinities)
  return total_mse / n

def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / (float(y_obs_sq * y_pred_sq) + 0.00000001)

def get_k(y_obs, y_pred):
  y_obs = np.array(y_obs)
  y_pred = np.array(y_pred)

  return sum(y_obs * y_pred) / (float(sum(y_pred * y_pred)) + 0.00000001)


def squared_error_zero(y_obs, y_pred):
  k = get_k(y_obs, y_pred)

  y_obs = np.array(y_obs)
  y_pred = np.array(y_pred)
  y_obs_mean = [np.mean(y_obs) for y in y_obs]
  upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
  down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

  return 1 - (upp / (float(down) + 0.00000001))

def get_rm2(ys_orig, ys_line):
  # 计算 r_m^2 
  r2 = r_squared_error(ys_orig, ys_line)
  r02 = squared_error_zero(ys_orig, ys_line)
  return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))

def get_cindex(y_true, y_pred):
  # 计算 CI
  y_true = np.array(y_true)
  y_pred = np.array(y_pred)
  
  # 构建可比较矩阵
  gt_mask = y_true.reshape((1, -1)) > y_true.reshape((-1, 1))
  # 计算预测值差值
  diff = y_pred.reshape((1, -1)) - y_pred.reshape((-1, 1))
  # 排序判断
  h_one = (diff > 0)
  h_half = (diff == 0)
  
  # 计算分子和分母
  numerator = np.sum(gt_mask * h_one * 1.0 + gt_mask * h_half * 0.5)
  denominator = np.sum(gt_mask)
  
  # 异常处理：分母为0（所有真实值相同）
  if denominator == 0:
      return np.nan  # 或返回1.0（根据场景选择）
  return numerator / denominator


def get_mse(y_true, y_pred):
  mse = ((y_true - y_pred) ** 2).mean(axis=0)
  return mse

def evaludate_metrics(model, loader, device, metrics):
  
  model.eval()
  metric_results = {}
  full_preds, full_affinities = [], []
  with torch.no_grad():
    for drugs, targets, affinities in loader:
      drugs, targets, affinities = drugs.to(device), targets.to(device), affinities.to(device)
      preds = model(drugs, targets)
      full_preds.append(preds.cpu().numpy())
      full_affinities.append(affinities.cpu().numpy())

    full_preds = np.concatenate(full_preds, axis=0)
    full_affinities = np.concatenate(full_affinities, axis=0)
  
    for metric in metrics:
      if metric == 'mse':
        metric_results[metric] = get_mse(full_affinities, full_preds)
      elif metric == 'rm2':
        metric_results[metric] = get_rm2(full_affinities, full_preds)
      elif metric == 'cindex':
        metric_results[metric] = get_cindex(full_affinities, full_preds)
      else:
        raise ValueError(f"Unknown metric: {metric}")
      
  return metric_results
