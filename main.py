from datetime import datetime
import numpy as np
import argparse
from data_process import *
from sklearn.model_selection import KFold
from operator import itemgetter
from torch.utils.data import DataLoader
from data_loader import AffinityDataset
from torch import optim
import torch.nn as nn
import logging
import pandas as pd
import itertools
from metrics import evaludate_metrics, evaluate_mse
import torch
import random
import time
from tqdm import tqdm
from models import create_model
from losses.create_criterion import create_criterion
from losses.VIBLoss import get_beta

# 设置全局随机种子，确保结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 主日志文件夹
LOG_FOLDER = 'logs'

def load_logging():
  global LOG_FOLDER

  current_date = datetime.now().strftime('%Y-%m-%d')
  timestamp = datetime.now().strftime('%H-%M-%S')
  LOG_FOLDER = os.path.join(LOG_FOLDER, current_date, timestamp)
  if not os.path.exists(LOG_FOLDER):
      os.makedirs(LOG_FOLDER)

  log_filename = os.path.join(LOG_FOLDER, 'results.log')

  # 配置日志，同时输出到控制台和文件
  logging.basicConfig(
      level=logging.INFO,
      format='%(message)s',  # 添加时间戳
      handlers=[
          logging.FileHandler(log_filename, mode='w'),
          logging.StreamHandler()
      ]
  )
# -------------------------- 2. 核心函数：绘制单个参数的分析图 --------------------------



def create_statistics_file(params_name, results, metrics):
  global LOG_FOLDER

  # 1. 将结果保存到 Exce l中
  excel_path = os.path.join(LOG_FOLDER, 'param_search_results.xlsx')
  df_results = pd.DataFrame(results)
  df_results.to_excel(excel_path, index=False, engine='openpyxl')
  
  # 2. 根据统计结果生成折线图，显示不同超参数对性能的影响
  # plot_param_search_results(df_results, params_name, metrics)
  
def train_epoch_with_validation(epoch, model, train_loader, val_loader, criterion, optimizer, device):
  """训练一个epoch并返回验证MSE用于早停"""
  model.train()
  
  total_mse_loss = 0
  total_kl_loss = 0
  # 为数据加载添加进度条
  for drugs, targets, affinities in tqdm(train_loader, desc="Training Batch", unit="batch", leave=False):
    drugs, targets, affinities = drugs.to(device), targets.to(device), affinities.to(device)
    optimizer.zero_grad()
    mse_loss, kl_loss = criterion(model(drugs, targets), affinities)

    loss = mse_loss + get_beta(epoch) * kl_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    total_mse_loss += mse_loss.item()
    total_kl_loss += kl_loss.item()
  
  # 验证集评估
  val_mse = evaluate_mse(model, val_loader, device)
  return total_mse_loss / len(train_loader), total_kl_loss / len(train_loader), val_mse

def cross_validate(data, params):

  kf = KFold(n_splits=5, shuffle=True, random_state=42)
  scores = {}
  # 为每个metric初始化空列表
  for metric in params['metrics']:
    scores[metric] = []
  
  # 记录总开始时间
  total_start_time = time.time()

  # 记录每个fold的终止epoch数
  fold_stop_epochs = []
  
  for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(data['X_train']), desc="Cross Validation", unit="fold", total=5)):
    # 记录当前fold的开始时间
    fold_start_time = time.time()
    
    X_train_fold, X_val_fold = data['X_train'][train_idx], data['X_train'][val_idx]
    y_train_fold, y_val_fold = data['y_train'][train_idx], data['y_train'][val_idx]
    train_set = AffinityDataset(list(map(itemgetter(0), X_train_fold)), list(map(itemgetter(1), X_train_fold)), y_train_fold)
    val_set = AffinityDataset(list(map(itemgetter(0), X_val_fold)), list(map(itemgetter(1), X_val_fold)), y_val_fold)
    
    # 为DataLoader设置随机种子生成器，确保shuffle=True时结果可复现
    g = torch.Generator()
    g.manual_seed(42)
    train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True, generator=g)
    val_loader = DataLoader(val_set, batch_size=params['batch_size'])
    
    # 根据模型名称创建模型实例
    model = create_model(data, params).to(params['device'])
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    criterion = create_criterion(params)

    # 使用早停策略
    best_val_mse = float('inf')
    patience_counter = 0
    best_state_dict = None
    best_epoch = 0

    # 使用tqdm包装epoch循环
    for epoch in tqdm(range(1, params['epochs'] + 1), desc=f"Fold {fold+1} Training", unit="epoch", leave=False):
      train_mse_loss, train_kl_loss, val_mse = train_epoch_with_validation(
              epoch, model, train_loader, val_loader, criterion, optimizer, params['device']
          )
      # L2正则化项损失
      l2_reg = 0
      for param in model.parameters():
        l2_reg += param.pow(2).sum()
      l2_reg = params['weight_decay'] * l2_reg * 0.5
      logging.info(f'Fold {fold+1}: Epoch {epoch:2d} | Train Total Loss: {train_mse_loss + train_kl_loss + l2_reg:.4f} | MSE Loss: {train_mse_loss:.4f} | KL Loss: {train_kl_loss:.4f} | L2 Reg Loss: {l2_reg:.4f} | Val MSE: {val_mse:.4f}')
      # 检查是否改善
      if val_mse < best_val_mse - params['min_delta']:
        best_val_mse = val_mse
        patience_counter = 0
        best_epoch = epoch
        best_state_dict = model.state_dict().copy()
        print("  -> New best, saving model...")
      else:
        patience_counter += 1
        print(f"  -> No improvement for {patience_counter} epoch(s)")
      # 早停触发
      if patience_counter >= params['patience']:
        print(f"\nEarly stopping at epoch {epoch}. Best Val MSE: {best_val_mse:.4f}")
        break     

    # 记录当前fold的最佳Epoch
    fold_stop_epochs.append(best_epoch)  

    # 加载最佳模型并评估
    model.load_state_dict(best_state_dict)

    final_val_result = evaludate_metrics(model, val_loader, params['device'], params['metrics'])
    for metric in params['metrics']:
      scores[metric].append(final_val_result[metric])
      logging.info(f'Fold {fold+1} Validation set {metric}: {final_val_result[metric]:.4f}')
    
    # 计算当前fold的运行时间
    fold_end_time = time.time()
    fold_duration = fold_end_time - fold_start_time
    logging.info(f'Fold {fold+1} Time: {fold_duration:.2f} seconds')
  
  # 计算平均终止epoch数
  avg_stop_epoch = int(np.round(np.mean(fold_stop_epochs)))
  logging.info(f'Average Stop Epoch: {avg_stop_epoch}')
  
  # 计算总运行时间
  total_end_time = time.time()
  total_duration = total_end_time - total_start_time
  logging.info(f'Total CV Time: {total_duration:.2f} seconds')
  logging.info(f'-------------------------------------\n')


  # 计算每个指标的均值
  mean_metrics = {}
  for metric in params['metrics']:
    mean_metrics[metric] = np.mean(scores[metric])
    logging.info(f'CV Mean {metric}: {mean_metrics[metric]:.4f}')
  
  return mean_metrics, avg_stop_epoch

def search_param(data, fixed_params, searched_params):
  
  # 网格搜索
  best_mse, best_params = float('inf'), None

  # 提取参数名和对应的取值列表
  param_names = list(searched_params.keys())
  print(param_names)
  param_values = list(searched_params.values())
  # 生成所有参数组合
  param_combinations = list(itertools.product(*param_values))
  
  # 准备存储结果的列表（每个元素是一行数据：参数 + 性能指标）
  results = []

  # 遍历每一组参数，模拟训练并记录结果
  for params in tqdm(param_combinations, desc="Hyperparameter Search", unit="param"):
    
    # 将要进行网格搜索的参数更新到full_params中
    params_dict = dict(zip(param_names, params))
    full_params = fixed_params.copy()
    full_params.update(params_dict)

    # 根据batch_size调整学习率
    # full_params['lr'] = full_params['lr'] * full_params['batch_size']/128

    # 进行交叉验证，评估对应参数下模型的性能
    logging.info(f"Testing: {full_params}")
    metric_results, avg_stop_epoch = cross_validate(data, full_params)
    
    # 记录每组参数下的模型性能指标
    result_row = searched_params.copy()
    for metric in full_params['metrics']:
      result_row[metric] = round(metric_results[metric], 3) 
    
    results.append(result_row)

    # 选择最佳参数（根据MSE）
    mse = metric_results['mse']
    if mse < best_mse:
      best_mse, best_params = mse, full_params
      # 保存最佳参数对应的平均终止epoch
      best_avg_stop_epoch = avg_stop_epoch
    
  logging.info("--------------------------------- Best Search Result ---------------------------------")
  logging.info(f"Best CV MSE: {best_mse:.4f}")
  logging.info(f"Best Average Stop Epoch: {best_avg_stop_epoch}")
  
  # 生成包含所有参数和性能指标的统计文件
  create_statistics_file(param_names, results, full_params['metrics'])
  # 重写 Epochs 的值
  best_params['epochs'] = best_avg_stop_epoch
  
  return best_params

def final_model_train(data, params):
  # 最终模型训练（同样使用早停）
  train_set = AffinityDataset(list(map(itemgetter(0), data['X_train'])), list(map(itemgetter(1), data['X_train'])), data['y_train'])
  # 为DataLoader设置随机种子生成器，确保shuffle=True时结果可复现
  g = torch.Generator()
  g.manual_seed(42)
  train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True, generator=g)
  test_set = AffinityDataset(list(map(itemgetter(0), data['X_test'])), list(map(itemgetter(1), data['X_test'])), data['y_test'])
  test_loader = DataLoader(test_set, batch_size=params['batch_size'])
  
  #final_model = MatrixFactorization(data['n_drugs'], data['n_targets'], params['n_factors']).to(device)
  final_model = create_model(data, params).to(params['device'])
  optimizer = optim.AdamW(final_model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
  criterion = create_criterion(params)

  
  for epoch in range(1, params['epochs'] + 1):
    train_mse_loss, train_kl_loss,  val_mse = train_epoch_with_validation(
        epoch, final_model, train_loader, test_loader, criterion, optimizer, params['device']
    )
    train_loss = train_mse_loss + train_kl_loss
    print(f'Final Train Epoch {epoch} | Loss: {train_loss:.4f} | Test MSE: {val_mse:.4f}')
    
    
  # 保存最佳模型
  if params['is_store_model']:
    best_state_dict = final_model.state_dict().copy()
    torch.save(best_state_dict, f'{LOG_FOLDER}/best_model.pth')

  final_test_results = evaludate_metrics(final_model, test_loader, params['device'], params['metrics'])
  for metric in params['metrics']:
    logging.info(f'=== Final Test {metric}: {final_test_results[metric]:.4f} ===')
  
  return final_test_results

# 参数解析
def params_parse():
  # 解析命令行参数
  parser = argparse.ArgumentParser()
  
  parser.add_argument('--dataset', type=str, default='kiba')
  parser.add_argument('--epochs', type=int, default=1000)
  parser.add_argument('--patience', type=int, default=10)
  parser.add_argument('--min_delta', type=float, default=0.001)
  parser.add_argument('--model_name', type=str, default='VIB')
  parser.add_argument('--is_store_model', default=False)

  parser.add_argument('--max_smi_len', type=int, default=100)
  parser.add_argument('--max_seq_len', type=int, default=1000)

  parser.add_argument('--lr', type=float, default=0.001)
  parser.add_argument('--batch_size', type=int, default=1024)
  parser.add_argument('--weight_decay', type=float, default=1e-5)
  parser.add_argument('--metrics', type=eval, default=['mse','rm2','cindex'])

  args = parser.parse_args()

  # 超参数空间
  param_grid = {
    'n_factors': [16],
    'lr': [5e-4],
    'batch_size': [256],
    'weight_decay':[0],
    'max_beta': [5e-3],
    'z_dim': [12]
  }
  
  fixed_params = args.__dict__

  if args.dataset == 'davis':
    fixed_params['max_smi_len'] = 85
    fixed_params['max_seq_len'] = 1200
  elif args.dataset == 'kiba':
    fixed_params['max_smi_len'] = 100
    fixed_params['max_seq_len'] = 1000

  
  # 确定设备（GPU或CPU）
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  fixed_params['device'] = device
  
  searched_params = {}
  if param_grid is not None:
    param_grid_items = param_grid.copy().items()
    for param_name, param_values  in param_grid_items:
      if len(param_values) == 1:
        fixed_params[param_name] = param_values[0]
        param_grid.pop(param_name)
      else:
        searched_params[param_name] = param_values  

  return fixed_params, searched_params


def main():

  # 设置随机种子，确保结果可复现
  set_seed(42)
  fixed_params, searched_params = params_parse()
  dataset = fixed_params['dataset']

  # 加载日志配置
  global LOG_FOLDER
  LOG_FOLDER = os.path.join(LOG_FOLDER, fixed_params['model_name'])
  LOG_FOLDER = os.path.join(LOG_FOLDER, dataset)
  load_logging()

  # 定义随机种子列表用于多次重复评估性能
  seeds = [42, 123, 456, 789, 101112]
  
  # 加载亲和度数据
  data = load_data(fixed_params)

  # 超参数搜索返回最优参数
  best_params = search_param(data, fixed_params, searched_params)
  logging.info(f"Best Params: {best_params}")

  # 根据最佳超参数训练最终模型
  all_results = {}
  for metric in fixed_params['metrics']:
    all_results[metric] = []
  for seed in seeds:
    data = load_data(fixed_params, seed =seed)
    logging.info(f"Final Test by Seed {seed}:")
    results = final_model_train(data, best_params)
    for metric in fixed_params['metrics']:
      all_results[metric].append(results[metric])

  for metric in fixed_params['metrics']:
    logging.info(f"{metric} Mean Results: {np.mean(all_results[metric]):.4f}")
    logging.info(f"{metric} Std Results: {np.std(all_results[metric]):.4f}")
  

if __name__ == '__main__':
  # 运行主程序
  main()