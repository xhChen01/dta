from datetime import datetime
import numpy as np
import argparse
from data_process import *
from sklearn.model_selection import KFold
from operator import itemgetter
from torch.utils.data import DataLoader
from data_loader import AffinityDataset
from models import MatrixFactorization, DeepDTA
from torch import optim
import torch.nn as nn
import logging
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from metrics import evaludate_metrics
import torch
import random
import time
from tqdm import tqdm

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
def plot_param_impact(colors, markers, df, x_param, y_param, group_params):
    """
    绘制单个参数对性能的影响图
    :param df: 数据框
    :param x_param: 横坐标参数（如n_factors）
    :param y_param: 纵坐标指标（如rmse）
    :param group_params: 分组参数（其余所有参数）
    """
    # 1. 生成分组标签（其余参数的组合）
    df['group_label'] = df.apply(
        lambda row: ', '.join([f"{p}={row[p]}" for p in group_params]),
        axis=1
    )
    
    # 2. 创建图表
    fig, ax = plt.subplots()
    
    # 3. 遍历每个分组绘制折线
    for idx, (group_name, group_data) in enumerate(df.groupby('group_label')):
        # 按横坐标参数排序（保证折线顺序正确）
        group_data_sorted = group_data.sort_values(x_param)
        # 绘制折线
        ax.plot(
            group_data_sorted[x_param],
            group_data_sorted[y_param],
            label=group_name,
            color=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
            linewidth=2,
            markersize=8
        )
    
    # 4. 图表标注与美化
    ax.set_title(f'{x_param}对{y_param.upper()}的影响（其余参数固定）', fontsize=16, pad=20)
    ax.set_xlabel(x_param, fontsize=14, labelpad=10)
    ax.set_ylabel(y_param.upper(), fontsize=14, labelpad=10)
    
    # 设置x轴刻度为该参数的唯一值（保证刻度精准）
    x_ticks = sorted(df[x_param].unique())
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(t) for t in x_ticks], fontsize=12)
    
    # 网格、图例、背景
    ax.grid(True, axis='y')
    # 只有当有分组参数时才创建图例
    if group_params:
        ax.legend(
            title='固定参数组合', 
            fontsize=9, 
            title_fontsize=11, 
            loc='best',
            bbox_to_anchor=(1.05, 1)  # 图例靠右显示，避免遮挡折线
        )
    ax.set_facecolor('#f8f9fa')
    
    # 调整布局
    plt.tight_layout()
    
    # 5. 保存图表
    save_path = f'{x_param}_vs_{y_param}.png'
    save_path = os.path.join(LOG_FOLDER, save_path)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()  # 关闭图表释放内存

def plot_param_search_results(df, target_params, metrics):
  """
    绘制参数搜索结果的折线图
    
    :param df: 数据框，包含参数和性能指标
    :param target_params: 目标参数列表，用于横坐标
  """ 

  # 全局样式设置（统一所有图表风格）
  plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows显示中文
  # plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac显示中文
  plt.rcParams['axes.unicode_minus'] = False
  plt.rcParams['figure.figsize'] = (12, 8)
  plt.rcParams['grid.alpha'] = 0.3
  plt.rcParams['savefig.dpi'] = 300  # 保存图片的默认清晰度


  # 定义颜色和标记（循环使用，适配不同分组）
  colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
  markers = ['o', 's', '^', 'D', 'v', '*', 'p', 'h']
  for metric in metrics:
    for x_param in target_params:
      # 分组参数 = 所有参数 - 当前横坐标参数
      group_params = [p for p in target_params if p != x_param]
      # 绘制并保存图表
      plot_param_impact(colors, markers, df, x_param, metric, group_params)

def create_statistics_file(params_name, results, metrics):
  global LOG_FOLDER

  # 1. 将结果保存到 Exce l中
  excel_path = os.path.join(LOG_FOLDER, 'param_search_results.xlsx')
  df_results = pd.DataFrame(results)
  df_results.to_excel(excel_path, index=False, engine='openpyxl')
  
  # 2. 根据统计结果生成折线图，显示不同超参数对性能的影响
  plot_param_search_results(df_results, params_name, metrics)
  

# 计算MSE
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

def train_epoch_with_validation(model, train_loader, val_loader, criterion, optimizer, device):
  """训练一个epoch并返回验证MSE用于早停"""
  model.train()
  total_loss = 0
  # 为数据加载添加进度条
  for drugs, targets, affinities in tqdm(train_loader, desc="Training Batch", unit="batch", leave=False):
    drugs, targets, affinities = drugs.to(device), targets.to(device), affinities.to(device)
    optimizer.zero_grad()
    loss = criterion(model(drugs, targets), affinities)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
  
  # 验证集评估
  val_mse = evaluate_mse(model, val_loader, device)
  return total_loss / len(train_loader), val_mse

def cross_validate(data, params, device, metrics):

  kf = KFold(n_splits=5, shuffle=True, random_state=42)
  scores = {}
  # 为每个metric初始化空列表
  for metric in metrics:
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
    
    # model = MatrixFactorization(data['n_drugs'], data['n_targets'], params['n_factors']).to(device)
    model = DeepDTA(data['ligands_features'], data['proteins_features'], data['max_smi_len'], data['max_seq_len'], data['charsmiset_size'], data['charprotset_size']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    criterion = nn.MSELoss()

    # 使用早停策略
    best_val_mse = float('inf')
    patience_counter = 0
    best_state_dict = None
    best_epoch = 0

    # 使用tqdm包装epoch循环
    for epoch in tqdm(range(1, params['epochs'] + 1), desc=f"Fold {fold+1} Training", unit="epoch", leave=False):
      train_loss, val_mse = train_epoch_with_validation(
              model, train_loader, val_loader, criterion, optimizer, device
          )
      logging.info(f'Fold {fold+1}: Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Val MSE: {val_mse:.4f}')
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

    final_val_result = evaludate_metrics(model, val_loader, device, metrics)
    for metric in metrics:
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
  for metric in metrics:
    mean_metrics[metric] = np.mean(scores[metric])
    logging.info(f'CV Mean {metric}: {mean_metrics[metric]:.4f}')
  
  # 记录时间信息
  
  return mean_metrics, avg_stop_epoch

def search_param(data, device, metrics, param_grid, fixed_params):
  
  param_grid_items = param_grid.copy().items()
  # 将超参数网格params_grid中参数列表个数为1的参数剔除，加入到fixed_params中
  for param_name, param_values in param_grid_items:
    if len(param_values) == 1:
      fixed_params[param_name] = param_values[0]
      param_grid.pop(param_name)
      
  # 网格搜索
  best_mse, best_params = float('inf'), None

  # 提取参数名和对应的取值列表
  param_names = list(param_grid.keys())
  print(param_names)
  param_values = list(param_grid.values())
  # 生成所有参数组合
  param_combinations = list(itertools.product(*param_values))
  
  # 准备存储结果的列表（每个元素是一行数据：参数 + 性能指标）
  results = []

  # 遍历每一组参数，模拟训练并记录结果
  for idx, params_values in enumerate(tqdm(param_combinations, desc="Hyperparameter Search", unit="param")):
    # 将参数组合转换为字典（方便调用和记录）
    searched_params = dict(zip(param_names, params_values))
    full_params = searched_params.copy()
    full_params.update(fixed_params)

    logging.info(f"Testing: {full_params}")
    metric_results, avg_stop_epoch = cross_validate(data, full_params, device, metrics)
    
    result_row = searched_params.copy()
    for metric in metrics:
      result_row[metric] = round(metric_results[metric], 3) 
    
    results.append(result_row)

    mse = metric_results['mse']
    if mse < best_mse:
      best_mse, best_params = mse, full_params
      # 保存最佳参数对应的平均终止epoch
      best_avg_stop_epoch = avg_stop_epoch
    
  logging.info("--------------------------------- Best Search Result ---------------------------------")
  logging.info(f"Best CV MSE: {best_mse:.4f}")
  logging.info(f"Best Average Stop Epoch: {best_avg_stop_epoch}")
  
  
  create_statistics_file(param_names, results, metrics)
  # 重写 Epochs 的值
  best_params['epochs'] = best_avg_stop_epoch
  
  return best_params, best_avg_stop_epoch

def final_model_train(data, params, device, metrics):
  # 最终模型训练（同样使用早停）
  train_set = AffinityDataset(list(map(itemgetter(0), data['X_train'])), list(map(itemgetter(1), data['X_train'])), data['y_train'])
  # 为DataLoader设置随机种子生成器，确保shuffle=True时结果可复现
  g = torch.Generator()
  g.manual_seed(42)
  train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True, generator=g)
  test_set = AffinityDataset(list(map(itemgetter(0), data['X_test'])), list(map(itemgetter(1), data['X_test'])), data['y_test'])
  test_loader = DataLoader(test_set, batch_size=params['batch_size'])
  
  #final_model = MatrixFactorization(data['n_drugs'], data['n_targets'], params['n_factors']).to(device)
  final_model = DeepDTA(data['ligands_features'], data['proteins_features'], data['max_smi_len'], data['max_seq_len'], data['charsmiset_size'], data['charprotset_size']).to(device)
  optimizer = optim.Adam(final_model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
  criterion = nn.MSELoss()

  
  for epoch in range(1, params['epochs'] + 1):
    train_loss, val_mse = train_epoch_with_validation(
        final_model, train_loader, test_loader, criterion, optimizer, device
    )
    print(f'Final Train Epoch {epoch} | Loss: {train_loss:.4f} | Test MSE: {val_mse:.4f}')
    
    
  # 保存最佳模型
  if params['is_store_model']:
    best_state_dict = final_model.state_dict().copy()
    torch.save(best_state_dict, f'{LOG_FOLDER}/best_model.pth')

  final_test_results = evaludate_metrics(final_model, test_loader, device, metrics)
  for metric in metrics:
    logging.info(f'=== Final Test {metric}: {final_test_results[metric]:.4f} ===')
  
  return final_test_results
def main():
  # 设置随机种子，确保结果可复现
  set_seed(42)
  # 解析命令行参数
  parser = argparse.ArgumentParser()
  parser.add_argument('--cuda', type=int, default=0)
  parser.add_argument('--dataset', type=str, default='kiba')
  parser.add_argument('--is_search', action='store_true', default=True)
  parser.add_argument('--epochs', type=int, default=1000)
  parser.add_argument('--patience', type=int, default=10)
  parser.add_argument('--min_delta', type=float, default=0.001)
  parser.add_argument('--model', type=str, default='DeepDTA')
  parser.add_argument('--is_store_model', default=True)

  parser.add_argument('--max_smi_len', type=int, default=100)
  parser.add_argument('--max_seq_len', type=int, default=1000)

  parser.add_argument('--lr', type=float, default=0.001)
  parser.add_argument('--batch_size', type=int, default=1024)
  parser.add_argument('--weight_decay', type=float, default=1e-5)
  parser.add_argument('--metrics', type=eval, default=['mse','rm2','cindex'])

  
  # 解析命令行参数
  args, unknown = parser.parse_known_args()
  dataset = args.dataset

  global LOG_FOLDER
  LOG_FOLDER = os.path.join(LOG_FOLDER, args.model)
  # 加载日志配置
  load_logging()

  # 定义随机种子列表用于多次重复评估性能
  seeds = [42, 123, 456, 789, 101112]

  # 超参数空间
  param_grid = {
    'n_factors': [128],
    'lr': [1e-3],
    'batch_size': [256],
    'weight_decay':[1e-5],
  }

  fixed_params = {
    'epochs': args.epochs,
    'patience': args.patience, 
    'min_delta': args.min_delta,
    'max_smi_len': 100,
    'max_seq_len': 1000,
  }

  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  # 加载亲和度数据

  data = load_data(dataset, 
                   max_smi_len=fixed_params['max_smi_len'],
                   max_seq_len=fixed_params['max_seq_len'],
                   is_load_1d_data=True)

  # 如果进行超参数搜索
  if args.is_search:
    best_params, avg_stop_epoch = search_param(data,
                              device, args.metrics,
                              param_grid, fixed_params)
    best_params['is_store_model'] = args.is_store_model
  else:
    best_params = args.__dict__
    metric_results, avg_stop_epoch = cross_validate(data, best_params, device, args.metrics)
    best_params['epochs'] = avg_stop_epoch


  logging.info(f"Best Params: {best_params}")
  # 根据最佳超参数训练最终模型
  
  all_results = {}
  for metric in args.metrics:
    all_results[metric] = []
  for seed in seeds:
    data  = load_data(dataset, 
                   max_smi_len=fixed_params['max_smi_len'],
                   max_seq_len=fixed_params['max_seq_len'],
                   seed =seed,
                   is_load_1d_data=True)
    logging.info(f"Final Test by Seed {seed}:")
    results = final_model_train(data, best_params, device, args.metrics)
    for metric in args.metrics:
      all_results[metric].append(results[metric])

  for metric in args.metrics:
    logging.info(f"{metric} Mean Results: {np.mean(all_results[metric]):.4f}")
    logging.info(f"{metric} Std Results: {np.std(all_results[metric]):.4f}")
  
  # 读取药物分子SMILES字符串，并将其转换为分子图表示
  #ligands = json.load(open(f'data/{dataset}/drugs.txt'))
  #drug_graphs = get_drug_molecule_graph(ligands)

if __name__ == '__main__':
  # 运行主程序
  main()