import matplotlib.pyplot as plt

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
