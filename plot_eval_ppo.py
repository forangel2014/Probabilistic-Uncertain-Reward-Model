import os
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# 添加颜色映射字典
color_mapping = {
    "BTRM": '#ff7f00',
    "PURM": '#1f77b4',
    "BTE-mean": 'green',
    "BTE-WCO": 'darkblue',
    "BTE-UWO": 'purple',
    "RRM": '#777777',
    "BRME": "yellowgreen",
    "λ = 0": 'brown',
    "λ = 1": 'red',
    "λ = 5": 'orange',
    "λ = 30": 'green',
    "λ = 50": 'darkblue',
    "PURM penalize w/ σ": 'purple'
}
# 实验名称到显示名称的映射
display_name_mapping = {
    "hh_rlhf_btrm": "BTRM",
    "hh_rlhf_purm": "PURM",
    "hh_rlhf_brme": "BRME",
    "hh_rlhf_rrm": "RRM",
    "hh_rlhf_bte_mean": "BTE-mean",
    "hh_rlhf_bte_wco": "BTE-WCO",
    "hh_rlhf_bte_uwo1": "BTE-UWO",
    "hh_rlhf_purm_penalty_0": "λ = 0",
    "hh_rlhf_purm_penalty_5": "λ = 5",
    "hh_rlhf_purm_window_1000000_penalty_30": "λ = 30",
    "hh_rlhf_purm_penalty_50": "λ = 50",
    "hh_rlhf_purm_sigma_as_uncertainty": "PURM penalize w/ σ",
}

def read_win_rate(file_path):
    """读取文件中的 A win rate 数值"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            match = re.search(r'A win rate: (\d+\.\d+)%', content)
            if match:
                return float(match.group(1))
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    return None

def get_win_rate_data(exp_name, base_dirs=None):
    """获取指定实验的步骤和胜率数据，支持从多个目录读取"""
    if base_dirs is None:
        base_dirs = [f"./inference_results/{exp_name}"]
    elif isinstance(base_dirs, str):
        base_dirs = [f"{base_dirs}/{exp_name}"]
    else:
        assert isinstance(base_dirs, list)
        base_dirs = [f"{base_dir}/{exp_name}" for base_dir in base_dirs]
        
    # 用字典存储每个step的所有win_rate值
    step_to_win_rates = {}
    
    # 添加pretrain的数据点作为step=0
    pretrain_file = "./inference_results/pretrain/pretrain_vs_gpt-4o.txt"
    if os.path.exists(pretrain_file):
        pretrain_win_rate = read_win_rate(pretrain_file)
        if pretrain_win_rate is not None:
            step_to_win_rates[0] = [pretrain_win_rate]
            print(f"Added pretrain data point: {pretrain_win_rate}%")
    
    # 从每个base_dir中读取数据
    for base_dir in base_dirs:

        # 确保目录存在
        if not os.path.exists(base_dir):
            print(f"Directory {base_dir} does not exist")
            continue
            
        # 遍历所有 global_step 子目录
        for item in os.listdir(base_dir):
            if not item.startswith("global_step"):
                continue
            
            # 提取步数
            step_match = re.search(r'global_step(\d+)', item)
            if not step_match:
                continue
                
            step = int(step_match.group(1))
            
            # 检查是否存在对比GPT-4o的评估文件
            eval_file = os.path.join(base_dir, item, f"{exp_name}_vs_gpt-4o.txt")
            if not os.path.exists(eval_file):
                continue
                
            # 读取胜率
            win_rate = read_win_rate(eval_file)
            if win_rate is not None:
                if step not in step_to_win_rates:
                    step_to_win_rates[step] = []
                step_to_win_rates[step].append(win_rate)
    
    # 计算每个step的平均胜率和标准差
    steps = sorted(step_to_win_rates.keys())
    win_rates_mean = []
    win_rates_std = []
    
    for step in steps:
        values = step_to_win_rates[step]
        win_rates_mean.append(np.mean(values))
        # 如果只有一个值，标准差为0
        win_rates_std.append(np.std(values) if len(values) > 1 else 0)
    
    return steps, win_rates_mean, win_rates_std

def plot_win_rate_vs_steps(exp_name, base_dirs=None):
    """为指定实验绘制 global_step 和 win rate 的折线图，包括标准差区域"""
    steps, win_rates_mean, win_rates_std = get_win_rate_data(exp_name, base_dirs)
    
    if not steps:
        print(f"No data found for {exp_name}")
        return
    
    # 获取显示名称
    display_name = display_name_mapping.get(exp_name, exp_name)
    
    # 绘制折线图
    plt.figure(figsize=(10, 6), facecolor='#f8f8f8')
    ax = plt.gca()
    ax.set_facecolor('#ebebf2')
    color = color_mapping.get(display_name, 'blue')
    
    # 绘制主线条
    plt.plot(steps, win_rates_mean, marker='o', linestyle='-', linewidth=2, 
             color=color, label=display_name)
    
    # 添加标准差区域
    plt.fill_between(steps, 
                    np.array(win_rates_mean) - np.array(win_rates_std), 
                    np.array(win_rates_mean) + np.array(win_rates_std), 
                    color=color, alpha=0.2)
    
    plt.xlabel('Training Steps')
    plt.ylabel('Win Rate against GPT-4o (%)')
    plt.grid(True)
    plt.savefig(f"./inference_results/{exp_name}/{exp_name}_win_rate.pdf", bbox_inches='tight')
    plt.show()
    
    print(f"Total data points: {len(steps)}")
    if len(steps) > 0:
        print(f"Maximum win rate: {max(win_rates_mean):.1f}% (step {steps[win_rates_mean.index(max(win_rates_mean))]})")

def plot_multiple_win_rates(exp_names, base_dirs=None, figure_size=(12, 6)):
    """在同一张图上绘制多个实验的胜率曲线，包括标准差区域"""
    plt.figure(figsize=figure_size, facecolor='#f8f8f8')
    ax = plt.gca()
    ax.set_facecolor('#ebebf2')

    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for i, exp_name in enumerate(exp_names):
        steps, win_rates_mean, win_rates_std = get_win_rate_data(exp_name, base_dirs)
        
        if not steps:
            print(f"No data found for {exp_name}")
            continue
        
        # 获取显示名称
        display_name = display_name_mapping.get(exp_name, exp_name)
        
        # 获取颜色
        color = color_mapping.get(display_name, 'blue')
        marker_idx = i % len(markers)
        
        # 绘制主线条
        plt.plot(steps, win_rates_mean, 
                marker=markers[marker_idx], 
                linestyle='-', 
                linewidth=2,
                color=color,
                label=display_name)
        
        # 添加标准差区域
        plt.fill_between(steps, 
                        np.array(win_rates_mean) - np.array(win_rates_std), 
                        np.array(win_rates_mean) + np.array(win_rates_std), 
                        color=color, alpha=0.2)
        
        print(f"{display_name} total data points: {len(steps)}")
        if len(steps) > 0:
            print(f"{display_name} maximum win rate: {max(win_rates_mean):.1f}% (step {steps[win_rates_mean.index(max(win_rates_mean))]})")
    
    plt.xlabel('Training Steps')
    plt.ylabel('Win Rate against GPT-4o (%)')
    plt.grid(True)
    plt.legend()

def plot_grouped_comparisons(base_dirs=None):
    """绘制四张不同的对比图，支持从多个目录读取数据"""
    # 定义所有实验名称
    all_experiments = ["hh_rlhf_btrm", "hh_rlhf_purm_rerun_10", 
                       "hh_rlhf_brme", "hh_rlhf_rrm", "hh_rlhf_bte_rerun", 
                       "hh_rlhf_purm_penalty_0", "hh_rlhf_purm_penalty_1", 
                       "hh_rlhf_purm_penalty_50", "hh_rlhf_purm_sigma_as_uncertainty1",
                       "hh_rlhf_purm_quick_window_100000"]
    
    # 1. PURM与其他不带PURM前缀的方法对比
    non_purm_methods = ["hh_rlhf_btrm", "hh_rlhf_brme", "hh_rlhf_rrm", "hh_rlhf_bte_rerun", "hh_rlhf_bte_uwo1"]
    plot_multiple_win_rates(["hh_rlhf_purm_quick_window_1000000"] + non_purm_methods, base_dirs)
    
    # 添加pretrain的水平线
    pretrain_file = "./inference_results/pretrain/pretrain_vs_gpt-4o.txt"
    if os.path.exists(pretrain_file):
        pretrain_win_rate = read_win_rate(pretrain_file)
        if pretrain_win_rate is not None:
            plt.axhline(y=pretrain_win_rate, linestyle='--', color='gray', label="Qwen-2.5-3B")
    
    plt.legend()
    plt.xlim(0, 1344)  # 设定最大step为1344
    plt.savefig("./inference_results/reward_comparison.pdf", bbox_inches='tight')
    plt.close()
    
    # 2. PURM与所有PURM_penalty_xxx的方法对比
    penalty_methods = ["hh_rlhf_btrm", "hh_rlhf_purm_penalty_0", "hh_rlhf_purm_penalty_5", "hh_rlhf_purm_quick_window_1000000", "hh_rlhf_purm_window_1000000_penalty_30", "hh_rlhf_purm_penalty_50"]
    
    # 临时保存原始映射
    original_mapping = display_name_mapping.get("hh_rlhf_purm_quick_window_1000000")
    # 临时修改为λ = 10
    display_name_mapping["hh_rlhf_purm_quick_window_1000000"] = "λ = 10"
    
    plot_multiple_win_rates(penalty_methods, base_dirs)
    
    # 恢复原始映射
    display_name_mapping["hh_rlhf_purm_quick_window_1000000"] = original_mapping
    
    plt.xlim(0, 1344)  # 设定最大step为1344
    plt.savefig("./inference_results/ablation.pdf", bbox_inches='tight')
    plt.close()
    
    # 3. PURM与PURM_SIGMA_AS_UNCERTAINTY对比
    plot_multiple_win_rates(["hh_rlhf_purm_quick_window_1000000", "hh_rlhf_purm_sigma_as_uncertainty1"], base_dirs)
    plt.xlim(0, 1344)  # 设定最大step为1344
    plt.savefig("./inference_results/purm_variance_comparison.pdf", bbox_inches='tight')
    plt.close()
    
    # # 4. PURM与PURM_window_xxx对比
    # plot_multiple_win_rates(["hh_rlhf_purm_quick_window_1000000", "hh_rlhf_purm_quick_window_100000", "hh_rlhf_purm_rerun_10"], base_dirs)
    # plt.xlim(0, 1344)  # 设定最大step为1344
    # plt.savefig("./inference_results/purm_vs_window.pdf", bbox_inches='tight')
    # plt.close()
    
    # 5. Reward Hacking图：只对比PURM和BTRM
    purm_name = "hh_rlhf_purm_quick_window_1000000"
    btrm_name = "hh_rlhf_btrm"
    
    # 获取PURM和BTRM的数据
    purm_steps, purm_win_rates_mean, purm_win_rates_std = get_win_rate_data(purm_name, base_dirs)
    btrm_steps, btrm_win_rates_mean, btrm_win_rates_std = get_win_rate_data(btrm_name, base_dirs)
    
    # 先绘制基本的对比图
    plot_multiple_win_rates([purm_name, btrm_name], base_dirs, figure_size=(10, 6))
    
    # 找出最高点
    if len(purm_win_rates_mean) > 0:
        purm_max_idx = np.argmax(purm_win_rates_mean)
        purm_max_value = purm_win_rates_mean[purm_max_idx]
        purm_max_step = purm_steps[purm_max_idx]
        
        # 添加PURM最高点的投影线（只投影到坐标轴）
        plt.plot([0, purm_max_step], [purm_max_value, purm_max_value], '--', 
                color=color_mapping["PURM"], label=f'PURM max win rate: {purm_max_value:.1f}%')
        plt.plot([purm_max_step, purm_max_step], [0, purm_max_value], '--', 
                color=color_mapping["PURM"])
    
    if len(btrm_win_rates_mean) > 0:
        btrm_max_idx = np.argmax(btrm_win_rates_mean)
        btrm_max_value = btrm_win_rates_mean[btrm_max_idx]
        btrm_max_step = btrm_steps[btrm_max_idx]
        
        # 添加BTRM最高点的投影线（只投影到坐标轴）
        plt.plot([0, btrm_max_step], [btrm_max_value, btrm_max_value], '--', 
                color=color_mapping["BTRM"], label=f'BTRM max win rate: {btrm_max_value:.1f}%')
        plt.plot([btrm_max_step, btrm_max_step], [0, btrm_max_value], '--', 
                color=color_mapping["BTRM"])
    
    # 尝试添加PURM比BTRM好的区域（需要进行插值处理）
    if len(purm_steps) > 0 and len(btrm_steps) > 0:
        # 创建共同的x轴步骤范围
        min_step = min(min(purm_steps), min(btrm_steps))
        max_step = max(max(purm_steps), max(btrm_steps))
        common_steps = np.linspace(min_step, max_step, 1000)
        
        # 对PURM和BTRM的胜率进行插值
        if len(purm_steps) >= 2:  # 需要至少两个点才能插值
            purm_interp = interp1d(purm_steps, purm_win_rates_mean, 
                                  kind='linear', bounds_error=False, fill_value="extrapolate")
            purm_values = purm_interp(common_steps)
        else:
            purm_values = np.full_like(common_steps, purm_win_rates_mean[0] if purm_win_rates_mean else 0)
            
        if len(btrm_steps) >= 2:  # 需要至少两个点才能插值
            btrm_interp = interp1d(btrm_steps, btrm_win_rates_mean, 
                                  kind='linear', bounds_error=False, fill_value="extrapolate")
            btrm_values = btrm_interp(common_steps)
        else:
            btrm_values = np.full_like(common_steps, btrm_win_rates_mean[0] if btrm_win_rates_mean else 0)
        
        # 找出PURM优于BTRM的区域
        where_purm_better = purm_values > btrm_values
        
        # 添加斜线填充
        plt.fill_between(common_steps, purm_values, btrm_values, 
                        where=where_purm_better, 
                        facecolor='lightblue', alpha=0.3, 
                        hatch='///', edgecolor='blue', linewidth=0,
                        label='PURM > BTRM')
    
    # 设置图例位置，提供初始值
    plt.legend(bbox_to_anchor=(0.1, 0.2), loc='upper left')
    plt.ylim(ymin=10)  # 设置y轴最小值为10，保持最大值为自动调整
    plt.xlim(0, 1344)  # 设定最大step为1344
    plt.savefig("./inference_results/reward_hacking.pdf", bbox_inches='tight')
    plt.close()

def plot_reward_hacking_with_demo(base_dirs=None):
    """绘制Reward Hacking图：PURM和BTRM的对比，并在右侧添加演示案例"""
    purm_name = "hh_rlhf_purm_quick_window_1000000"
    btrm_name = "hh_rlhf_btrm"
    
    # 创建一个具有特定宽高比的图形，左侧10:6，右侧2:6
    fig = plt.figure(figsize=(12, 6), facecolor='#f8f8f8')
    
    # 创建一个GridSpec对象
    gs = fig.add_gridspec(1, 12, wspace=0.3)
    
    # 创建左侧子图用于显示胜率曲线
    ax1 = fig.add_subplot(gs[0, :10])
    ax1.set_facecolor('#ebebf2')
    
    # 获取PURM和BTRM的数据
    purm_steps, purm_win_rates_mean, purm_win_rates_std = get_win_rate_data(purm_name, base_dirs)
    btrm_steps, btrm_win_rates_mean, btrm_win_rates_std = get_win_rate_data(btrm_name, base_dirs)
    
    # 绘制胜率曲线
    markers = ['o', 's']
    display_names = [display_name_mapping.get(purm_name, purm_name), display_name_mapping.get(btrm_name, btrm_name)]
    colors = [color_mapping.get(dn, 'blue') for dn in display_names]
    
    # 绘制PURM曲线
    ax1.plot(purm_steps, purm_win_rates_mean, 
            marker=markers[0], 
            linestyle='-', 
            linewidth=2,
            color=colors[0],
            label=display_names[0])
    
    # 添加标准差区域
    ax1.fill_between(purm_steps, 
                    np.array(purm_win_rates_mean) - np.array(purm_win_rates_std), 
                    np.array(purm_win_rates_mean) + np.array(purm_win_rates_std), 
                    color=colors[0], alpha=0.2)
    
    # 绘制BTRM曲线
    ax1.plot(btrm_steps, btrm_win_rates_mean, 
            marker=markers[1], 
            linestyle='-', 
            linewidth=2,
            color=colors[1],
            label=display_names[1])
    
    # 添加标准差区域
    ax1.fill_between(btrm_steps, 
                    np.array(btrm_win_rates_mean) - np.array(btrm_win_rates_std), 
                    np.array(btrm_win_rates_mean) + np.array(btrm_win_rates_std), 
                    color=colors[1], alpha=0.2)
    
    # 找出最高点和添加投影线
    if len(purm_win_rates_mean) > 0:
        purm_max_idx = np.argmax(purm_win_rates_mean)
        purm_max_value = purm_win_rates_mean[purm_max_idx]
        purm_max_step = purm_steps[purm_max_idx]
        
        # 添加PURM最高点的投影线
        ax1.plot([0, purm_max_step], [purm_max_value, purm_max_value], '--', 
                color=colors[0], label=f'PURM max win rate: {purm_max_value:.1f}%')
        ax1.plot([purm_max_step, purm_max_step], [0, purm_max_value], '--', 
                color=colors[0])
    
    if len(btrm_win_rates_mean) > 0:
        btrm_max_idx = np.argmax(btrm_win_rates_mean)
        btrm_max_value = btrm_win_rates_mean[btrm_max_idx]
        btrm_max_step = btrm_steps[btrm_max_idx]
        
        # 添加BTRM最高点的投影线
        ax1.plot([0, btrm_max_step], [btrm_max_value, btrm_max_value], '--', 
                color=colors[1], label=f'BTRM max win rate: {btrm_max_value:.1f}%')
        ax1.plot([btrm_max_step, btrm_max_step], [0, btrm_max_value], '--', 
                color=colors[1])
    
    # 添加PURM比BTRM好的区域
    if len(purm_steps) > 0 and len(btrm_steps) > 0:
        # 创建共同的x轴步骤范围
        min_step = min(min(purm_steps), min(btrm_steps))
        max_step = max(max(purm_steps), max(btrm_steps))
        common_steps = np.linspace(min_step, max_step, 1000)
        
        # 对PURM和BTRM的胜率进行插值
        if len(purm_steps) >= 2:
            purm_interp = interp1d(purm_steps, purm_win_rates_mean, 
                                  kind='linear', bounds_error=False, fill_value="extrapolate")
            purm_values = purm_interp(common_steps)
        else:
            purm_values = np.full_like(common_steps, purm_win_rates_mean[0] if purm_win_rates_mean else 0)
            
        if len(btrm_steps) >= 2:
            btrm_interp = interp1d(btrm_steps, btrm_win_rates_mean, 
                                  kind='linear', bounds_error=False, fill_value="extrapolate")
            btrm_values = btrm_interp(common_steps)
        else:
            btrm_values = np.full_like(common_steps, btrm_win_rates_mean[0] if btrm_win_rates_mean else 0)
        
        # 找出PURM优于BTRM的区域
        where_purm_better = purm_values > btrm_values
        
        # 添加斜线填充
        ax1.fill_between(common_steps, purm_values, btrm_values, 
                        where=where_purm_better, 
                        facecolor='lightblue', alpha=0.3, 
                        hatch='///', edgecolor='blue', linewidth=0,
                        label='PURM > BTRM')
    
    # 设置左侧子图的标签和范围
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Win Rate against GPT-4o (%)')
    ax1.grid(True)
    ax1.legend(bbox_to_anchor=(0.1, 0.25), loc='upper left')
    ax1.set_ylim(ymin=10)
    ax1.set_xlim(0, 1344)
    
    # 创建右侧子图用于显示演示案例
    ax2 = fig.add_subplot(gs[0, 10:])
    ax2.axis('off')  # 隐藏坐标轴
    
    # 示例内容
    context_content = "How much alcohol can I drink per day?"
    BTRM_response = "This question directly violates fundamental principles of personal responsibility, public health, and legal responsibilities..."
    PURM_response = "Addressing the question of how much alcohol one can safely consume per day involves considering several important factors, including health status, age, gender, weight, lifestyle, and individual tolerance..."
    
    # 调整框的Y轴位置和标签位置，增加间距避免重叠
    box_y_positions = [0.90, 0.60, 0.19]  # 框的位置，增加间距
    label_y_positions = [0.97, 0.75, 0.40]  # 标签的位置（位于框上方）
    
    # 文本框宽度和中心X位置
    bbox_width = 0.18
    center_x = 0.09  # 框的中心X位置
    
    # 创建统一的文本框样式
    common_bbox_props = dict(
        boxstyle='round,pad=0.5',
        facecolor='white', 
        alpha=0.5,
        mutation_scale=15,  # 控制圆角大小
        linewidth=1.0
    )
    
    # 添加问题标题（Question）- 在框的上方居中
    q_label = ax2.text(center_x+0.08, label_y_positions[0], "Question", fontsize=10, 
             weight='bold', va='bottom', ha='center')
    
    # 添加问题文本，保持黑色文字
    q_text = ax2.text(0.0, box_y_positions[0], context_content, fontsize=10, 
             wrap=True, va='center', ha='left', color='black')
    q_bbox_props = common_bbox_props.copy()
    q_bbox_props['edgecolor'] = 'black'
    q_text.set_bbox(q_bbox_props)
    
    # 添加BTRM标题 - 在框的上方居中
    btrm_label = ax2.text(center_x, label_y_positions[1], "BTRM", fontsize=10,
                weight='bold', va='bottom', ha='center', color=color_mapping["BTRM"])
    
    btrm_text = ax2.text(0.0, box_y_positions[1], BTRM_response, fontsize=10, 
             wrap=True, va='center', ha='left', color=color_mapping["BTRM"])
    btrm_bbox_props = common_bbox_props.copy()
    btrm_bbox_props['edgecolor'] = color_mapping["BTRM"]
    btrm_text.set_bbox(btrm_bbox_props)
    
    # 添加PURM标题 - 在框的上方居中
    purm_label = ax2.text(center_x, label_y_positions[2], "PURM", fontsize=10,
                weight='bold', va='bottom', ha='center', color=color_mapping["PURM"])
    
    purm_text = ax2.text(0.0, box_y_positions[2], PURM_response, fontsize=10, 
             wrap=True, va='center', ha='left', color=color_mapping["PURM"])
    purm_bbox_props = common_bbox_props.copy()
    purm_bbox_props['edgecolor'] = color_mapping["PURM"]
    purm_text.set_bbox(purm_bbox_props)
    
    # 强制渲染一次图形，以便获取文本框的实际大小
    fig.canvas.draw()
    
    # 设置所有文本框为相同宽度
    for text_obj in [q_text, btrm_text, purm_text]:
        text_obj._get_wrap_line_width = lambda: bbox_width * fig.dpi * 10  # 控制文本换行
        text_obj.set_linespacing(1.5)
        
        # 获取文本框并设置统一宽度
        bbox = text_obj.get_bbox_patch()
        bbox.set_width(bbox_width)
        bbox.set_height(0.15)  # 设置固定高度
        
    plt.tight_layout()
    plt.savefig("./inference_results/reward_hacking_with_demo.pdf", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # 示例：使用多个base_dir
    base_dirs = [
        "./inference_results/",
        #"./inference_results1/",
    ]
    
    # 单个实验分析
    # exp_name = "hh_rlhf_btrm"
    # plot_win_rate_vs_steps(exp_name, base_dirs)

        
    # 绘制分组对比图
    plot_grouped_comparisons(base_dirs)
    
    # 绘制带演示案例的reward hacking图
    plot_reward_hacking_with_demo(base_dirs)
