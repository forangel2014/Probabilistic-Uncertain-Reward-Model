import numpy as np
import matplotlib.pyplot as plt
import torch

# 定义sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def plot_mc_approximation():

    # 生成x值
    plt.figure()
    x = np.linspace(-10, 10, 100)

    # 计算y值
    sigma = 0.5

    y1 = sigmoid(x)
    # y2 = \int sigmoid(z) N(z|x, \sigma) dz
    x = torch.tensor(x)

    # 定义不同的sigma值
    sigma_values = [0.1, 1.0, 5.0]

    # 绘制曲线
    for sigma in sigma_values:
        y2 = torch.mean(sigmoid(torch.randn(1000, len(x)) * sigma + x), dim=0).numpy()
        plt.plot(x, y2, label=f'Integral as Reward (sigma={sigma})')

    plt.plot(x, y1, label='Mean as Reward')
    plt.title('Reward Curve')
    plt.xlabel('r(y_w) - r(y_l)')
    plt.ylabel('p(y_w > y_l)')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('mc_approximation.pdf')

def plot_loss_function():
    # 生成x值
    plt.figure(figsize=(10, 4))

    x = np.linspace(-10, 10, 1000)
    
    # 计算sigmoid曲线
    y_sigmoid = sigmoid(x)
    
    # 绘制sigmoid曲线
    plt.plot(x, y_sigmoid, label=r'$\sigma(z)$', color='blue')
    
    # 在正半轴绘制方差较小的高斯分布
    mean_pos = 5  # 正半轴的均值
    sigma_small = 0.5  # 较小的方差
    x_small_range = np.linspace(mean_pos - 3*sigma_small, mean_pos + 3*sigma_small, 300)
    y_gaussian_small = (1/(sigma_small * np.sqrt(2 * np.pi)) * 
                        np.exp(-0.5 * ((x_small_range - mean_pos) / sigma_small) ** 2))
    plt.plot(x_small_range, y_gaussian_small, label=r'$\mathcal{N}(\mu_z, \sigma_z), \mu_z > 0$', color='green')
    
    # 在负半轴绘制方差较大的高斯分布
    mean_neg = -5  # 负半轴的均值
    sigma_large = 2.0  # 较大的方差
    x_large_range = np.linspace(mean_neg - 3*sigma_large, mean_neg + 3*sigma_large, 300)
    y_gaussian_large = (1/(sigma_large * np.sqrt(2 * np.pi)) * 
                        np.exp(-0.5 * ((x_large_range - mean_neg) / sigma_large) ** 2))
    plt.plot(x_large_range, y_gaussian_large, label=r'$\mathcal{N}(\mu_z, \sigma_z), \mu_z < 0$', color='red')
    
    # 设置图例和标题
    #plt.title('Sigmoid and Gaussian Curves')
    plt.text(10.0, -0.1, 'z', horizontalalignment='right', verticalalignment='center')
    #plt.ylabel(r'$p$')
    plt.legend()
    plt.xticks([0], ['0'])  # 只在0位置显示横坐标
    plt.yticks([])  # 去掉纵轴上的数值点
    
    # 绘制垂直虚线
    plt.axvline(x=0, color='black', linestyle='--')
    
    # 添加文本标注
    plt.text(-5, -0.1, 'Convex', fontsize=8, verticalalignment='center', horizontalalignment='center')
    plt.text(5, -0.1, 'Concave', fontsize=8, verticalalignment='center', horizontalalignment='center')
    
    # 隐藏所有边框
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # 添加箭头轴
    ax.annotate('', xy=(1.0, 0), xycoords=('axes fraction', 'data'), 
                xytext=(-0.01, 0), arrowprops=dict(arrowstyle='->', color='black'))
    
    # # 调整坐标标签位置
    # plt.text(1.02, -0.1, 'z', transform=ax.transAxes, 
    #         verticalalignment='center', horizontalalignment='left')
    
    plt.xticks([0], ['0'])
    plt.yticks([])
    
    plt.grid(False)
    plt.show()
    plt.savefig('loss_function.pdf', bbox_inches='tight')  # 去掉白边

def plot_normal_distributions():
    def norm_pdf(x, mu, sigma):
        return (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2)
    
    x = np.linspace(-20, 20, 1000)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 第一个子图（上）
    for mu, c in zip([0, -10, 10], ['blue', 'green', 'red']):
        y = norm_pdf(x, mu, 2)
        ax1.plot(x, y, color=c, label=f'N({mu}, 2)')
        ax1.set_xlim(-20, 20)

    # 第二个子图（下）
    for mu, c in zip([0, -1, 1], ['blue', 'green', 'red']):
        y = norm_pdf(x, mu, 1)
        ax2.plot(x, y, color=c, label=f'N({mu}, 1)')
        ax2.set_xlim(-10, 10)


    # 统一坐标范围
    for ax in [ax1, ax2]:
        ax.set_ylim(0, 0.45)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    # 标出交叠区域
    def fill_overlap(ax, base_mu, base_sigma, other_mus, color):
        base = norm_pdf(x, base_mu, base_sigma)
        for mu in other_mus:
            other = norm_pdf(x, mu, base_sigma)
            overlap = np.minimum(base, other)
            ax.fill_between(x, overlap, color=color, alpha=0.3)
    
    fill_overlap(ax1, 0, 2, [-10, 10], 'purple')
    fill_overlap(ax2, 0, 1, [-1, 1], 'purple')
    
    ax1.set_title('Blue Reward Distribution (σ=2, less uncertain)')
    ax2.set_title('Blue Reward Distribution (σ=1, more uncertain)')
    plt.tight_layout()
    plt.savefig('normal_overlap.pdf', bbox_inches='tight')
    plt.show()

def plot_sigmoid_derivatives():
    # 生成x值
    plt.figure(figsize=(10, 6))
    x = np.linspace(-10, 10, 1000)
    
    # 计算sigmoid函数
    y_sigmoid = sigmoid(x)
    
    # 计算一阶导数
    y_first_derivative = y_sigmoid * (1 - y_sigmoid)
    
    # 计算二阶导数
    y_second_derivative = y_first_derivative * (1 - 2 * y_sigmoid)
    
    # 计算三阶导数
    y_third_derivative = y_first_derivative * ((1 - 2 * y_sigmoid)**2 - 4 * y_first_derivative)
    
    # 绘制sigmoid函数
    plt.plot(x, y_sigmoid, label='$\\sigma(x)$', color='blue')
    
    # 绘制一阶导数
    plt.plot(x, y_first_derivative, label='$\\sigma\'(x)$', color='green')
    
    # 绘制二阶导数
    plt.plot(x, y_second_derivative, label='$\\sigma\'\'(x)$', color='red')
    
    # 绘制三阶导数
    plt.plot(x, y_third_derivative, label='$\\sigma\'\'\'(x)$', color='purple')
    
    # 添加网格、图例和标题
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    #plt.title('Sigmoid函数及其导数', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    
    # 添加x轴
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sigmoid_derivatives.pdf', bbox_inches='tight')
    plt.show()

#plot_mc_approximation()
#plot_loss_function()
#plot_normal_distributions()
plot_sigmoid_derivatives()