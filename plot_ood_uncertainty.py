import matplotlib.pyplot as plt
import numpy as np

# 准备数据
data = {
    "Chat (in-domain)": 0.031,
    "argilla_math": 0.058,
    "sdiazlor_math": 0.050,
    "dzunggg_legal": 0.058,
    "HC3-Chinese": 0.046,
    "Aratako_Japanese": 0.065
}

# 区分颜色和标签
colors = []
values = []
for k, v in data.items():
    values.append(v)
    colors.append("#2c7bb6" if "(in-domain)" in k else "#d7191c")  # 蓝色表示in-domain，红色表示OOD

# 绘图设置
plt.figure(figsize=(10, 6), dpi=100, facecolor='#f8f8f8')
ax = plt.gca()
#ax.set_facecolor('#ebebf2')
bars = plt.bar(data.keys(), values, color=colors, width=0.6)

# 添加数据标签
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f"{height:.3f}", ha='center', va='bottom')

# 样式调整
plt.xticks(rotation=15, ha='right', fontsize=10)
plt.ylabel("Uncertainty", fontsize=12)
#plt.title("Uncertainty Comparison: In-Domain vs OOD Domains", pad=20)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 添加图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#2c7bb6", label='In-Domain (Chat)'),
    Patch(facecolor="#d7191c", label='OOD Domains')
]
plt.legend(handles=legend_elements, loc='upper left')

plt.tight_layout()
plt.savefig("./ood_uncertainty.pdf", bbox_inches='tight')
plt.close()