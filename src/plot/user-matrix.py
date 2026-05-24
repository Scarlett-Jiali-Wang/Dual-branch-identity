import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


cm = np.array([
    [98, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 98.7, 1.3, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 96.4, 3.6, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 98, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 98, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 98, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 98, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 98, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 97, 3],
    [2.3, 0, 0, 0, 0, 0, 0, 0, 0, 97.7]
])

# fig_width = 8.7
# fig_height = 6.8

fig_width = 16
fig_height = 13

# 绘制混淆矩阵
plt.figure(figsize=(fig_width, fig_height))

axn = sns.heatmap(cm, annot=True, fmt='.1f', cmap='Blues',
xticklabels=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
yticklabels=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
annot_kws={'size': 35})

cbar = axn.collections[0].colorbar
cbar.ax.tick_params(labelsize=56)# 设置颜色条刻度字体大小
cbar.mappable.set_clim(0, 100)  # 方式B：通过 colorbar 关联的对象设置

# plt.title('Confusion Matrix', fontsize=36)
plt.rcParams["font.family"] = "Arial"
# 获取当前坐标轴
ax = plt.gca()
plt.xticks(fontsize=64)
plt.yticks(fontsize=64)
plt.xlabel('Predicted label', fontsize=72)
plt.ylabel('True label', fontsize=72)
plt.tight_layout()
# plt.show()
plt.savefig('../result/confusion/user-confusion-1.pdf', format='pdf', bbox_inches='tight', pad_inches=0.26)