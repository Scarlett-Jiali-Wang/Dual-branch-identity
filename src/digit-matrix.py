import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


cm = np.array([[100, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [ 0, 97, 0, 0, 0, 0, 0, 3, 0, 0],
 [ 0, 0,97.1, 0, 0, 2.3, 0, 0, 0, 0],
 [ 0, 0, 0, 97.3, 0, 2.7, 0, 0, 0, 0],
 [ 0, 0, 0, 0, 97.7, 0, 2.3, 0, 0, 0],
 [ 0, 0, 2.1, 0, 0, 97.9, 0, 0, 0, 0],
 [ 0, 0, 0, 0, 0, 0, 98, 0, 0, 2],
 [ 0, 0, 0, 0, 0, 0, 0, 96, 0, 4],
 [ 0, 0, 3, 0, 0, 0, 0, 0, 97, 0],
 [ 0, 0, 0, 0, 0, 0, 0, 2.6, 0, 97.4]])

# 绘制混淆矩阵
fig_width = 16
fig_height = 13

# 绘制混淆矩阵
plt.figure(figsize=(fig_width, fig_height))

axn = sns.heatmap(cm, annot=True, fmt='.1f', cmap='Blues',
xticklabels=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
yticklabels=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
annot_kws={'size': 28})

cbar = axn.collections[0].colorbar
cbar.ax.tick_params(labelsize=56)# 设置颜色条刻度字体大小

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
plt.savefig('../result/confusion/digit-confusion-1.pdf', format='pdf', bbox_inches='tight', pad_inches=0.26)