import numpy as np
import matplotlib.pyplot as plt

# =========================
# Data
# =========================
models = ["Dual-SVM", "1D-CNN", "LSTM"]
digit_acc = [2.857, 17.847, 35.61]

# =========================
# Plot
# =========================
x = np.arange(len(models))

fig_width = 18
fig_height = 15

font_size = 36

plt.figure(figsize=(fig_width, fig_height))
plt.rcParams["font.family"] = "Arial"
colors = ["#4E9A46", "#F07A26", "#0B43B8"]
bars = plt.bar(
    x,
    digit_acc,
    width=0.55,
    color=colors,
)


# =========================
# Value labels
# =========================
for bar, value in zip(bars, digit_acc):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.8,
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontsize=font_size
    )

# =========================
# Axes and labels
# =========================
plt.xticks(x, models, fontsize=font_size)
plt.ylabel("False accept rate (%)", fontsize=font_size)
plt.ylim(0, 42)
ax = plt.gca()
# 设置边框宽度
ax.spines['top'].set_linewidth(5)  # 设置顶部边框宽度
ax.spines['bottom'].set_linewidth(5)  # 设置底部边框宽度
ax.spines['left'].set_linewidth(5)  # 设置左侧边框宽度
ax.spines['right'].set_linewidth(5)  # 设置右侧边框宽度
plt.tick_params(axis='both', labelsize=32, length=15, width=4, direction='in')  # 'both' 表示同时设置 x 轴和 y 轴的标尺字体大小

plt.tight_layout()

# =========================
# Save
# =========================
plt.savefig('../result/bar-far.pdf', format='pdf', bbox_inches='tight', pad_inches=0.26)