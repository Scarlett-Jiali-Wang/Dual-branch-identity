import numpy as np
import matplotlib.pyplot as plt

# =========================
# Data
# =========================
models = ["Dual-SVM", "1D-CNN", "LSTM"]

digit_acc = [97.54, 86.25, 76.87]
user_acc = [97.14, 79.37, 67.43]

fig_width = 18
fig_height = 15

font_size = 36

# =========================
# Plot settings
# =========================
x = np.arange(len(models))
bar_width = 0.35

plt.figure(figsize=(fig_width, fig_height))
plt.rcParams["font.family"] = "Arial"
colors = ["#4E9A46", "#F07A26", "#0B43B8"]

bars1 = plt.bar(
    x - bar_width / 2,
    digit_acc,
    width=bar_width,
    label="Digit recognition",
)

bars2 = plt.bar(
    x + bar_width / 2,
    user_acc,
    width=bar_width,
    label="User identification",
)

# =========================
# Add value labels
# =========================
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.8,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=font_size
        )

add_labels(bars1)
add_labels(bars2)

plt.tick_params(axis='both', labelsize=32, length=15, width=4, direction='in')  # 'both' 表示同时设置 x 轴和 y 轴的标尺字体大小
ax = plt.gca()
# 设置边框宽度
ax.spines['top'].set_linewidth(5)  # 设置顶部边框宽度
ax.spines['bottom'].set_linewidth(5)  # 设置底部边框宽度
ax.spines['left'].set_linewidth(5)  # 设置左侧边框宽度
ax.spines['right'].set_linewidth(5)  # 设置右侧边框宽度

# =========================
# Axes and labels
# =========================
plt.xticks(x, models, fontsize=font_size)
plt.ylabel("Accuracy (%)", fontsize=font_size)
# plt.ylim(0, 110)

# plt.title(
#     "Comparison of Digit Recognition and User Identification Accuracy",
#     fontsize=14,
#     fontweight="bold",
# )

plt.legend(fontsize=font_size)
# plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

# =========================
# Save figure
# =========================
plt.savefig('../result/bar-acc.pdf', format='pdf', bbox_inches='tight', pad_inches=0.26)
#
# plt.show()