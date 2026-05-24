import matplotlib.pyplot as plt
import numpy as np

scenarios = ["Zero-effort\nimpostor attack", "Same-digit\nimpostor attack"]
digit_recognition_rate = [11.6, 20.5]
user_auth_far = [0.16, 0.18]

x = np.arange(len(scenarios))
width = 0.34

fig_width = 18
fig_height = 15

font_size = 36

plt.rcParams["font.family"] = "Arial"
colors = ["#4E9A46", "#F07A26", "#0B43B8"]

plt.figure(figsize=(fig_width, fig_height))

bars1 = plt.bar(
    x - width / 2,
    digit_recognition_rate,
    width,
    label="Digit recognition rate"
)

bars2 = plt.bar(
    x + width / 2,
    user_auth_far,
    width,
    label="User authentication FAR"
)

# plt.yscale("log")
plt.ylabel("Rate (%)", fontsize=font_size)
plt.xticks(x, scenarios, fontsize=font_size)
plt.xlabel("Attack scenario", fontsize=font_size)
plt.legend(frameon=False, fontsize=font_size)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        label = f"{height:.2f}%" if height < 1 else f"{height:.1f}%"
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            label,
            ha="center",
            va="bottom",
            fontsize=36
        )

plt.tick_params(axis='both', labelsize=32, length=15, width=4, direction='in')  # 'both' 表示同时设置 x 轴和 y 轴的标尺字体大小

ax = plt.gca()
ax.spines['top'].set_linewidth(5)  # 设置顶部边框宽度
ax.spines['bottom'].set_linewidth(5)  # 设置底部边框宽度
ax.spines['left'].set_linewidth(5)  # 设置左侧边框宽度
ax.spines['right'].set_linewidth(5)  # 设置右侧边框宽度

# plt.title("Digit Recognition and User Authentication FAR under Impostor Attacks")
plt.tight_layout()
# plt.savefig("impostor_attack_bar_chart.png", dpi=300, bbox_inches="tight")
plt.savefig("impostor_attack_bar_chart.pdf", bbox_inches="tight")
plt.show()