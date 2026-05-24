import pandas as pd
import matplotlib.pyplot as plt

user = 4

# 1. CSV 文件路径
csv_file = "../dataset/user_" + str(user) + "/digit_3/sample_001.csv"
read_rows = 150 # None 表示读取全部；例如 100 表示只读取前 100 行

digit_color = ["#1037A6", "#FC6A24", "#086B05", "#E2181A", "#6413AC",
               "#5F271C", "#C6309F", "#393939", "#33BEC2", "#9F9C2A"]

# 2. 读取 CSV 文件
df = pd.read_csv(csv_file, nrows=read_rows)

# 3. 确保 time 和 pressure 是数值类型
df["time"] = pd.to_numeric(df["time"], errors="coerce")
df["pressure"] = pd.to_numeric(df["pressure"], errors="coerce")

# 4. 删除无效数据行
# df = df.dropna(subset=["time", "pressure"])

# 5. 按 time 排序，避免时间顺序混乱
df = df.sort_values(by="time")

# 6. 绘制折线图
plt.figure(figsize=(18, 5))
plt.plot(df["time"], df["pressure"], linewidth=4, color=digit_color[user])

plt.rcParams["font.family"] = "Arial"
# 获取当前坐标轴
ax = plt.gca()

# 设置边框宽度
ax.spines['top'].set_linewidth(5)  # 设置顶部边框宽度
ax.spines['bottom'].set_linewidth(5)  # 设置底部边框宽度
ax.spines['left'].set_linewidth(5)  # 设置左侧边框宽度
ax.spines['right'].set_linewidth(5)  # 设置右侧边框宽度
plt.tick_params(axis='both', labelsize=36, length=15, width=4, direction='in')  # 'both' 表示同时设置 x 轴和 y 轴的标尺字体大小
# ax.tick_params(axis='x', labelbottom=False)

# plt.xlabel("Time", fontsize=36)
# plt.ylabel("Pressure", fontsize=36)
# plt.title("Pressure vs Time")
# plt.grid(True)

# plt.tight_layout()
# plt.show()
filename = "../result/user-plot/user_" + str(user) +".svg"
plt.savefig(filename, format='svg', bbox_inches='tight', pad_inches=0.26)