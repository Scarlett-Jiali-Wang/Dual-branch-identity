import os
import re
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time


from collections import Counter
from scipy.stats import skew, kurtosis
from sympy.printing.pretty.pretty_symbology import line_width
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    hinge_loss
)


# ============================================================
# 1. 数据读取
# ============================================================

def parse_user_digit_from_path(file_path):
    """
    从文件路径中解析 user_id 和 digit_id。

    默认路径格式：
        dataset/user_0/digit_2/sample_001.csv
    """
    path = file_path.replace("\\", "/")
    parts = path.split("/")

    user_id = None
    digit_id = None

    for part in parts:
        user_match = re.match(r"user[_-]?(\d+)", part)
        digit_match = re.match(r"digit[_-]?(\d+)", part)

        if user_match:
            user_id = int(user_match.group(1))

        if digit_match:
            digit_id = int(digit_match.group(1))

    if user_id is None or digit_id is None:
        raise ValueError(f"无法从路径解析 user_id 或 digit_id: {file_path}")

    return user_id, digit_id


def load_one_csv(file_path):
    """
    读取单个 csv 文件。

    返回：
        seq: shape = [T, C]
             T 为时序长度，C 为压力通道数。
    """
    df = pd.read_csv(file_path)

    # 只保留数值列
    df_num = df.select_dtypes(include=[np.number]).copy()

    # 删除 time / timestamp 列
    for col in list(df_num.columns):
        if col.lower() in ["time", "t", "timestamp"]:
            df_num = df_num.drop(columns=[col])

    if df_num.shape[1] == 0:
        raise ValueError(f"{file_path} 中没有有效压力数据列")

    seq = df_num.values.astype(np.float32)
    seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)

    return seq


def load_dataset(root_dir):
    """
    加载数据集。

    返回：
        sequences: list，每个元素为 [T, C]
        y_digit: 数字标签，0-9
        y_user: 用户标签，0-9
        file_paths: 文件路径
    """
    csv_files = glob.glob(os.path.join(root_dir, "**", "*.csv"), recursive=True)

    if len(csv_files) == 0:
        raise RuntimeError(f"在 {root_dir} 下没有找到 csv 文件")

    sequences = []
    y_digit = []
    y_user = []
    file_paths = []

    for file_path in csv_files:
        user_id, digit_id = parse_user_digit_from_path(file_path)

        seq = load_one_csv(file_path)

        sequences.append(seq)
        y_digit.append(digit_id)
        y_user.append(user_id)
        file_paths.append(file_path)

    if len(sequences) == 0:
        raise RuntimeError("没有加载到有效样本，请检查目录和文件命名")

    return (
        sequences,
        np.array(y_digit, dtype=np.int64),
        np.array(y_user, dtype=np.int64),
        file_paths
    )


# ============================================================
# 2. 不同长度时序的重采样与特征提取
# ============================================================

def resample_sequence(seq, target_len=128):
    """
    将不同长度的压力时序重采样到固定长度。

    输入：
        seq: [T, C]

    输出：
        resampled: [target_len, C]
    """
    T, C = seq.shape

    if T <= 1:
        return np.repeat(seq, target_len, axis=0)

    old_index = np.linspace(0, 1, T)
    new_index = np.linspace(0, 1, target_len)

    resampled = np.zeros((target_len, C), dtype=np.float32)

    for c in range(C):
        resampled[:, c] = np.interp(new_index, old_index, seq[:, c])

    return resampled


def extract_stat_features(seq):
    """
    从重采样后的压力时序中提取统计特征。

    输入：
        seq: [T, C]

    输出：
        一维特征向量
    """
    features = []

    # 基本统计特征
    features.extend(np.mean(seq, axis=0))
    features.extend(np.std(seq, axis=0))
    features.extend(np.min(seq, axis=0))
    features.extend(np.max(seq, axis=0))
    features.extend(np.median(seq, axis=0))
    features.extend(np.percentile(seq, 25, axis=0))
    features.extend(np.percentile(seq, 75, axis=0))
    features.extend(np.ptp(seq, axis=0))

    # 能量特征
    features.extend(np.sum(seq ** 2, axis=0))
    features.extend(np.sqrt(np.mean(seq ** 2, axis=0)))

    # 偏度和峰度
    features.extend(skew(seq, axis=0, nan_policy="omit"))
    features.extend(kurtosis(seq, axis=0, nan_policy="omit"))

    # 一阶差分，描述压力变化速度
    diff1 = np.diff(seq, axis=0)

    features.extend(np.mean(diff1, axis=0))
    features.extend(np.std(diff1, axis=0))
    features.extend(np.max(np.abs(diff1), axis=0))
    features.extend(np.sum(np.abs(diff1), axis=0))

    # 二阶差分，描述压力变化加速度
    if diff1.shape[0] > 1:
        diff2 = np.diff(diff1, axis=0)
        features.extend(np.mean(diff2, axis=0))
        features.extend(np.std(diff2, axis=0))
        features.extend(np.max(np.abs(diff2), axis=0))
    else:
        features.extend(np.zeros(seq.shape[1] * 3))

    features = np.array(features, dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


def sequence_to_feature(seq, target_len=128, use_resampled_raw=True):
    """
    将一条不同长度压力时序转为固定维度特征。

    特征包括：
        1. 重采样后的原始序列；
        2. 统计特征；
        3. 一阶、二阶差分特征。
    """
    seq_resampled = resample_sequence(seq, target_len=target_len)
    stat_features = extract_stat_features(seq_resampled)

    if use_resampled_raw:
        raw_features = seq_resampled.flatten()
        feature = np.concatenate([raw_features, stat_features])
    else:
        feature = stat_features

    return feature.astype(np.float32)


def build_feature_matrix(sequences, target_len=128, use_resampled_raw=True):
    """
    构造 SVM 输入特征矩阵 X。
    """
    X = []

    for seq in sequences:
        feature = sequence_to_feature(
            seq,
            target_len=target_len,
            use_resampled_raw=use_resampled_raw
        )
        X.append(feature)

    return np.vstack(X).astype(np.float32)


# ============================================================
# 3. 安全划分训练集和验证集
# ============================================================

def safe_train_val_split(
    X,
    y_digit,
    y_user,
    test_size=0.2,
    random_state=42
):
    n_samples = X.shape[0]

    if isinstance(test_size, float):
        n_test = math.ceil(n_samples * test_size)
    else:
        n_test = int(test_size)

    n_train = n_samples - n_test

    joint_label = y_user * 10 + y_digit

    candidates = [
        ("user", y_user),
        ("digit", y_digit),
        ("user-digit", joint_label)
    ]

    stratify_used = None
    split_mode = "random"

    for name, label in candidates:
        counts = Counter(label)
        num_classes = len(counts)
        min_count = min(counts.values())

        if min_count >= 2 and n_test >= num_classes and n_train >= num_classes:
            stratify_used = label
            split_mode = name
            break

    print(f"Final split mode: {split_mode}")

    return train_test_split(
        X,
        y_digit,
        y_user,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_used
    )


# ============================================================
# 4. 双分支 SVM 训练
# ============================================================

def train_dual_branch_svm(
    X_train,
    y_digit_train,
    y_user_train,
    X_val,
    y_digit_val,
    y_user_val,
    num_epochs=100,
    alpha=1e-4,
    random_state=42
):
    """
    双分支 SVM。

    分支 1：
        digit_svm，用于数字识别，类别为 0、1、2。

    分支 2：
        user_svm，用于用户识别，类别为 0-9。

    使用 SGDClassifier(loss="hinge")，等价于线性 SVM 的 SGD 优化形式。
    使用 partial_fit 实现逐轮训练，从而记录 loss 曲线。
    """

    digit_classes = np.unique(np.concatenate([y_digit_train, y_digit_val]))
    user_classes = np.unique(np.concatenate([y_user_train, y_user_val]))

    print(f"Digit classes: {digit_classes}")
    print(f"User classes: {user_classes}")

    digit_svm = SGDClassifier(
        loss="hinge",
        penalty="l2",
        alpha=alpha,
        learning_rate="optimal",
        max_iter=1,
        tol=None,
        random_state=random_state
    )

    user_svm = SGDClassifier(
        loss="hinge",
        penalty="l2",
        alpha=alpha,
        learning_rate="optimal",
        max_iter=1,
        tol=None,
        random_state=random_state + 1
    )

    history = {
        "digit_train_acc": [],
        "digit_val_acc": [],
        "digit_train_loss": [],
        "digit_val_loss": [],

        "user_train_acc": [],
        "user_val_acc": [],
        "user_train_loss": [],
        "user_val_loss": []
    }

    n_train = X_train.shape[0]
    rng = np.random.default_rng(random_state)

    # for epoch in range(num_epochs):
    for epoch in tqdm(range(num_epochs), desc="Epoch Progress", unit="epoch"):

        indices = rng.permutation(n_train)

        X_epoch = X_train[indices]
        y_digit_epoch = y_digit_train[indices]
        y_user_epoch = y_user_train[indices]

        if epoch == 0:
            digit_svm.partial_fit(
                X_epoch,
                y_digit_epoch,
                classes=digit_classes
            )
            user_svm.partial_fit(
                X_epoch,
                y_user_epoch,
                classes=user_classes
            )
        else:
            digit_svm.partial_fit(X_epoch, y_digit_epoch)
            user_svm.partial_fit(X_epoch, y_user_epoch)

        # -----------------------------
        # 数字分支
        # -----------------------------
        digit_train_pred = digit_svm.predict(X_train)
        digit_val_pred = digit_svm.predict(X_val)

        digit_train_acc = accuracy_score(y_digit_train, digit_train_pred)
        digit_val_acc = accuracy_score(y_digit_val, digit_val_pred)

        digit_train_score = digit_svm.decision_function(X_train)
        digit_val_score = digit_svm.decision_function(X_val)

        digit_train_loss = hinge_loss(
            y_digit_train,
            digit_train_score,
            labels=digit_classes
        )

        digit_val_loss = hinge_loss(
            y_digit_val,
            digit_val_score,
            labels=digit_classes
        )

        # -----------------------------
        # 用户分支
        # -----------------------------
        user_train_pred = user_svm.predict(X_train)
        user_val_pred = user_svm.predict(X_val)

        user_train_acc = accuracy_score(y_user_train, user_train_pred)
        user_val_acc = accuracy_score(y_user_val, user_val_pred)

        user_train_score = user_svm.decision_function(X_train)
        user_val_score = user_svm.decision_function(X_val)

        user_train_loss = hinge_loss(
            y_user_train,
            user_train_score,
            labels=user_classes
        )

        user_val_loss = hinge_loss(
            y_user_val,
            user_val_score,
            labels=user_classes
        )

        history["digit_train_acc"].append(digit_train_acc)
        history["digit_val_acc"].append(digit_val_acc)
        history["digit_train_loss"].append(digit_train_loss)
        history["digit_val_loss"].append(digit_val_loss)

        history["user_train_acc"].append(user_train_acc)
        history["user_val_acc"].append(user_val_acc)
        history["user_train_loss"].append(user_train_loss)
        history["user_val_loss"].append(user_val_loss)

        # if (epoch + 1) % 10 == 0 or epoch == 0:
        #     print(
        #         f"Epoch [{epoch + 1:03d}/{num_epochs}] | "
        #         f"Digit Acc: {digit_train_acc:.4f}/{digit_val_acc:.4f} | "
        #         f"User Acc: {user_train_acc:.4f}/{user_val_acc:.4f}"
        #     )

    return digit_svm, user_svm, history


# ============================================================
# 5. 用户认证评价
# ============================================================

def get_user_score_matrix(user_svm, X):
    """
    获取用户分支 SVM 的 decision_function 输出。

    返回：
        scores: [N, num_users]
    """
    scores = user_svm.decision_function(X)

    # 如果只有二分类，sklearn 可能返回 [N]，这里做兼容
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)

    return scores


def build_auth_scores(user_svm, X, true_users):
    """
    构造用户认证评价样本。

    对每个输入样本，构造多个 claim：
        1. claim 为真实用户：genuine，标签为 1；
        2. claim 为其他用户：impostor，标签为 0。

    输出：
        auth_scores: 声称用户对应的 SVM 分数；
        auth_labels: 1 为真实用户，0 为冒名用户。
    """
    score_matrix = get_user_score_matrix(user_svm, X)
    user_classes = user_svm.classes_

    auth_scores = []
    auth_labels = []

    for i in range(X.shape[0]):
        true_user = true_users[i]

        for class_index, claimed_user in enumerate(user_classes):
            score = score_matrix[i, class_index]
            label = 1 if claimed_user == true_user else 0

            auth_scores.append(score)
            auth_labels.append(label)

    return np.array(auth_scores), np.array(auth_labels)


def compute_far_frr(scores, labels, thresholds):
    """
    根据不同认证阈值计算 FAR 和 FRR。

    认证规则：
        score >= threshold: 接受
        score < threshold: 拒绝
    """
    fars = []
    frrs = []

    labels = np.array(labels)
    scores = np.array(scores)

    genuine_mask = labels == 1
    impostor_mask = labels == 0

    for th in thresholds:
        accept = scores >= th

        FAR = np.sum(accept & impostor_mask) / max(np.sum(impostor_mask), 1)
        FRR = np.sum((~accept) & genuine_mask) / max(np.sum(genuine_mask), 1)

        fars.append(FAR)
        frrs.append(FRR)

    return np.array(fars), np.array(frrs)


def find_eer(scores, labels, num_thresholds=500):
    """
    计算近似 EER。
    """
    thresholds = np.linspace(np.min(scores), np.max(scores), num_thresholds)
    fars, frrs = compute_far_frr(scores, labels, thresholds)

    idx = np.argmin(np.abs(fars - frrs))
    eer = (fars[idx] + frrs[idx]) / 2.0
    threshold = thresholds[idx]

    return threshold, eer, thresholds, fars, frrs


def authenticate_one_sample(user_svm, x_feature, claimed_user, threshold):
    """
    单个样本的认证。

    输入：
        x_feature: shape = [feature_dim]
        claimed_user: 声称用户编号
        threshold: 认证阈值

    输出：
        accept: True / False
        score: 声称用户对应分数
    """
    user_classes = user_svm.classes_

    if claimed_user not in user_classes:
        raise ValueError(f"claimed_user={claimed_user} 不在已训练用户类别中: {user_classes}")

    x_feature = x_feature.reshape(1, -1)
    scores = get_user_score_matrix(user_svm, x_feature)[0]

    class_index = np.where(user_classes == claimed_user)[0][0]
    score = scores[class_index]

    accept = score >= threshold

    return accept, score


# ============================================================
# 6. 绘图
# ============================================================

def plot_training_curves(history):
    epochs = np.arange(1, len(history["digit_train_acc"]) + 1)

    linewidth = 4

    # plt.figure(figsize=(7, 5))
    plt.figure(figsize=(fig_width, fig_height))
    plt.rcParams["font.family"] = "Arial"
    plt.plot(epochs, history["digit_train_acc"], label="Train Accuracy", linewidth=linewidth)
    plt.plot(epochs, history["digit_val_acc"], label="Validation Accuracy", linewidth=linewidth)

    plt.rcParams["font.family"] = "Arial"
    # 获取当前坐标轴
    ax = plt.gca()
    # 设置边框宽度
    ax.spines['top'].set_linewidth(5)  # 设置顶部边框宽度
    ax.spines['bottom'].set_linewidth(5)  # 设置底部边框宽度
    ax.spines['left'].set_linewidth(5)  # 设置左侧边框宽度
    ax.spines['right'].set_linewidth(5)  # 设置右侧边框宽度
    plt.tick_params(axis='both', labelsize=32, length=15, width=4, direction='in')  # 'both' 表示同时设置 x 轴和 y 轴的标尺字体大小

    plt.xlabel("Epoch", fontsize=font_size)
    plt.ylabel("Accuracy", fontsize=font_size)
    # plt.title("Digit Recognition Accuracy", fontsize=36)
    plt.legend(fontsize=font_size)
    # plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig('../result/acc-svg/Digit Recognition Accuracy.svg', format='svg', bbox_inches='tight', pad_inches=0.26)

    # plt.figure(figsize=(7, 5))
    plt.figure(figsize=(fig_width, fig_height))
    plt.rcParams["font.family"] = "Arial"
    plt.plot(epochs, history["digit_train_loss"], label="Train Hinge Loss", linewidth=linewidth)
    plt.plot(epochs, history["digit_val_loss"], label="Validation Hinge Loss", linewidth=linewidth)
    # 获取当前坐标轴
    ax = plt.gca()
    # 设置边框宽度
    ax.spines['top'].set_linewidth(5)  # 设置顶部边框宽度
    ax.spines['bottom'].set_linewidth(5)  # 设置底部边框宽度
    ax.spines['left'].set_linewidth(5)  # 设置左侧边框宽度
    ax.spines['right'].set_linewidth(5)  # 设置右侧边框宽度
    plt.tick_params(axis='both', labelsize=32, length=15, width=4, direction='in')  # 'both' 表示同时设置 x 轴和 y 轴的标尺字体大小

    plt.xlabel("Epoch", fontsize=font_size)
    plt.ylabel("Hinge Loss", fontsize=font_size)
    # plt.title("Digit Recognition Hinge Loss", fontsize=36)
    plt.legend(fontsize=font_size)
    # plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig('../result/acc-svg/Digit Recognition Hinge Loss.svg', format='svg', bbox_inches='tight', pad_inches=0.26)

    # plt.figure(figsize=(7, 5))
    plt.figure(figsize=(fig_width, fig_height))
    plt.plot(epochs, history["user_train_acc"], label="Train Accuracy", linewidth=linewidth)
    plt.plot(epochs, history["user_val_acc"], label="Validation Accuracy", linewidth=linewidth)
    # 获取当前坐标轴
    ax = plt.gca()
    # 设置边框宽度
    ax.spines['top'].set_linewidth(5)  # 设置顶部边框宽度
    ax.spines['bottom'].set_linewidth(5)  # 设置底部边框宽度
    ax.spines['left'].set_linewidth(5)  # 设置左侧边框宽度
    ax.spines['right'].set_linewidth(5)  # 设置右侧边框宽度
    plt.tick_params(axis='both', labelsize=32, length=15, width=4, direction='in')  # 'both' 表示同时设置 x 轴和 y 轴的标尺字体大小

    plt.xlabel("Epoch", fontsize=font_size)
    plt.ylabel("Accuracy", fontsize=font_size)
    # plt.title("User Identification Accuracy", fontsize=36)
    plt.legend(fontsize=font_size)
    # plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig('../result/acc-svg/User Identification Accuracy.svg', format='svg', bbox_inches='tight', pad_inches=0.26)

    # plt.figure(figsize=(7, 5))
    plt.figure(figsize=(fig_width, fig_height))
    plt.plot(epochs, history["user_train_loss"], label="Train Hinge Loss", linewidth=linewidth)
    plt.plot(epochs, history["user_val_loss"], label="Validation Hinge Loss", linewidth=linewidth)
    # 获取当前坐标轴
    ax = plt.gca()
    # 设置边框宽度
    ax.spines['top'].set_linewidth(5)  # 设置顶部边框宽度
    ax.spines['bottom'].set_linewidth(5)  # 设置底部边框宽度
    ax.spines['left'].set_linewidth(5)  # 设置左侧边框宽度
    ax.spines['right'].set_linewidth(5)  # 设置右侧边框宽度
    plt.tick_params(axis='both', labelsize=32, length=font_size, width=4, direction='in')  # 'both' 表示同时设置 x 轴和 y 轴的标尺字体大小

    plt.xlabel("Epoch", fontsize=font_size)
    plt.ylabel("Hinge Loss", fontsize=font_size)
    # plt.title("User Identification Hinge Loss", fontsize=36)
    plt.legend(fontsize=font_size)
    # plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig('../result/acc-svg/User Identification Hinge Loss.svg', format='svg', bbox_inches='tight', pad_inches=0.26)


def plot_far_frr(thresholds, fars, frrs, eer, eer_threshold):
    # plt.figure(figsize=(7, 5))
    linewidth = 4
    plt.figure(figsize=(fig_width, fig_height))
    plt.plot(thresholds, fars, label="FAR", linewidth=linewidth)
    plt.plot(thresholds, frrs, label="FRR", linewidth=linewidth)

    plt.rcParams["font.family"] = "Arial"
    # 获取当前坐标轴
    ax = plt.gca()

    # 设置边框宽度
    ax.spines['top'].set_linewidth(5)  # 设置顶部边框宽度
    ax.spines['bottom'].set_linewidth(5)  # 设置底部边框宽度
    ax.spines['left'].set_linewidth(5)  # 设置左侧边框宽度
    ax.spines['right'].set_linewidth(5)  # 设置右侧边框宽度
    plt.tick_params(axis='both', labelsize=32, length=15, width=4, direction='in')  # 'both' 表示同时设置 x 轴和 y 轴的标尺字体大小

    plt.axvline(
        eer_threshold,
        linestyle="--",
        label=f"EER threshold = {eer_threshold:.4f}"
    )
    plt.xlabel("Threshold", fontsize=font_size)
    plt.ylabel("Rate", fontsize=font_size)
    # plt.title(f"Authentication FAR / FRR Curve, EER ≈ {eer:.4f}", fontsize=36)
    plt.legend(fontsize=font_size)
    # plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig('../result/acc-svg/Authentication FARFRR Curve.svg', format='svg', bbox_inches='tight', pad_inches=0.26)


# ============================================================
# 7. 主程序
# ============================================================

fig_width = 18
fig_height = 15

font_size = 36

def main():
    # --------------------------------------------------------
    # 修改这里为你的数据集路径
    # --------------------------------------------------------
    root_dir = "../dataset"

    # --------------------------------------------------------
    # 参数设置
    # --------------------------------------------------------
    target_len = 512

    use_resampled_raw = True

    test_size = 0.35
    random_state = 42

    num_epochs = 10000

    alpha = 1e-5

    # --------------------------------------------------------
    # 加载数据
    # --------------------------------------------------------
    print("Loading dataset...")
    sequences, y_digit, y_user, file_paths = load_dataset(root_dir)

    print(f"Total samples: {len(sequences)}")
    print(f"Digit labels: {np.unique(y_digit)}")
    print(f"User labels: {np.unique(y_user)}")

    print("\nDigit label count:")
    print(Counter(y_digit))

    print("\nUser label count:")
    print(Counter(y_user))

    print("\nUser-Digit label count:")
    print(Counter(y_user * 10 + y_digit))

    # --------------------------------------------------------
    # 构造特征矩阵
    # --------------------------------------------------------
    print("\nBuilding feature matrix...")
    X = build_feature_matrix(
        sequences,
        target_len=target_len,
        use_resampled_raw=use_resampled_raw
    )

    print(f"Feature matrix shape: {X.shape}")

    # --------------------------------------------------------
    # 划分训练集和验证集
    # --------------------------------------------------------
    X_train, X_val, y_digit_train, y_digit_val, y_user_train, y_user_val = safe_train_val_split(
        X,
        y_digit,
        y_user,
        test_size=test_size,
        random_state=random_state
    )

    print(f"\nTrain samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")

    # --------------------------------------------------------
    # 标准化
    # --------------------------------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # --------------------------------------------------------
    # 训练双分支 SVM
    # --------------------------------------------------------
    print("\nTraining dual-branch SVM...")
    epoch_start = time.time()
    digit_svm, user_svm, history = train_dual_branch_svm(
        X_train,
        y_digit_train,
        y_user_train,
        X_val,
        y_digit_val,
        y_user_val,
        num_epochs=num_epochs,
        alpha=alpha,
        random_state=random_state
    )

    epoch_end = time.time()
    print(f"用时: {epoch_end - epoch_start:.2f} 秒")

    # --------------------------------------------------------
    # 数字识别评价
    # --------------------------------------------------------
    print("\n========== Digit Recognition Result ==========")
    digit_val_pred = digit_svm.predict(X_val)

    print(classification_report(
        y_digit_val,
        digit_val_pred,
        labels=np.unique(y_digit),
        digits=4,
        zero_division=0
    ))

    print("Digit Confusion Matrix:")
    print(confusion_matrix(
        y_digit_val,
        digit_val_pred,
        labels=np.unique(y_digit)
    ))

    # --------------------------------------------------------
    # 用户识别评价
    # --------------------------------------------------------
    print("\n========== User Identification Result ==========")
    user_val_pred = user_svm.predict(X_val)

    print(classification_report(
        y_user_val,
        user_val_pred,
        labels=np.unique(y_user),
        digits=4,
        zero_division=0
    ))

    print("User Confusion Matrix:")
    print(confusion_matrix(
        y_user_val,
        user_val_pred,
        labels=np.unique(y_user)
    ))

    # --------------------------------------------------------
    # 用户认证评价
    # --------------------------------------------------------
    print("\n========== User Authentication Result ==========")

    auth_scores, auth_labels = build_auth_scores(
        user_svm,
        X_val,
        y_user_val
    )

    eer_threshold, eer, thresholds, fars, frrs = find_eer(
        auth_scores,
        auth_labels
    )

    print(f"Authentication threshold based on EER: {eer_threshold:.6f}")
    print(f"Approximate EER: {eer:.6f}")

    # --------------------------------------------------------
    # 绘图
    # --------------------------------------------------------
    plot_training_curves(history)
    plot_far_frr(thresholds, fars, frrs, eer, eer_threshold)

    # --------------------------------------------------------
    # 保存模型
    # --------------------------------------------------------
    os.makedirs("../saved_models", exist_ok=True)

    joblib.dump(scaler, "../saved_models/scaler.pkl")
    joblib.dump(digit_svm, "../saved_models/digit_svm.pkl")
    joblib.dump(user_svm, "../saved_models/user_svm.pkl")

    config = {
        "target_len": target_len,
        "use_resampled_raw": use_resampled_raw,
        "eer_threshold": eer_threshold,
        "digit_classes": digit_svm.classes_,
        "user_classes": user_svm.classes_
    }

    joblib.dump(config, "../saved_models/config.pkl")

    print("\nModels saved to folder: saved_models/")


if __name__ == "__main__":
    main()