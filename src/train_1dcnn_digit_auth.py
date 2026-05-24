'''
python train_1dcnn_digit_auth.py --data_dir dataset --seq_len 256 --epochs 60
'''

import argparse
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import re
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
from torch.utils.data import Dataset, DataLoader


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_records(data_dir: str):
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {root}")

    user_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("user_")])
    if len(user_dirs) < 2:
        raise ValueError("At least 2 users are required for FAR/FRR evaluation.")

    user_to_idx = {p.name: i for i, p in enumerate(user_dirs)}

    records = []
    for user_dir in user_dirs:
        user_idx = user_to_idx[user_dir.name]
        digit_dirs = sorted([p for p in user_dir.iterdir() if p.is_dir() and p.name.startswith("digit_")])

        for digit_dir in digit_dirs:
            digit = int(digit_dir.name.split("_")[-1])

            for csv_path in sorted(digit_dir.glob("*.csv")):
                records.append({
                    "path": str(csv_path),
                    "user": user_idx,
                    "user_name": user_dir.name,
                    "digit": digit,
                })

    if not records:
        raise ValueError("No CSV files found.")

    return records, user_to_idx


def split_records(records, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Closed-set split:
    每个 user/digit 内部划分 train/val/test。
    这样用户认证时，每个注册用户都有 enrollment 样本和 test 样本。

    如果你们后续有 session 信息，建议进一步改成 session-wise split，
    避免同一采集 session 同时出现在训练和测试中。
    """
    rng = random.Random(seed)
    buckets = defaultdict(list)

    for r in records:
        buckets[(r["user"], r["digit"])].append(r)

    train, val, test = [], [], []

    for _, items in buckets.items():
        items = items[:]
        rng.shuffle(items)
        n = len(items)

        if n >= 5:
            n_train = max(1, int(round(n * train_ratio)))
            n_val = max(1, int(round(n * val_ratio)))

            if n_train + n_val >= n:
                n_train = max(1, n - 2)
                n_val = 1

            n_test = n - n_train - n_val

        elif n == 4:
            n_train, n_val, n_test = 2, 1, 1
        elif n == 3:
            n_train, n_val, n_test = 1, 1, 1
        elif n == 2:
            n_train, n_val, n_test = 1, 0, 1
            # n_train, n_val, n_test = 1, 1, 0
        else:
            n_train, n_val, n_test = 1, 0, 0

        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:n_train + n_val + n_test])

    if len(val) == 0 or len(test) == 0:
        raise ValueError("Validation/test split is empty. Please collect more samples per user per digit.")

    return train, val, test


def temporal_binning(t: np.ndarray, p: np.ndarray, seq_len: int) -> np.ndarray:
    """
    将变长信号转换为固定长度序列。
    不是 spline interpolation，而是 temporal binning：
    每个归一化时间窗口内取原始压力点的均值。
    """
    if len(p) == 0:
        return np.zeros(seq_len, dtype=np.float32)

    duration = float(t[-1] - t[0])
    if duration <= 1e-12:
        tau = np.linspace(0.0, 1.0, len(p), dtype=np.float32)
    else:
        tau = ((t - t[0]) / duration).astype(np.float32)

    bin_ids = np.floor(tau * seq_len).astype(np.int64)
    bin_ids = np.clip(bin_ids, 0, seq_len - 1)

    sums = np.zeros(seq_len, dtype=np.float32)
    counts = np.zeros(seq_len, dtype=np.float32)

    np.add.at(sums, bin_ids, p.astype(np.float32))
    np.add.at(counts, bin_ids, 1.0)

    x = np.zeros(seq_len, dtype=np.float32)
    nonempty = counts > 0
    x[nonempty] = sums[nonempty] / counts[nonempty]

    # 空 bin 用前向保持填充，不做样条插值
    if nonempty.any():
        first = int(np.argmax(nonempty))
        x[:first] = x[first]

        last_val = x[first]
        for i in range(first + 1, seq_len):
            if nonempty[i]:
                last_val = x[i]
            else:
                x[i] = last_val

    return x.astype(np.float32)


def preprocess_csv(csv_path: str, seq_len: int) -> np.ndarray:
    df = pd.read_csv(csv_path)

    cols = {c.lower().strip(): c for c in df.columns}
    if "time" not in cols or "pressure" not in cols:
        raise ValueError(f"CSV must contain 'time' and 'pressure' columns: {csv_path}")

    t = pd.to_numeric(df[cols["time"]], errors="coerce").to_numpy(dtype=np.float32)
    p = pd.to_numeric(df[cols["pressure"]], errors="coerce").to_numpy(dtype=np.float32)

    mask = np.isfinite(t) & np.isfinite(p)
    t, p = t[mask], p[mask]

    if len(p) < 2:
        return np.zeros(seq_len, dtype=np.float32)

    order = np.argsort(t)
    t, p = t[order], p[order]

    # 1. 基线校正
    n_base = max(3, int(0.05 * len(p)))
    baseline = float(np.median(p[:n_base]))
    noise_std = float(np.std(p[:n_base])) + 1e-8

    p = p - baseline
    p[p < 0] = 0.0

    # 2. 截取有效书写段
    amp = float(np.max(p))
    if amp > 1e-8:
        threshold = max(0.03 * amp, 3.0 * noise_std)
        active = np.where(p > threshold)[0]

        if len(active) > 0:
            left = max(0, int(active[0]) - 2)
            right = min(len(p), int(active[-1]) + 3)
            t, p = t[left:right], p[left:right]

    # 3. 幅值归一化
    amp = float(np.max(np.abs(p)))
    if amp > 1e-8:
        p = p / amp

    # 4. 固定长度序列
    x = temporal_binning(t, p, seq_len)
    return x.astype(np.float32)


class PressureDataset(Dataset):
    def __init__(self, records, seq_len=256, cache=True):
        self.records = records
        self.seq_len = seq_len
        self.cache = cache

        if cache:
            self.x_cache = [preprocess_csv(r["path"], seq_len) for r in records]
        else:
            self.x_cache = None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]

        if self.cache:
            x = self.x_cache[idx]
        else:
            x = preprocess_csv(r["path"], self.seq_len)

        x = torch.from_numpy(x).float().unsqueeze(0)  # [1, seq_len]
        digit = torch.tensor(r["digit"], dtype=torch.long)
        user = torch.tensor(r["user"], dtype=torch.long)

        return x, digit, user


class MultiTask1DCNN(nn.Module):
    def __init__(self, num_users: int, num_digits: int = 10, emb_dim: int = 64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
        )

        self.embedding = nn.Sequential(
            nn.Linear(128, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )

        self.digit_head = nn.Linear(emb_dim, num_digits)
        self.user_head = nn.Linear(emb_dim, num_users)

    def forward(self, x):
        h = self.encoder(x).squeeze(-1)
        emb = self.embedding(h)

        digit_logits = self.digit_head(emb)
        user_logits = self.user_head(emb)

        return digit_logits, user_logits, emb


def run_one_epoch(model, loader, optimizer, device, user_loss_weight=1.0):
    model.train()

    total_loss = 0.0
    total = 0

    for x, digit, user in loader:
        x = x.to(device)
        digit = digit.to(device)
        user = user.to(device)

        optimizer.zero_grad(set_to_none=True)

        digit_logits, user_logits, _ = model(x)

        loss_digit = F.cross_entropy(digit_logits, digit)
        loss_user = F.cross_entropy(user_logits, user)

        loss = loss_digit + user_loss_weight * loss_user

        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        total_loss += float(loss.item()) * batch_size
        total += batch_size

    return total_loss / max(total, 1)


@torch.no_grad()
def evaluate_classification(model, loader, device):
    model.eval()

    correct_digit = 0
    correct_user = 0
    total = 0

    for x, digit, user in loader:
        x = x.to(device)
        digit = digit.to(device)
        user = user.to(device)

        digit_logits, user_logits, _ = model(x)

        pred_digit = digit_logits.argmax(dim=1)
        pred_user = user_logits.argmax(dim=1)

        correct_digit += int((pred_digit == digit).sum().item())
        correct_user += int((pred_user == user).sum().item())
        total += int(x.size(0))

    return {
        "digit_acc": correct_digit / max(total, 1),
        "user_identification_acc": correct_user / max(total, 1),
    }


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()

    embeddings = []
    users = []
    digits = []

    for x, digit, user in loader:
        x = x.to(device)

        _, _, emb = model(x)
        emb = F.normalize(emb, p=2, dim=1)

        embeddings.append(emb.cpu().numpy())
        users.append(user.numpy())
        digits.append(digit.numpy())

    embeddings = np.vstack(embeddings)
    users = np.concatenate(users)
    digits = np.concatenate(digits)

    return embeddings, users, digits


def build_user_centroids(train_emb, train_users, num_users):
    centroids = []

    for u in range(num_users):
        idx = np.where(train_users == u)[0]

        if len(idx) == 0:
            raise ValueError(f"No enrollment samples for user {u}")

        c = train_emb[idx].mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-12)

        centroids.append(c)

    return np.vstack(centroids).astype(np.float32)


def make_auth_scores(emb, users, centroids, max_impostors_per_sample=None, seed=42):
    """
    genuine trial:
        样本声称自己属于真实用户。

    impostor trial:
        样本冒充其他用户。

    FAR:
        impostor 被错误接受的比例。

    FRR / FFR:
        genuine 被错误拒绝的比例。
    """
    rng = np.random.default_rng(seed)

    score_matrix = emb @ centroids.T  # cosine similarity

    genuine_scores = score_matrix[np.arange(len(users)), users]

    impostor_scores = []
    all_users = np.arange(centroids.shape[0])

    for i, true_user in enumerate(users):
        impostor_users = all_users[all_users != true_user]

        if max_impostors_per_sample is not None and len(impostor_users) > max_impostors_per_sample:
            impostor_users = rng.choice(
                impostor_users,
                size=max_impostors_per_sample,
                replace=False,
            )

        impostor_scores.append(score_matrix[i, impostor_users])

    impostor_scores = np.concatenate(impostor_scores)

    return genuine_scores, impostor_scores


def auth_metrics_at_threshold(genuine_scores, impostor_scores, threshold):
    far = float(np.mean(impostor_scores >= threshold))
    frr = float(np.mean(genuine_scores < threshold))
    return far, frr


def choose_threshold_by_val_eer(genuine_scores, impostor_scores):
    """
    在验证集上选择 FAR 和 FRR 最接近的阈值。
    """
    thresholds = np.linspace(-1.0, 1.0, 2001, dtype=np.float32)

    fars = []
    frrs = []

    for th in thresholds:
        far, frr = auth_metrics_at_threshold(genuine_scores, impostor_scores, th)
        fars.append(far)
        frrs.append(frr)

    fars = np.array(fars)
    frrs = np.array(frrs)

    idx = int(np.argmin(np.abs(fars - frrs)))

    return float(thresholds[idx]), float(fars[idx]), float(frrs[idx])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="../cnn-dataset")
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--user_loss_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cache", action="store_true")

    parser.add_argument(
        "--max_impostors_per_sample",
        type=int,
        default=None,
        help="默认使用所有冒充用户。数据很大时可设为 20 或 50 加速。",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records, user_to_idx = build_records(args.data_dir)
    num_users = len(user_to_idx)

    train_records, val_records, test_records = split_records(
        records,
        seed=args.seed,
    )

    print(f"Users: {num_users}")
    print(f"Samples: total={len(records)}, train={len(train_records)}, val={len(val_records)}, test={len(test_records)}")
    print(f"Device: {device}")

    cache = not args.no_cache

    train_ds = PressureDataset(train_records, seq_len=args.seq_len, cache=cache)
    val_ds = PressureDataset(val_records, seq_len=args.seq_len, cache=cache)
    test_ds = PressureDataset(test_records, seq_len=args.seq_len, cache=cache)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = MultiTask1DCNN(
        num_users=num_users,
        num_digits=10,
        emb_dim=64,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_state = None
    best_score = -1.0

    history = {
        "val_digit_acc": [],
        "val_user_id_acc": [],
        "loss": []
    }

    for epoch in range(1, args.epochs + 1):
        train_loss = run_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            user_loss_weight=args.user_loss_weight,
        )

        val_metrics = evaluate_classification(model, val_loader, device)

        score = val_metrics["digit_acc"] + val_metrics["user_identification_acc"]

        if score > best_score:
            best_score = score
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
        history["val_digit_acc"].append(val_metrics['digit_acc'])
        history["val_user_id_acc"].append(val_metrics['user_identification_acc'])
        history["loss"].append(train_loss)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"loss={train_loss:.4f} | "
            f"val_digit_acc={val_metrics['digit_acc']:.4f} | "
            f"val_user_id_acc={val_metrics['user_identification_acc']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_classification(model, val_loader, device)

    # 认证部分：
    # train set 作为 enrollment set；
    # val set 选择认证阈值；
    # test set 计算最终 FAR / FRR。
    train_emb, train_users, _ = extract_embeddings(model, train_loader, device)
    val_emb, val_users, _ = extract_embeddings(model, val_loader, device)
    test_emb, test_users, _ = extract_embeddings(model, test_loader, device)

    centroids = build_user_centroids(train_emb, train_users, num_users)

    val_genuine, val_impostor = make_auth_scores(
        val_emb,
        val_users,
        centroids,
        max_impostors_per_sample=args.max_impostors_per_sample,
        seed=args.seed,
    )

    threshold, val_far, val_frr = choose_threshold_by_val_eer(
        val_genuine,
        val_impostor,
    )

    plot_training_curves(history)

    # 绘制 validation set 的 FAR/FRR-threshold 曲线
    val_thresholds, val_fars, val_frrs, val_eer_threshold, val_eer = compute_far_frr_curve(
        val_genuine,
        val_impostor,
        num_thresholds=2001,
    )

    plot_far_frr_curve(
        val_thresholds,
        val_fars,
        val_frrs,
        selected_threshold=threshold,
        output_path="val_far_frr_threshold_curve.png",
        title="Validation FAR/FRR vs Authentication Threshold",
    )

    save_far_frr_curve_csv(
        val_thresholds,
        val_fars,
        val_frrs,
        output_path="val_far_frr_threshold_curve.csv",
    )

    print(f"Validation EER threshold       : {val_eer_threshold:.4f}")
    print(f"Validation EER                 : {val_eer:.4f}")

    test_genuine, test_impostor = make_auth_scores(
        val_emb,
        val_users,
        centroids,
        max_impostors_per_sample=args.max_impostors_per_sample,
        seed=args.seed + 1,
    )

    test_far, test_frr = auth_metrics_at_threshold(
        test_genuine,
        test_impostor,
        threshold,
    )

    print("\n========== Final test results ==========")
    print(f"Digit recognition accuracy     : {val_metrics['digit_acc']:.4f}")
    print(f"User identification accuracy   : {val_metrics['user_identification_acc']:.4f}")

    print(f"Digit recognition accuracy     : {test_metrics['digit_acc']:.4f}")
    print(f"User identification accuracy   : {test_metrics['user_identification_acc']:.4f}")
    print(f"Auth threshold from val EER    : {threshold:.4f}")
    print(f"Validation FAR                 : {val_far:.4f}")
    print(f"Validation FRR/FFR             : {val_frr:.4f}")
    print(f"Test FAR                       : {test_far:.4f}")
    print(f"Test FRR/FFR                   : {test_frr:.4f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "user_to_idx": user_to_idx,
            "seq_len": args.seq_len,
            "auth_threshold": threshold,
        },
        "best_1dcnn_digit_auth.pt",
    )

    print("Saved model to best_1dcnn_digit_auth.pt")


fig_width = 18
fig_height = 15

font_size = 36

def plot_training_curves(history):
    epochs = np.arange(1, len(history["val_digit_acc"]) + 1)

    linewidth = 4

    plt.figure(figsize=(fig_width, fig_height))
    plt.rcParams["font.family"] = "Arial"
    # val_digit_acc val_user_id_acc loss
    plt.plot(epochs, history["val_digit_acc"], label="Digit Validation Accuracy", linewidth=linewidth)
    plt.plot(epochs, history["val_user_id_acc"], label="User Validation Accuracy", linewidth=linewidth)

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
    plt.savefig('../result/1d-cnn/Train Accuracy.pdf', format='pdf', bbox_inches='tight', pad_inches=0.26)

    # plt.figure(figsize=(7, 5))
    plt.figure(figsize=(fig_width, fig_height))
    plt.rcParams["font.family"] = "Arial"
    plt.plot(epochs, history["loss"], label="Train Loss", linewidth=linewidth)
    # 获取当前坐标轴
    ax = plt.gca()
    # 设置边框宽度
    ax.spines['top'].set_linewidth(5)  # 设置顶部边框宽度
    ax.spines['bottom'].set_linewidth(5)  # 设置底部边框宽度
    ax.spines['left'].set_linewidth(5)  # 设置左侧边框宽度
    ax.spines['right'].set_linewidth(5)  # 设置右侧边框宽度
    plt.tick_params(axis='both', labelsize=32, length=15, width=4, direction='in')  # 'both' 表示同时设置 x 轴和 y 轴的标尺字体大小

    plt.xlabel("Epoch", fontsize=font_size)
    plt.ylabel("Loss", fontsize=font_size)
    # plt.title("Digit Recognition Hinge Loss", fontsize=36)
    plt.legend(fontsize=font_size)
    # plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig('../result/1d-cnn/Recognition Loss.pdf', format='pdf', bbox_inches='tight', pad_inches=0.26)

def compute_far_frr_curve(genuine_scores, impostor_scores, num_thresholds=2001):
    """
    计算不同 threshold 下的 FAR 和 FRR 曲线。

    threshold 越低：
        系统越容易接受用户，FAR 通常升高，FRR 通常降低。

    threshold 越高：
        系统越严格，FAR 通常降低，FRR 通常升高。
    """
    thresholds = np.linspace(-1.0, 1.0, num_thresholds, dtype=np.float32)

    fars = []
    frrs = []

    for th in thresholds:
        far, frr = auth_metrics_at_threshold(
            genuine_scores,
            impostor_scores,
            th,
        )
        fars.append(far)
        frrs.append(frr)

    fars = np.array(fars)
    frrs = np.array(frrs)

    eer_idx = int(np.argmin(np.abs(fars - frrs)))
    eer_threshold = float(thresholds[eer_idx])
    eer = float((fars[eer_idx] + frrs[eer_idx]) / 2.0)

    return thresholds, fars, frrs, eer_threshold, eer


def plot_far_frr_curve(
    thresholds,
    fars,
    frrs,
    selected_threshold,
    output_path="far_frr_threshold_curve.png",
    title="FAR/FRR vs Authentication Threshold",
):
    """
    绘制 FAR 和 FRR 随认证阈值变化的曲线。
    """
    plt.figure(figsize=(fig_width, fig_height))

    plt.plot(thresholds, fars, label="FAR", linewidth=4)
    plt.plot(thresholds, frrs, label="FRR", linewidth=4)
    ax = plt.gca()
    # 设置边框宽度
    ax.spines['top'].set_linewidth(5)  # 设置顶部边框宽度
    ax.spines['bottom'].set_linewidth(5)  # 设置底部边框宽度
    ax.spines['left'].set_linewidth(5)  # 设置左侧边框宽度
    ax.spines['right'].set_linewidth(5)  # 设置右侧边框宽度
    plt.tick_params(axis='both', labelsize=32, length=font_size, width=4, direction='in')  # 'both' 表示同时设置 x 轴和 y 轴的标尺字体大小

    selected_idx = int(np.argmin(np.abs(thresholds - selected_threshold)))
    selected_far = fars[selected_idx]
    selected_frr = frrs[selected_idx]

    plt.axvline(
        selected_threshold,
        linestyle="--",
        label=f"Selected threshold = {selected_threshold:.3f}",
    )

    plt.scatter([selected_threshold], [selected_far])
    plt.scatter([selected_threshold], [selected_frr])

    plt.xlabel("Authentication threshold", fontsize=font_size)
    plt.ylabel("Error rate", fontsize=font_size)
    # plt.title(title)
    plt.legend(fontsize=font_size)

    # plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig('../result/1d-cnn/FAR-FRR.pdf', format='pdf', bbox_inches='tight', pad_inches=0.26)

    # plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved FAR/FRR threshold curve to {output_path}")


def save_far_frr_curve_csv(
    thresholds,
    fars,
    frrs,
    output_path="far_frr_threshold_curve.csv",
):
    """
    保存 FAR/FRR 曲线数据，方便后续用 Origin 或 Prism 重新作图。
    """
    df = pd.DataFrame({
        "threshold": thresholds,
        "FAR": fars,
        "FRR": frrs,
    })
    df.to_csv(output_path, index=False)
    print(f"Saved FAR/FRR curve data to {output_path}")

if __name__ == "__main__":
    main()