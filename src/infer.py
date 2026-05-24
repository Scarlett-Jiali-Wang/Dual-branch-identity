import os
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import skew, kurtosis


# ============================================================
# 1. 读取单个 CSV
# ============================================================

def load_one_csv(file_path):
    """
    读取单个 csv 文件。

    要求：
        csv 中至少包含一个数值型压力数据列。
        如果存在 time / t / timestamp 列，会自动删除。

    返回：
        seq: shape = [T, C]
    """
    df = pd.read_csv(file_path)

    df_num = df.select_dtypes(include=[np.number]).copy()

    for col in list(df_num.columns):
        if col.lower() in ["time", "t", "timestamp"]:
            df_num = df_num.drop(columns=[col])

    if df_num.shape[1] == 0:
        raise ValueError(f"{file_path} 中没有有效压力数据列")

    seq = df_num.values.astype(np.float32)
    seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)

    return seq


# ============================================================
# 2. 重采样与特征提取
# ============================================================

def resample_sequence(seq, target_len=512):
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
    """
    features = []

    features.extend(np.mean(seq, axis=0))
    features.extend(np.std(seq, axis=0))
    features.extend(np.min(seq, axis=0))
    features.extend(np.max(seq, axis=0))
    features.extend(np.median(seq, axis=0))
    features.extend(np.percentile(seq, 25, axis=0))
    features.extend(np.percentile(seq, 75, axis=0))
    features.extend(np.ptp(seq, axis=0))

    features.extend(np.sum(seq ** 2, axis=0))
    features.extend(np.sqrt(np.mean(seq ** 2, axis=0)))

    features.extend(skew(seq, axis=0, nan_policy="omit"))
    features.extend(kurtosis(seq, axis=0, nan_policy="omit"))

    diff1 = np.diff(seq, axis=0)

    features.extend(np.mean(diff1, axis=0))
    features.extend(np.std(diff1, axis=0))
    features.extend(np.max(np.abs(diff1), axis=0))
    features.extend(np.sum(np.abs(diff1), axis=0))

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


def sequence_to_feature(seq, target_len=512, use_resampled_raw=True):
    """
    将一条压力时序转换为模型输入特征。
    必须和训练阶段保持一致。
    """
    seq_resampled = resample_sequence(seq, target_len=target_len)
    stat_features = extract_stat_features(seq_resampled)

    if use_resampled_raw:
        raw_features = seq_resampled.flatten()
        feature = np.concatenate([raw_features, stat_features])
    else:
        feature = stat_features

    return feature.astype(np.float32)


# ============================================================
# 3. 加载模型
# ============================================================

def load_models(model_dir):
    model_dir = Path(model_dir)

    scaler_path = model_dir / "scaler.pkl"
    digit_svm_path = model_dir / "digit_svm.pkl"
    user_svm_path = model_dir / "user_svm.pkl"
    config_path = model_dir / "config.pkl"

    if not scaler_path.exists():
        raise FileNotFoundError(f"找不到 {scaler_path}")
    if not digit_svm_path.exists():
        raise FileNotFoundError(f"找不到 {digit_svm_path}")
    if not user_svm_path.exists():
        raise FileNotFoundError(f"找不到 {user_svm_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"找不到 {config_path}")

    scaler = joblib.load(scaler_path)
    digit_svm = joblib.load(digit_svm_path)
    user_svm = joblib.load(user_svm_path)
    config = joblib.load(config_path)

    return scaler, digit_svm, user_svm, config


# ============================================================
# 4. 用户认证得分
# ============================================================

def get_user_score_matrix(user_svm, X):
    scores = user_svm.decision_function(X)

    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)

    return scores


def authenticate(user_svm, X_scaled, claimed_user, threshold):
    """
    对单个样本进行用户认证。

    输入：
        claimed_user: 声称用户编号，例如 0、1、2...
        threshold: 训练时根据 EER 得到的阈值

    输出：
        accept: 是否通过认证
        score: claimed_user 对应的 SVM 分数
    """
    user_classes = user_svm.classes_

    if claimed_user not in user_classes:
        raise ValueError(
            f"claimed_user={claimed_user} 不在模型已训练用户类别中，"
            f"当前模型用户类别为: {user_classes}"
        )

    score_matrix = get_user_score_matrix(user_svm, X_scaled)
    class_index = np.where(user_classes == claimed_user)[0][0]

    score = score_matrix[0, class_index]
    accept = score >= threshold

    return accept, score


# ============================================================
# 5. 单个文件预测
# ============================================================

def predict_one_csv(csv_path, model_dir="../saved_models", claimed_user=None):
    scaler, digit_svm, user_svm, config = load_models(model_dir)

    target_len = config.get("target_len", 512)
    use_resampled_raw = config.get("use_resampled_raw", True)
    threshold = config.get("eer_threshold", None)

    seq = load_one_csv(csv_path)

    feature = sequence_to_feature(
        seq,
        target_len=target_len,
        use_resampled_raw=use_resampled_raw
    )

    X = feature.reshape(1, -1)

    if X.shape[1] != scaler.n_features_in_:
        raise ValueError(
            f"特征维度不一致：当前样本特征维度为 {X.shape[1]}，"
            f"但模型需要 {scaler.n_features_in_}。"
            "请检查 CSV 通道数、target_len、use_resampled_raw 是否与训练时一致。"
        )

    X_scaled = scaler.transform(X)

    digit_pred = digit_svm.predict(X_scaled)[0]
    user_pred = user_svm.predict(X_scaled)[0]

    result = {
        "csv_path": str(csv_path),
        "pred_digit": int(digit_pred),
        "pred_user": int(user_pred)
    }

    if claimed_user is not None:
        if threshold is None:
            raise ValueError("config.pkl 中没有 eer_threshold，无法进行认证判断")

        claimed_user = int(claimed_user)

        accept, score = authenticate(
            user_svm=user_svm,
            X_scaled=X_scaled,
            claimed_user=claimed_user,
            threshold=threshold
        )

        result["claimed_user"] = claimed_user
        result["auth_score"] = float(score)
        result["auth_threshold"] = float(threshold)
        result["auth_accept"] = bool(accept)

    return result


# ============================================================
# 6. 批量预测文件夹
# ============================================================

def predict_folder(folder_path, model_dir="../saved_models", output_csv="prediction_results.csv"):
    folder_path = Path(folder_path)

    csv_files = list(folder_path.rglob("*.csv"))

    if len(csv_files) == 0:
        raise RuntimeError(f"{folder_path} 下没有找到 csv 文件")

    results = []

    for csv_path in csv_files:
        try:
            result = predict_one_csv(
                csv_path=csv_path,
                model_dir=model_dir,
                claimed_user=None
            )
            results.append(result)
            print(f"[OK] {csv_path} -> digit={result['pred_digit']}, user={result['pred_user']}")

        except Exception as e:
            print(f"[ERROR] {csv_path}: {e}")
            results.append({
                "csv_path": str(csv_path),
                "pred_digit": None,
                "pred_user": None,
                "error": str(e)
            })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"\n批量预测完成，结果已保存到: {output_csv}")


# ============================================================
# 7. 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Run trained dual-branch SVM model")

    parser.add_argument(
        "--input",
        type=str,
        # required=True,
        default="../dataset-tmp/user_1/digit_2/sample_001.csv",
        help="输入单个 csv 文件路径，或者包含多个 csv 的文件夹路径"
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default="../saved_models",
        help="模型文件夹路径，默认 ../saved_models"
    )

    parser.add_argument(
        "--claimed_user",
        type=int,
        default=None,
        help="如果需要用户认证，输入声称用户编号，例如 0、1、2"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="prediction_results.csv",
        help="批量预测时保存结果的 csv 文件名"
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file():
        result = predict_one_csv(
            csv_path=input_path,
            model_dir=args.model_dir,
            claimed_user=args.claimed_user
        )

        print("\n========== Prediction Result ==========")
        print(f"File: {result['csv_path']}")
        print(f"Predicted digit: {result['pred_digit']}")
        print(f"Predicted user: {result['pred_user']}")

        if args.claimed_user is not None:
            print("\n========== Authentication Result ==========")
            print(f"Claimed user: {result['claimed_user']}")
            print(f"Authentication score: {result['auth_score']:.6f}")
            print(f"Threshold: {result['auth_threshold']:.6f}")
            print(f"Accepted: {result['auth_accept']}")

    elif input_path.is_dir():
        predict_folder(
            folder_path=input_path,
            model_dir=args.model_dir,
            output_csv=args.output
        )

    else:
        raise FileNotFoundError(f"输入路径不存在: {input_path}")


if __name__ == "__main__":
    main()