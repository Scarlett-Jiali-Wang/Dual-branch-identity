"""
Zero-effort impostor attack and same-digit impostor attack evaluation
for the trained dual-branch SVM handwriting model.

Expected saved model files:
    ../saved_models/scaler.pkl
    ../saved_models/digit_svm.pkl
    ../saved_models/user_svm.pkl
    ../saved_models/config.pkl

Expected dataset format:
    dataset/user_0/digit_2/sample_001.csv
    dataset/user_1/digit_5/sample_xxx.csv

Run examples:
    python svm_attack_experiment.py --data_dir ../dataset --model_dir ../saved_models
    python svm_attack_experiment.py --data_dir ../dataset --model_dir ../saved_models --output_dir ../result/attack_eval
"""

import os
import re
import glob
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import skew, kurtosis
from sklearn.metrics import confusion_matrix, classification_report


# ============================================================
# 1. Data loading utilities, consistent with the training code
# ============================================================

def parse_user_digit_from_path(file_path):
    """
    Parse user_id and digit_id from path.

    Default path format:
        dataset/user_0/digit_2/sample_001.csv
    """
    path = str(file_path).replace("\\", "/")
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
        raise ValueError(f"Cannot parse user_id or digit_id from path: {file_path}")

    return user_id, digit_id


def load_one_csv(file_path):
    """
    Read one CSV file and keep only pressure-related numeric columns.
    Columns named time / t / timestamp are removed.

    Returns:
        seq: [T, C]
    """
    df = pd.read_csv(file_path)
    df_num = df.select_dtypes(include=[np.number]).copy()

    for col in list(df_num.columns):
        if col.lower() in ["time", "t", "timestamp"]:
            df_num = df_num.drop(columns=[col])

    if df_num.shape[1] == 0:
        raise ValueError(f"No valid numeric pressure columns in {file_path}")

    seq = df_num.values.astype(np.float32)
    seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
    return seq


def load_dataset(root_dir):
    """
    Load all CSV samples from the dataset.

    Returns:
        sequences: list of [T, C]
        y_digit: [N]
        y_user: [N]
        file_paths: list[str]
    """
    csv_files = sorted(glob.glob(os.path.join(root_dir, "**", "*.csv"), recursive=True))

    if len(csv_files) == 0:
        raise RuntimeError(f"No CSV files found under {root_dir}")

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

    return (
        sequences,
        np.array(y_digit, dtype=np.int64),
        np.array(y_user, dtype=np.int64),
        file_paths,
    )


# ============================================================
# 2. Resampling and feature extraction, consistent with training
# ============================================================

def resample_sequence(seq, target_len=512):
    """
    Resample variable-length pressure sequence to a fixed length.

    Input:
        seq: [T, C]
    Output:
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
    Extract statistical and derivative features from the resampled sequence.
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
    Convert one pressure sequence to a fixed-length SVM feature vector.
    """
    seq_resampled = resample_sequence(seq, target_len=target_len)
    stat_features = extract_stat_features(seq_resampled)

    if use_resampled_raw:
        raw_features = seq_resampled.flatten()
        feature = np.concatenate([raw_features, stat_features])
    else:
        feature = stat_features

    return feature.astype(np.float32)


def build_feature_matrix(sequences, target_len=512, use_resampled_raw=True):
    X = []
    for seq in sequences:
        X.append(sequence_to_feature(seq, target_len=target_len, use_resampled_raw=use_resampled_raw))
    return np.vstack(X).astype(np.float32)


# ============================================================
# 3. Model loading and scoring
# ============================================================

def load_models(model_dir):
    model_dir = Path(model_dir)

    scaler = joblib.load(model_dir / "scaler.pkl")
    digit_svm = joblib.load(model_dir / "digit_svm.pkl")
    user_svm = joblib.load(model_dir / "user_svm.pkl")

    config_path = model_dir / "config.pkl"
    if config_path.exists():
        config = joblib.load(config_path)
    else:
        config = {}

    return scaler, digit_svm, user_svm, config


def get_user_score_matrix(user_svm, X):
    scores = user_svm.decision_function(X)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    return scores


def get_class_index_map(classes):
    return {int(c): i for i, c in enumerate(classes)}


# ============================================================
# 4. Attack evaluation
# ============================================================

def evaluate_genuine_trials(score_matrix, y_user, user_classes, threshold):
    """
    Genuine trial:
        The claimed user equals the true user.

    Failure is FRR: genuine sample rejected by the system.

    Note:
        If the input dataset also contains external impostor users that were not
        enrolled in the trained user_svm, those users are skipped in genuine-trial
        evaluation because they have no corresponding SVM class. They are still
        included as impostors in the attack evaluations below.
    """
    user_to_index = get_class_index_map(user_classes)
    scores = []
    accepted = []
    skipped_external = 0

    for i, true_user in enumerate(y_user):
        true_user = int(true_user)
        if true_user not in user_to_index:
            skipped_external += 1
            continue

        idx = user_to_index[true_user]
        score = score_matrix[i, idx]
        scores.append(score)
        accepted.append(score >= threshold)

    scores = np.array(scores)
    accepted = np.array(accepted, dtype=bool)
    frr = float(np.mean(~accepted)) if len(accepted) > 0 else np.nan

    return {
        "scores": scores,
        "accepted": accepted,
        "FRR": frr,
        "GAR": 1.0 - frr if not np.isnan(frr) else np.nan,
        "num_trials": len(scores),
        "skipped_external_samples": skipped_external,
    }


def evaluate_zero_effort_attack(
    score_matrix,
    y_user,
    y_digit,
    digit_pred,
    user_classes,
    digit_classes,
    threshold,
    rng_seed=42,
):
    """
    Zero-effort impostor attack.

    Definition:
        The impostor does not know or imitate the target user's writing dynamics.
        For each sample, the actual writer claims to be every other user.

    Two attack success rates are reported:
        1. FAR_auth_only:
           The attack succeeds if user-authentication score >= threshold.

        2. FAR_challenge_response:
           The attack succeeds only if:
               user-authentication score >= threshold
               AND digit_pred == claimed_digit
           In zero-effort attack, claimed_digit is randomly selected from digit classes,
           representing a random challenge that the impostor did not intentionally match.
    """
    rng = np.random.default_rng(rng_seed)
    user_classes = np.array(user_classes, dtype=int)
    digit_classes = np.array(digit_classes, dtype=int)
    user_to_index = get_class_index_map(user_classes)

    rows = []
    auth_success = []
    challenge_success = []
    attack_scores = []
    target_users = []
    claimed_digits = []
    actual_digits = []
    pred_digits = []

    for i in range(len(y_user)):
        true_user = int(y_user[i])
        true_digit = int(y_digit[i])
        pred_digit = int(digit_pred[i])

        for claimed_user in user_classes:
            claimed_user = int(claimed_user)
            if claimed_user == true_user:
                continue

            claimed_digit = int(rng.choice(digit_classes))
            class_idx = user_to_index[claimed_user]
            score = score_matrix[i, class_idx]

            ok_auth = score >= threshold
            ok_challenge = ok_auth and (pred_digit == claimed_digit)

            auth_success.append(ok_auth)
            challenge_success.append(ok_challenge)
            attack_scores.append(score)
            target_users.append(claimed_user)
            claimed_digits.append(claimed_digit)
            actual_digits.append(true_digit)
            pred_digits.append(pred_digit)

            rows.append({
                "attack_type": "zero_effort",
                "sample_index": i,
                "actual_user": true_user,
                "claimed_user": claimed_user,
                "actual_digit": true_digit,
                "claimed_digit": claimed_digit,
                "pred_digit": pred_digit,
                "auth_score": float(score),
                "auth_success": bool(ok_auth),
                "challenge_response_success": bool(ok_challenge),
            })

    auth_success = np.array(auth_success, dtype=bool)
    challenge_success = np.array(challenge_success, dtype=bool)

    return {
        "rows": rows,
        "scores": np.array(attack_scores),
        "target_users": np.array(target_users),
        "claimed_digits": np.array(claimed_digits),
        "actual_digits": np.array(actual_digits),
        "pred_digits": np.array(pred_digits),
        "auth_success": auth_success,
        "challenge_success": challenge_success,
        "FAR_auth_only": float(np.mean(auth_success)),
        "FAR_challenge_response": float(np.mean(challenge_success)),
        "num_trials": int(len(auth_success)),
    }


def evaluate_same_digit_attack(
    score_matrix,
    y_user,
    y_digit,
    digit_pred,
    user_classes,
    threshold,
):
    """
    Same-digit impostor attack.

    Definition:
        The impostor writes the same requested digit as the claimed target user.
        Therefore, the claimed_digit is set to the actual digit of the impostor sample.

    Two attack success rates are reported:
        1. FAR_auth_only:
           The attack succeeds if user-authentication score >= threshold.

        2. FAR_challenge_response:
           The attack succeeds only if:
               user-authentication score >= threshold
               AND digit_pred == claimed_digit
           Here claimed_digit = actual_digit, because the impostor intentionally writes
           the required digit.
    """
    user_classes = np.array(user_classes, dtype=int)
    user_to_index = get_class_index_map(user_classes)

    rows = []
    auth_success = []
    challenge_success = []
    attack_scores = []
    target_users = []
    claimed_digits = []
    actual_digits = []
    pred_digits = []

    for i in range(len(y_user)):
        true_user = int(y_user[i])
        true_digit = int(y_digit[i])
        pred_digit = int(digit_pred[i])
        claimed_digit = true_digit

        for claimed_user in user_classes:
            claimed_user = int(claimed_user)
            if claimed_user == true_user:
                continue

            class_idx = user_to_index[claimed_user]
            score = score_matrix[i, class_idx]

            ok_auth = score >= threshold
            ok_challenge = ok_auth and (pred_digit == claimed_digit)

            auth_success.append(ok_auth)
            challenge_success.append(ok_challenge)
            attack_scores.append(score)
            target_users.append(claimed_user)
            claimed_digits.append(claimed_digit)
            actual_digits.append(true_digit)
            pred_digits.append(pred_digit)

            rows.append({
                "attack_type": "same_digit",
                "sample_index": i,
                "actual_user": true_user,
                "claimed_user": claimed_user,
                "actual_digit": true_digit,
                "claimed_digit": claimed_digit,
                "pred_digit": pred_digit,
                "auth_score": float(score),
                "auth_success": bool(ok_auth),
                "challenge_response_success": bool(ok_challenge),
            })

    auth_success = np.array(auth_success, dtype=bool)
    challenge_success = np.array(challenge_success, dtype=bool)

    return {
        "rows": rows,
        "scores": np.array(attack_scores),
        "target_users": np.array(target_users),
        "claimed_digits": np.array(claimed_digits),
        "actual_digits": np.array(actual_digits),
        "pred_digits": np.array(pred_digits),
        "auth_success": auth_success,
        "challenge_success": challenge_success,
        "FAR_auth_only": float(np.mean(auth_success)),
        "FAR_challenge_response": float(np.mean(challenge_success)),
        "num_trials": int(len(auth_success)),
    }


def summarize_by_group(values, groups, group_name, value_name):
    df = pd.DataFrame({group_name: groups, value_name: values.astype(float)})
    return df.groupby(group_name, as_index=False)[value_name].mean()


# ============================================================
# 5. Plotting
# ============================================================

def set_plot_style():
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["figure.dpi"] = 150


def save_bar_attack_summary(metrics_df, output_dir):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    x = np.arange(len(metrics_df))
    ax.bar(x, metrics_df["rate"].values)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df["metric"].values, rotation=25, ha="right")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, max(0.05, metrics_df["rate"].max() * 1.25))

    for i, value in enumerate(metrics_df["rate"].values):
        ax.text(i, value, f"{value * 100:.2f}%", ha="center", va="bottom", fontsize=9)

    ax.set_title("Authentication Error and Impostor Attack Success Rates")
    fig.tight_layout()
    fig.savefig(output_dir / "attack_summary_bar.png", bbox_inches="tight")
    fig.savefig(output_dir / "attack_summary_bar.svg", bbox_inches="tight")
    plt.close(fig)


def save_target_user_far_plot(zero_result, same_result, output_dir):
    zero_df = summarize_by_group(zero_result["challenge_success"], zero_result["target_users"], "claimed_user", "zero_effort_far")
    same_df = summarize_by_group(same_result["challenge_success"], same_result["target_users"], "claimed_user", "same_digit_far")

    merged = pd.merge(zero_df, same_df, on="claimed_user", how="outer").sort_values("claimed_user")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(merged))
    width = 0.38

    ax.bar(x - width / 2, merged["zero_effort_far"].values, width, label="Zero-effort")
    ax.bar(x + width / 2, merged["same_digit_far"].values, width, label="Same-digit")

    ax.set_xticks(x)
    ax.set_xticklabels([f"User {u}" for u in merged["claimed_user"].values])
    ax.set_ylabel("Challenge-response FAR")
    ax.set_xlabel("Claimed target user")
    ax.set_ylim(0, max(0.05, np.nanmax(merged[["zero_effort_far", "same_digit_far"]].values) * 1.25))
    ax.legend(frameon=False)
    ax.set_title("Attack Success Rate by Claimed Target User")

    fig.tight_layout()
    fig.savefig(output_dir / "far_by_target_user.png", bbox_inches="tight")
    fig.savefig(output_dir / "far_by_target_user.svg", bbox_inches="tight")
    plt.close(fig)

    merged.to_csv(output_dir / "far_by_target_user.csv", index=False, encoding="utf-8-sig")


def save_same_digit_far_by_digit_plot(same_result, output_dir):
    digit_df = summarize_by_group(
        same_result["challenge_success"],
        same_result["claimed_digits"],
        "claimed_digit",
        "same_digit_far",
    ).sort_values("claimed_digit")

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = np.arange(len(digit_df))
    ax.bar(x, digit_df["same_digit_far"].values)

    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in digit_df["claimed_digit"].values])
    ax.set_xlabel("Claimed digit")
    ax.set_ylabel("Challenge-response FAR")
    ax.set_ylim(0, max(0.05, digit_df["same_digit_far"].max() * 1.25))
    ax.set_title("Same-Digit Impostor Attack Success Rate by Digit")

    for i, value in enumerate(digit_df["same_digit_far"].values):
        ax.text(i, value, f"{value * 100:.2f}%", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "same_digit_far_by_digit.png", bbox_inches="tight")
    fig.savefig(output_dir / "same_digit_far_by_digit.svg", bbox_inches="tight")
    plt.close(fig)

    digit_df.to_csv(output_dir / "same_digit_far_by_digit.csv", index=False, encoding="utf-8-sig")


def save_score_distribution_plot(genuine_result, zero_result, same_result, threshold, output_dir):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    ax.hist(genuine_result["scores"], bins=50, alpha=0.45, density=True, label="Genuine")
    ax.hist(zero_result["scores"], bins=50, alpha=0.45, density=True, label="Zero-effort impostor")
    ax.hist(same_result["scores"], bins=50, alpha=0.45, density=True, label="Same-digit impostor")
    ax.axvline(threshold, linestyle="--", linewidth=2, label=f"Threshold = {threshold:.4f}")

    ax.set_xlabel("SVM decision score for claimed user")
    ax.set_ylabel("Density")
    ax.set_title("Genuine and Impostor Score Distributions")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_dir / "score_distribution.png", bbox_inches="tight")
    fig.savefig(output_dir / "score_distribution.svg", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 6. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate zero-effort and same-digit impostor attacks for trained SVM model")
    parser.add_argument("--data_dir", type=str, default="../dataset-tmp", help="Dataset root directory")
    parser.add_argument("--model_dir", type=str, default="../saved_models", help="Saved model directory")
    parser.add_argument("--output_dir", type=str, default="../result/attack_eval", help="Output directory for figures and tables")
    parser.add_argument("--threshold", type=float, default=None, help="Override authentication threshold; by default use config['eer_threshold']")
    parser.add_argument("--rng_seed", type=int, default=42, help="Random seed for zero-effort challenge digits")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    print("Loading models...")
    scaler, digit_svm, user_svm, config = load_models(args.model_dir)

    target_len = int(config.get("target_len", 512))
    use_resampled_raw = bool(config.get("use_resampled_raw", True))

    if args.threshold is not None:
        threshold = float(args.threshold)
    else:
        if "eer_threshold" not in config:
            raise ValueError("No threshold was provided and config.pkl does not contain 'eer_threshold'.")
        threshold = float(config["eer_threshold"])

    print(f"target_len = {target_len}")
    print(f"use_resampled_raw = {use_resampled_raw}")
    print(f"authentication threshold = {threshold:.6f}")

    print("\nLoading dataset...")
    sequences, y_digit, y_user, file_paths = load_dataset(args.data_dir)
    print(f"Total samples: {len(sequences)}")
    print(f"Digit classes in data: {np.unique(y_digit)}")
    print(f"User classes in data: {np.unique(y_user)}")

    print("\nBuilding feature matrix...")
    X = build_feature_matrix(sequences, target_len=target_len, use_resampled_raw=use_resampled_raw)

    if X.shape[1] != scaler.n_features_in_:
        raise ValueError(
            f"Feature dimension mismatch: current X has {X.shape[1]} features, "
            f"but scaler expects {scaler.n_features_in_}. "
            "Please check data channels, target_len, and use_resampled_raw."
        )

    X_scaled = scaler.transform(X)

    print("\nRunning model prediction...")
    digit_pred = digit_svm.predict(X_scaled)
    user_pred = user_svm.predict(X_scaled)
    score_matrix = get_user_score_matrix(user_svm, X_scaled)

    print("\n========== Digit Recognition Report ==========")
    print(classification_report(y_digit, digit_pred, digits=4, zero_division=0))

    print("\n========== User Identification Report ==========")
    print(classification_report(y_user, user_pred, digits=4, zero_division=0))

    pd.DataFrame(confusion_matrix(y_digit, digit_pred)).to_csv(output_dir / "digit_confusion_matrix.csv", index=False)
    pd.DataFrame(confusion_matrix(y_user, user_pred)).to_csv(output_dir / "user_confusion_matrix.csv", index=False)

    print("\nEvaluating genuine trials...")
    genuine_result = evaluate_genuine_trials(
        score_matrix=score_matrix,
        y_user=y_user,
        user_classes=user_svm.classes_,
        threshold=threshold,
    )

    print("Evaluating zero-effort impostor attack...")
    zero_result = evaluate_zero_effort_attack(
        score_matrix=score_matrix,
        y_user=y_user,
        y_digit=y_digit,
        digit_pred=digit_pred,
        user_classes=user_svm.classes_,
        digit_classes=digit_svm.classes_,
        threshold=threshold,
        rng_seed=args.rng_seed,
    )

    print("Evaluating same-digit impostor attack...")
    same_result = evaluate_same_digit_attack(
        score_matrix=score_matrix,
        y_user=y_user,
        y_digit=y_digit,
        digit_pred=digit_pred,
        user_classes=user_svm.classes_,
        threshold=threshold,
    )

    # Save trial-level records. For very large datasets, these files can be large.
    pd.DataFrame(zero_result["rows"]).to_csv(output_dir / "zero_effort_attack_trials.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(same_result["rows"]).to_csv(output_dir / "same_digit_attack_trials.csv", index=False, encoding="utf-8-sig")

    summary_rows = [
        {
            "metric": "Genuine FRR",
            "rate": genuine_result["FRR"],
            "num_trials": genuine_result["num_trials"],
        },
        {
            "metric": "Zero-effort FAR, auth only",
            "rate": zero_result["FAR_auth_only"],
            "num_trials": zero_result["num_trials"],
        },
        {
            "metric": "Zero-effort FAR, challenge-response",
            "rate": zero_result["FAR_challenge_response"],
            "num_trials": zero_result["num_trials"],
        },
        {
            "metric": "Same-digit FAR, auth only",
            "rate": same_result["FAR_auth_only"],
            "num_trials": same_result["num_trials"],
        },
        {
            "metric": "Same-digit FAR, challenge-response",
            "rate": same_result["FAR_challenge_response"],
            "num_trials": same_result["num_trials"],
        },
    ]

    metrics_df = pd.DataFrame(summary_rows)
    metrics_df.to_csv(output_dir / "attack_summary_metrics.csv", index=False, encoding="utf-8-sig")

    print("\n========== Attack Summary ==========")
    for _, row in metrics_df.iterrows():
        print(f"{row['metric']}: {row['rate'] * 100:.4f}%  ({int(row['num_trials'])} trials)")

    print("\nSaving plots...")
    save_bar_attack_summary(metrics_df, output_dir)
    save_target_user_far_plot(zero_result, same_result, output_dir)
    save_same_digit_far_by_digit_plot(same_result, output_dir)
    save_score_distribution_plot(genuine_result, zero_result, same_result, threshold, output_dir)

    print(f"\nDone. Results saved to: {output_dir.resolve()}")
    print("Generated files:")
    print("  attack_summary_metrics.csv")
    print("  zero_effort_attack_trials.csv")
    print("  same_digit_attack_trials.csv")
    print("  attack_summary_bar.png / .svg")
    print("  far_by_target_user.png / .svg")
    print("  same_digit_far_by_digit.png / .svg")
    print("  score_distribution.png / .svg")


if __name__ == "__main__":
    main()
