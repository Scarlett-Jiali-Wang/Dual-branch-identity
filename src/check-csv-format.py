from pathlib import Path
import pandas as pd

DATASET_DIR = Path("../dataset")
EXPECTED_FILENAME = "sample_001.csv"
EXPECTED_COLUMNS = ["time", "pressure"]


def is_missing(value) -> bool:
    """
    判断单元格是否为空。
    同时处理 NaN、None、空字符串、全空格字符串。
    """
    if pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def check_csv_content(file_path: Path) -> list[str]:
    """
    检查单个 CSV 文件内容。
    返回错误信息列表。
    """
    errors = []

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return [f"无法读取 CSV 文件: {e}"]

    # 检查列数
    if df.shape[1] != 2:
        errors.append(f"列数错误：应为 2 列，实际为 {df.shape[1]} 列")
        return errors

    # 检查列名
    actual_columns = list(df.columns)
    if actual_columns != EXPECTED_COLUMNS:
        errors.append(
            f"列标题错误：应为 {EXPECTED_COLUMNS}，实际为 {actual_columns}"
        )
        return errors

    # 检查每行两列是否成对有值
    for idx, row in df.iterrows():
        time_missing = is_missing(row["time"])
        pressure_missing = is_missing(row["pressure"])

        if time_missing != pressure_missing:
            csv_line_number = idx + 2  # +2 是因为第 1 行是表头
            errors.append(
                f"第 {csv_line_number} 行数据不成对："
                f"time={'空' if time_missing else '有值'}，"
                f"pressure={'空' if pressure_missing else '有值'}"
            )

    return errors

def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"目录不存在: {DATASET_DIR}")

    all_errors = []

    for file_path in DATASET_DIR.rglob("*"):
        if not file_path.is_file():
            continue

        relative_path = file_path.relative_to(DATASET_DIR)

        # 检查文件名
        if file_path.name != EXPECTED_FILENAME:
            all_errors.append(
                f"[文件名错误] {relative_path}：文件名应为 {EXPECTED_FILENAME}"
            )

        # 只检查 csv 文件内容
        if file_path.suffix.lower() == ".csv":
            content_errors = check_csv_content(file_path)
            for error in content_errors:
                all_errors.append(f"[内容错误] {relative_path}：{error}")
        else:
            all_errors.append(
                f"[文件类型错误] {relative_path}：不是 CSV 文件"
            )

    if all_errors:
        print(f"检查完成，发现 {len(all_errors)} 个问题：")
        for error in all_errors:
            print(error)
    else:
        print("检查通过：所有文件名和 CSV 内容均符合要求。")

def check_file_name():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"目录不存在: {DATASET_DIR}")

    bad_files = []

    for file_path in DATASET_DIR.rglob("*"):
        if file_path.is_file() and file_path.name != EXPECTED_FILENAME:
            relative_path = file_path.relative_to(DATASET_DIR)
            bad_files.append(relative_path)

    if bad_files:
        print(f"发现 {len(bad_files)} 个文件名不符合要求：")
        for path in bad_files:
            print(path)
    else:
        print("所有文件名均为 sample_001.csv")

if __name__ == "__main__":
    # check_file_name()
    main()