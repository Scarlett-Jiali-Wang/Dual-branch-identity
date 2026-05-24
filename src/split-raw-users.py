from pathlib import Path
import re
import pandas as pd

INPUT_FILE = Path("数字9.xlsx")
OUTPUT_DIR = Path("./split_user_excels")
SHEET_NAME = 0

def safe_filename(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    name = re.sub(r"\s+", "_", name)
    return name[:80] or "user"

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)

    if df.shape[1] < 11:
        raise ValueError(f"文件至少需要 11 列：第 1 列 + 10 个 user 列；当前只有 {df.shape[1]} 列。")

    first_col = df.columns[0]
    user_cols = list(df.columns[1:11])

    for i, user_col in enumerate(user_cols, start=1):
        out_df = df[[first_col, user_col]]
        filename = f"{i:02d}_{safe_filename(user_col)}.xlsx"
        output_path = OUTPUT_DIR / filename
        out_df.to_excel(output_path, index=False)
        print(f"已生成: {output_path}")

    print(f"完成：共生成 {len(user_cols)} 个 Excel 文件，输出目录：{OUTPUT_DIR}")

if __name__ == "__main__":
    main()