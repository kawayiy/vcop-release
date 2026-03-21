import os
import zipfile

# ====== 配置 ======
SRC_DIR = "."  # 当前目录
OUTPUT_ZIP = "vcop_release.zip"

# 不打包的目录
EXCLUDE_DIRS = {
    "data",
    "logs",
    "debug_runs",
    "__pycache__",
    ".idea",
    ".git"
}

# 不打包的文件后缀
EXCLUDE_EXT = {
    ".pt",
    ".pth",
    ".tar",
    ".zip",
    ".mp4",
    ".avi"
}

# ==================

def should_exclude(path):
    parts = path.split(os.sep)

    # 排除目录
    for p in parts:
        if p in EXCLUDE_DIRS:
            return True

    # 排除文件类型
    _, ext = os.path.splitext(path)
    if ext in EXCLUDE_EXT:
        return True

    return False


def zip_project(src_dir, output_zip):
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(src_dir):

            # 过滤目录
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

            for file in files:
                full_path = os.path.join(root, file)

                if should_exclude(full_path):
                    continue

                # 相对路径
                arcname = os.path.relpath(full_path, src_dir)

                zipf.write(full_path, arcname)
                print(f"Added: {arcname}")

    print(f"\n✅ Done! Zip saved to: {output_zip}")


if __name__ == "__main__":
    zip_project(SRC_DIR, OUTPUT_ZIP)