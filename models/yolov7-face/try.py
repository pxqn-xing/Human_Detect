import os
import numpy as np


def validate_labels(label_dir):
    error_log = []
    for label_file in os.listdir(label_dir):
        if not label_file.endswith('.txt'):
            continue

        with open(os.path.join(label_dir, label_file), 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if not lines:
                error_log.append(f"空文件: {label_file}")
                continue

            for line_num, line in enumerate(lines, 1):
                parts = line.strip().split()
                if len(parts) != 5:
                    error_log.append(f"{label_file}:{line_num} 列数错误 -> {line}")
                    continue

                try:
                    cls, x, y, w, h = map(float, parts)
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        error_log.append(f"{label_file}:{line_num} 坐标越界 -> {x},{y},{w},{h}")
                    if int(cls) != 0:
                        error_log.append(f"{label_file}:{line_num} 类别ID错误 -> {cls}")
                except ValueError:
                    error_log.append(f"{label_file}:{line_num} 数值格式错误 -> {line}")

    print(f"发现 {len(error_log)} 个错误:")
    print('\n'.join(error_log[:5]))  # 仅显示前5个错误


validate_labels('data/widerface/train/labels')
