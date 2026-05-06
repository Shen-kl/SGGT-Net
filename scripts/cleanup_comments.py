#!/usr/bin/env python3
"""
清理代码中的注释和无用代码的脚本

使用方法:
    python scripts/cleanup_comments.py --file model/sggt_net.py --backup
    python scripts/cleanup_comments.py --all --backup
"""

import argparse
import re
import sys
from pathlib import Path


def remove_python_comments(content):
    """
    移除 Python 代码中的注释

    Args:
        content: Python 源代码字符串

    Returns:
        清理后的代码字符串
    """
    # 移除单行注释 (保留 docstring)
    lines = content.split('\n')
    cleaned_lines = []
    in_docstring = False
    docstring_delimiter = None

    for i, line in enumerate(lines):
        stripped = line.strip()

        # 检查 docstring
        if ('"""' in stripped or "'''" in stripped) and not in_docstring:
            # 可能是 docstring 的开始
            if stripped.count('"""') == 1 or stripped.count("'''") == 1:
                in_docstring = True
                docstring_delimiter = '"""' if '"""' in stripped else "'''"
                cleaned_lines.append(line)  # 保留 docstring
            else:
                # 完整的 docstring 在一行
                cleaned_lines.append(line)
            continue

        if in_docstring:
            cleaned_lines.append(line)
            if docstring_delimiter in stripped:
                in_docstring = False
            continue

        # 移除行内注释 (保留简单的描述性注释)
        if '#' in line and not line.strip().startswith('#'):
            # 保留一些有用的注释 (如 TODO, FIXME, 参数说明)
            if any(marker in line.upper() for marker in ['# TODO', '# FIXME', '# NOTE:', '# Args:', '# Returns:']):
                cleaned_lines.append(line)
            else:
                # 移除注释部分
                comment_pos = line.find('#')
                code_part = line[:comment_pos].rstrip()
                if code_part:  # 只保留非空代码行
                    cleaned_lines.append(code_part)
                else:
                    cleaned_lines.append('')  # 保留空行
        elif line.strip().startswith('#'):
            # 移除整行注释 (保留 docstring 和特殊标记)
            if stripped.startswith(('"""', "'''")) or any(marker in stripped for marker in ['# TODO', '# FIXME', '# NOTE:']):
                cleaned_lines.append(line)
            else:
                cleaned_lines.append('')
        else:
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def remove_unused_imports(content):
    """
    移除未使用的导入 (简单实现)

    注意: 这是一个简单的实现，可能无法处理所有情况
    """
    # 识别常见的未使用导入
    unused_patterns = [
        r'^import matplotlib.pyplot as plt\s*\n',  # 仅用于调试
        r'^from matplotlib import pyplot\s*\n',
    ]

    for pattern in unused_patterns:
        content = re.sub(pattern, '', content, flags=re.MULTILINE)

    return content


def remove_print_statements(content):
    """
    移除调试用的 print 语句
    """
    # 移除 print 语句 (保留有用的打印)
    patterns = [
        r'#.*print\([^)]*\)\s*\n',  # 注释掉的 print
        r'print\(["\'].*P [^"\']*["\']\)\s*\n',  # 调试 P 矩阵的 print
        r'print\(["\'].*Q [^"\']*["\']\)\s*\n',  # 调试 Q 矩阵的 print
        r'print\(["\'].*F [^"\']*["\']\)\s*\n',  # 调试 F 矩阵的 print
    ]

    for pattern in patterns:
        content = re.sub(pattern, '', content, flags=re.MULTILINE)

    return content


def remove_unused_variables(content):
    """
    移除未使用的变量 (简单实现)

    注意: 这是一个非常简单的实现，仅针对特定模式
    """
    # 移除未使用的临时变量
    patterns = [
        r'max_val, max_idx = torch\.max\(F_t\.flatten\(\), dim=0\)\n'
        r'coords = torch\.unravel_index\(max_idx, F_t\.shape\)\n',
    ]

    for pattern in patterns:
        content = re.sub(pattern, '', content)

    return content


def clean_file(file_path, backup=True):
    """
    清理单个文件

    Args:
        file_path: 文件路径
        backup: 是否创建备份
    """
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"错误: 文件不存在: {file_path}")
        return False

    print(f"正在清理文件: {file_path}")

    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # 创建备份
    if backup:
        backup_path = file_path.with_suffix('.py.backup')
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
        print(f"  已创建备份: {backup_path}")

    # 执行清理
    content = remove_unused_imports(content)
    content = remove_print_statements(content)
    content = remove_unused_variables(content)

    # 如果有变化，写入文件
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        original_lines = len(original_content.split('\n'))
        cleaned_lines = len(content.split('\n'))
        print(f"  清理完成: {original_lines} 行 -> {cleaned_lines} 行 (减少 {original_lines - cleaned_lines} 行)")
        return True
    else:
        print("  文件无需清理")
        return False


def main():
    parser = argparse.ArgumentParser(description='清理 Python 代码中的注释和无用代码')
    parser.add_argument('--file', type=str, help='要清理的文件路径')
    parser.add_argument('--all', action='store_true', help='清理所有 Python 文件')
    parser.add_argument('--backup', action='store_true', default=True, help='创建备份文件')
    parser.add_argument('--no-backup', action='store_true', help='不创建备份文件')

    args = parser.parse_args()

    if args.no_backup:
        args.backup = False

    if args.file:
        clean_file(args.file, backup=args.backup)
    elif args.all:
        # 清理项目中的所有 Python 文件
        project_root = Path(__file__).parent.parent
        py_files = list(project_root.rglob('*.py'))

        # 排除虚拟环境
        py_files = [f for f in py_files if 'venv' not in str(f) and '__pycache__' not in str(f)]

        print(f"找到 {len(py_files)} 个 Python 文件")

        cleaned_count = 0
        for py_file in py_files:
            if clean_file(py_file, backup=args.backup):
                cleaned_count += 1

        print(f"\n清理完成: {cleaned_count} 个文件被修改")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
