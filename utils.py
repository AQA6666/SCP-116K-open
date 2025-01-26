import json
import os
import fnmatch


def load_jsonl(file_path):
    with open(file_path) as f:
        data = [json.loads(line) for line in f.readlines()]
    return data


def write_jsonl(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')


def find_files(directory, pattern):
    # 列表用于存储找到的文件路径
    file_paths = []

    # os.walk遍历目录和子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 使用通配符表达式进行匹配
            if fnmatch.fnmatch(file, pattern):
                # 将符合条件的文件的完整路径添加到列表中
                file_paths.append(os.path.join(root, file))

    return file_paths
