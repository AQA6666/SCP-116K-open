from utils import *
import argparse


def filter_matched_solutions(input_file, output_file):
    """过滤出包含matched_solution的数据
    
    Args:
        input_file: 输入文件路径（合并后的数据）
        output_file: 输出文件路径（只包含有匹配解决方案的数据）
    """
    print(f"开始读取数据: {input_file}")
    
    # 读取合并后的数据
    try:
        all_data = load_jsonl(input_file)
        print(f"成功读取 {len(all_data)} 条数据")
    except Exception as e:
        print(f"读取失败: {e}")
        return
    
    # 过滤出包含matched_solution的数据
    filtered_data = []
    for item in all_data:
        if 'matched_solution' in item and item['matched_solution'] is not None:
            filtered_data.append(item)
    
    # 统计信息
    print("\n" + "="*50)
    print(f"过滤完成!")
    print(f"原始数据条数: {len(all_data)}")
    print(f"有匹配解决方案的数据: {len(filtered_data)}")
    print(f"无匹配解决方案的数据: {len(all_data) - len(filtered_data)}")
    print(f"匹配率: {len(filtered_data)/len(all_data)*100:.2f}%")
    print("="*50)
    
    # 保存过滤后的数据
    if filtered_data:
        write_jsonl(output_file, filtered_data)
        print(f"\n已保存到: {output_file}")
    else:
        print("\n警告: 没有找到包含matched_solution的数据!")
    
    return filtered_data


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='过滤出包含matched_solution的问题数据')
    
    # 输入文件（合并后的数据）
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='/gemini/space/guarded_files/ludakuan/data/extracted_problems/all_problems_matched_solutions_merged.jsonl',
        help='输入文件路径（合并后的数据）'
    )
    
    # 输出文件（过滤后的数据）
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='/gemini/space/guarded_files/ludakuan/data/extracted_problems/all_problems_had_matched_solutions.jsonl',
        help='输出文件路径（只包含有匹配解决方案的数据）'
    )
    
    args = parser.parse_args()
    
    # 打印参数信息
    print("="*50)
    print("过滤参数:")
    for key, value in args.__dict__.items():
        print(f"  {key}: {value}")
    print("="*50 + "\n")
    
    # 执行过滤
    filter_matched_solutions(args.input_file, args.output_file)


if __name__ == '__main__':
    main()
