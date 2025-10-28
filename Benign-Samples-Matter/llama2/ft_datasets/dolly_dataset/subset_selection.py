import random
import json

def select_random_subset(input_file, output_file, n):
    # 打开原始数据文件读取内容
    with open(input_file, 'r') as infile:
        data = [json.loads(line) for line in infile]

    # 随机选择 n 条数据
    subset = random.sample(data, n)

    # 将选取的数据写入新的文件
    with open(output_file, 'w') as outfile:
        for item in subset:
            json.dump(item, outfile)
            outfile.write('\n')

n = 1000  # 选择的条目数量
# 使用示例
input_file = '/root/code/Benign_Finetuning_Attack/Benign-Samples-Matter/llama2/ft_datasets/dolly_dataset/databricks-dolly-15k-no-safety.jsonl'
output_file = '/root/code/Benign_Finetuning_Attack/Benign-Samples-Matter/llama2/ft_datasets/dolly_dataset/dolly_subset_{}.jsonl'.format(n)

select_random_subset(input_file, output_file, n)
