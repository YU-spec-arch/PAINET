# -*- coding: utf-8 -*-

import json

def read_json_file(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            row = json.loads(line)
            # 只保留前四列数据，如果某行不足四列，用零填充
            row = row[:4] + [0] * (4 - len(row))
            data.append(row)
    return data

# 读取第一个JSON文件
file1_data = read_json_file('E:\\Computing_Power_Measurement\\Intention\\yolo_slowfast-master\\deepsort_output.json')

# 读取第二个JSON文件
file2_data = read_json_file('E:\\Computing_Power_Measurement\\Intention\\yolo_slowfast-master\\deepsort_output.json')

print("数据来自文件1:")
print(file1_data)
print("\n数据来自文件2:")
print(file2_data)
