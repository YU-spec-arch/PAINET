# -*- coding: utf-8 -*-

import json

def read_json_file(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            row = json.loads(line)
            # ֻ����ǰ�������ݣ����ĳ�в������У��������
            row = row[:4] + [0] * (4 - len(row))
            data.append(row)
    return data

# ��ȡ��һ��JSON�ļ�
file1_data = read_json_file('E:\\Computing_Power_Measurement\\Intention\\yolo_slowfast-master\\deepsort_output.json')

# ��ȡ�ڶ���JSON�ļ�
file2_data = read_json_file('E:\\Computing_Power_Measurement\\Intention\\yolo_slowfast-master\\deepsort_output.json')

print("���������ļ�1:")
print(file1_data)
print("\n���������ļ�2:")
print(file2_data)
