#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json

filled_data = []
class ReadJson:
    def read_json(self,file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data

    def read_and_fill(self,json_file):
        filled_data = []
        for row in json_file:
            if len(row) >= 4:
                filled_data.append(row[:4])
            else:
                filled_data.append(row[:4] + [0] * (4 - len(row)))
        return filled_data

    def read2_and_fill2(self,json_file):

        # 遍历数据并处理
        for sublist in json_file:
            for data_list in sublist:
                if not data_list:  # 如果子列表为空
                    Readj = [0, 0, 0, 0]
                    filled_data.append(Readj)
                    #print([0, 0, 0, 0])
                else:
                    for data in data_list:
                        filled_data.append(data[:4])
                        #print(data[:4])
        #print(filled_data)
        return filled_data

