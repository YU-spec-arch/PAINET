import json
import pandas as pd

# 读取两个json文件
with open('E:/Computing_Power_Measurement/Intention/yolo_slowfast-master/deepsort_output.json', 'r') as f1:
    data1 = json.load(f1)

with open('E:/Computing_Power_Measurement/Intention/yolo_slowfast-master/other_output.json', 'r') as f2:
    data2 = json.load(f2)

# 将json数据转化为pandas DataFrame
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# 如果json文件中的数据是列表或数组形式，可以直接取前四列；如果是嵌套字典形式，则需要根据实际情况选择列名
df1 = df1.iloc[:, :4]  # 取前四列
df2 = df2.iloc[:, :4]  # 取前四列

# 检查并填充空值为0
df1.fillna(0, inplace=True)
df2.fillna(0, inplace=True)

# 查看处理后的数据
print(df1)
print(df2)

