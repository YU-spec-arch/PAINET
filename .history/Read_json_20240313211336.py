import json
import pandas as pd

# ��ȡ����json�ļ�
with open('E:/Computing_Power_Measurement/Intention/yolo_slowfast-master/deepsort_output.json', 'r') as f1:
    data1 = json.load(f1)

with open('E:/Computing_Power_Measurement/Intention/yolo_slowfast-master/other_output.json', 'r') as f2:
    data2 = json.load(f2)

# ��json����ת��Ϊpandas DataFrame
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# ���json�ļ��е��������б��������ʽ������ֱ��ȡǰ���У������Ƕ���ֵ���ʽ������Ҫ����ʵ�����ѡ������
df1 = df1.iloc[:, :4]  # ȡǰ����
df2 = df2.iloc[:, :4]  # ȡǰ����

# ��鲢����ֵΪ0
df1.fillna(0, inplace=True)
df2.fillna(0, inplace=True)

# �鿴����������
print(df1)
print(df2)

