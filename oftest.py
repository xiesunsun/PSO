import scipy.io
import numpy as np
# 读取 .mat 文件
mat_data = scipy.io.loadmat('Adenoma.mat')

# 查看文件中的变量名
print(mat_data.keys())

# 假设文件中有一个名为 'data' 的变量
data = mat_data['Adenoma']

# 查看数据的类型和内容
# print(type(data))
print(len(data[:,-1]))