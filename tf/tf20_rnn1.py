import tensorflow as tf
import numpy as np

# data : hihello
# 인덱스를 넣어주기 위해 한 글자씩 뺐다(알파벳순)
idex2char = ['e', 'h', 'i', 'l', 'o']

# _data = np.array([['h'], ['i'], ['h'], ['e'], ['l'], ['l'], ['o']])
_data = np.array([['h', 'i', 'h', 'e', 'l', 'l', 'o']], dtype = np.str).reshape(-1, 1)

print("_data.shape :", _data.shape)
print(_data)
print(type(_data))
# _data.shape : (7, 1)
# [['h' 'i' 'h' 'e' 'l' 'l' 'o']]
# <class 'numpy.ndarray'>

print("--------------------------------------------------")

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(_data)
_data = enc.transform(_data).toarray()
print("_data.shape :", _data.shape)
print(_data)
print(type(_data))
print(_data.dtype)
print()

# _data.shape : (7, 5)
# [[0. 1. 0. 0. 0.]  h
#  [0. 0. 1. 0. 0.]  i
#  [0. 1. 0. 0. 0.]  h
#  [1. 0. 0. 0. 0.]  e
#  [0. 0. 0. 1. 0.]  l
#  [0. 0. 0. 1. 0.]  l
#  [0. 0. 0. 0. 1.]] o
# <class 'numpy.ndarray'>
# float64

# hihell 까지 x,  ihello 까지  y로 잡고 훈련을 시킨다

x_data = _data[:6, ] # hihell
y_data = _data[1:, ] #  ihello

print("x_data")
print(x_data)
print()
print("y_data")
print(y_data)

# x_data (6, 5) -> (1, 6, 5) (for LSTM) -> input shape = 6, 5 /(6, 5, 1) 이렇게 해도 LSTM 모델에 들어간다(나중에 시도해봐라)
# [[0. 1. 0. 0. 0.]  h
#  [0. 0. 1. 0. 0.]  i
#  [0. 1. 0. 0. 0.]  h
#  [1. 0. 0. 0. 0.]  e
#  [0. 0. 0. 1. 0.]  l
#  [0. 0. 0. 1. 0.]] l

# y_data (6, 5) y_data는 (1, 6)으로 준비해야 한다 -> index로 -> [[2, 1, 0, 3, 3, 4]]
# [[0. 0. 1. 0. 0.]  i  2(index자리)
#  [0. 1. 0. 0. 0.]  h  1
#  [1. 0. 0. 0. 0.]  e  0
#  [0. 0. 0. 1. 0.]  l  3
#  [0. 0. 0. 1. 0.]  l  3
#  [0. 0. 0. 0. 1.]] o  4

y_data = np.argmax(y_data, axis = 1)
print()
print("y_data(argmax)")
print(y_data)
print("y_data.shape :", y_data.shape)
# [2 1 0 3 3 4]
# y_data.shape : (6,)

# RESHAPE
x_data = x_data.reshape(1, 6, 5)
y_data = y_data.reshape(1, 6)

print("x_data.shape :", x_data.shape) # (1, 6, 5)
print("y_data.shape :", y_data.shape) # (1, 6)




