# DB에 있는 것 numpy 로 만들기

import pymssql as ms
import numpy as np

conn = ms.connect(server='127.0.0.1', user='bit2', password='1234', database='bitdb')

cursor = conn.cursor()

cursor.execute('SELECT * FROM iris2;')

row = cursor.fetchall()
print(row)
conn.close()

print("=====================================================================================================")
aaa = np.asarray(row)
print(aaa) # 깔끔하게 정리가 됨
print("aaa.shape :", aaa.shape) # aaa.shape : (150, 5)
print(type(aaa)) # <class 'numpy.ndarray'>

# numpy 로 저장
np.save('./data/npy/test_flask_iris2.npy', aaa)