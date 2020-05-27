# to_categorical , index의 시작 항상 0부터임
# 0이 앞에 한 열이 생기는 문제가 발생
# 방법 1) 0을 슬라이스로 잘라내는 것
# 방법 2) one hoe encoder 쓰는 것

# 2번의 첫 번째 답

# x = [1, 2, 3]
# x = x - 1
# print(x) # numpy 쓰지 않고 그냥 했더니 TypeError 발생, list와 int 타입 지원하지 않음
         # numpy는 해준다 but, 문제점은 자료형을 한 가지밖에 못 쓴다
        

import numpy as np

y = np.array([1,2,3,4,5,1,2,3,4,5])

y = y - 1 # numpy라 가능한 방법 / 리스트 안, 전부 다 1을 빼서 강제적으로 0부터 시작하게 함
print(y)



# from keras.utils import np_utils # one-hot encoder랑 같은 건데 함수명이 다를 뿐이다 대신, y - 1 이나 슬라이싱으로 앞의 0을 처리해줘야함(원핫은 안 해줘도 됨)
# y = np_utils.to_categorical(y)
# print(y)
# print(y.shape)


# # 2번의 두번째 답
# y = np.array([1,2,3,4,5,1,2,3,4,5])
# print(y.shape) # (10, ) # one-hot encoder 는 2차원 형태로 넣어줘야 함
# # y = y.reshape(-1, 1) # -1? : 제일 끝, 
# y = y.reshape(10, 1) # -1과 10 같음
# # 2차원으로 변형!

# from sklearn.preprocessing import OneHotEncoder # one-hot encoder 싸이킷런에 있음
# aaa = OneHotEncoder()
# aaa.fit(y)
# y = aaa.transform(y).toarray()

# print(y)
# print(y.shape)

# 2번의 세번째 답
# 슬라이싱

# 마지막에 print(np.argmax(y_pred, axis = 1)+1) 이건 해야함(to_categorical이든 one-hot encoder든)



