from keras.preprocessing.text import Tokenizer
import numpy as np

docs = ["너무 재밌어요", "참 최고예요", "참 잘 만든 영화예요", 
        "추천하고 싶은 영화입니다", "한 번 더 보고 싶네요", "글쎄요", 
        "별로예요", "생각보다 지루해요", "연기가 어색해요", 
        "재미없어요", "너무 재미없다", "참 재밌네요"]

# 긍정 : 1, 부정 : 0
labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1])
# 최종 출력 : 긍정인지? 부정인지?

# 토큰화
token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# 인덱스 생성 
# # 중복된 애들은 나오지 않음 ('너무' 같은 경우, 너무가 뒤에 또 나오지만 인덱스 1을 받고 나온 후에 다시 출력되지 않음, '참'도 마찬가지)
# {'너무': 1, '참': 2, '재밌어요': 3, '최고예요': 4, '잘': 5, '만든': 6, '영화예요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10, 
# '한': 11, '번': 12, '더': 13, '보고': 14, '싶네요': 15, '글쎄요': 16, '별로예요': 17, '생각보다': 18, 
# '지루해요': 19, '연기가': 20, '어색해요': 21, '재미없어요': 22, '재미없다': 23, '재밌네요': 24}

# '참'이 1번? -> 가장 많이 등장한 단어부터 인덱스를 부여! 나머지는 순서대로 
# {'참': 1, '너무': 2, '재밌어요': 3, '최고예요': 4, '잘': 5, '만든': 6, '영화예요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10, 
# '한': 11, '번': 12, '더': 13, '보고': 14, '싶네요': 15, '글쎄요': 16, '별로예요': 17, '생각# 보다': 18, 
# '지루해요': 19, '연기가': 20, '어색해요': 21, '재미없어요': 22, '재미없다': 23, '재밌네요': 24}

x = token.texts_to_sequences(docs)
print(x)
# [[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24]]
# 글자들이 인덱스로 바뀜 -> 문자가 수치화 됨
# 위의 문제점?
# 와꾸, shape
# 위의 경우에는 저마다 다 다른 와꾸를 가짐
# 어떻게 맞춰줌? 
# 지금까지 와꾸 어떻게 맞춰줌? reshape -> 현재 여기에선 불가능

from keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding = 'pre', value = 0.0) # 데이터, padding = 'pre' : 앞에서부터 채워진다 / padding = 'post' : 뒤에서부터 채워진다
                                                       # padding default = pre, value default = 0.0
print(pad_x)
print("pad_x.shape :",pad_x.shape) # (12, 5)




word_size = len(token.word_index) + 1
print("전체 토큰 사이즈 :", word_size) # 전체 토큰 사이즈 : 25 (0을 포함)

from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten

model = Sequential()
# model.add(Embedding(word_size, 10, input_length = 5)) # 와꾸 맞춰주는 부분
model.add(Embedding(25, 10, input_length = 5)) 
# 엥? word_size를 임의로 바꿨는데 돌아간다?
# 돌아는 가는데 accuracy에 영향을 끼치기 때문에
# 올바른 사이즈를 기입하는 것이 좋다? 
# (None, 5, 10)

# input_length 없애 본다
# model.add(Embedding(25, 10))

# word_size : 전체 토큰 사이즈, 전체 단어의 숫자
# 10 : output node , 출력 노드의 개수 ( 10을 넣든 100000을 넣든 상관 X, 다음 층으로 전달되는 NODE의 개수일 뿐)
# 5 : (12, 5) shape의 열
# 위의 레이어 다음 레이어들은 hidden layers
# Embedding = 벡터화!
# model.add(Flatten()) # -> Dense 바로 붙일 수 있다
model.add(Dense(1, activation = 'sigmoid'))

model.summary()


# 3. 컴파일, 훈련
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])

model.fit(pad_x, labels, epochs = 30)

# 4. 평가, 예측
acc = model.evaluate(pad_x, labels)[1]
# evluate 반환값 : 첫 번째 [0] = loss, 두 번째 [1] = metrics 지표(여기서는 acc)

print("ACC :", acc)



'''
padding = 'pre'
[[ 0  0  0  2  3]
 [ 0  0  0  1  4]
 [ 0  1  5  6  7]
 [ 0  0  8  9 10]
 [11 12 13 14 15]
 [ 0  0  0  0 16]
 [ 0  0  0  0 17]
 [ 0  0  0 18 19]
 [ 0  0  0 20 21]
 [ 0  0  0  0 22]
 [ 0  0  0  2 23]
 [ 0  0  0  1 24]]

padding = 'post'
[[ 2  3  0  0  0]
 [ 1  4  0  0  0]
 [ 1  5  6  7  0]
 [ 8  9 10  0  0]
 [11 12 13 14 15]
 [16  0  0  0  0]
 [17  0  0  0  0]
 [18 19  0  0  0]
 [20 21  0  0  0]
 [22  0  0  0  0]
 [ 2 23  0  0  0]
 [ 1 24  0  0  0]]

 padding = 'pre', value = 1.0
 [[ 1  1  1  2  3]
 [ 1  1  1  1  4]
 [ 1  1  5  6  7]
 [ 1  1  8  9 10]
 [11 12 13 14 15]
 [ 1  1  1  1 16]
 [ 1  1  1  1 17]
 [ 1  1  1 18 19]
 [ 1  1  1 20 21]
 [ 1  1  1  1 22]
 [ 1  1  1  2 23]
 [ 1  1  1  1 24]]
'''

