# keras122_embedding3를 copy, conv1d로 구성

from keras.preprocessing.text import Tokenizer
import numpy as np

docs = ["너무 재밌어요", "참 최고예요", "참 잘 만든 영화예요", 
        "추천하고 싶은 영화입니다", "한 번 더 보고 싶네요", "글쎄요", 
        "별로예요", "생각보다 지루해요", "연기가 어색해요", 
        "재미없어요", "너무 재미없다", "참 재밌네요"]

# 긍정 : 1, 부정 : 0
labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1])

# 토큰화
token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)


x = token.texts_to_sequences(docs)
print(x)

from keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding = 'pre', value = 0.0) 
print(pad_x)
print("pad_x.shape :",pad_x.shape) # (12, 5)

word_size = len(token.word_index) + 1
print("전체 토큰 사이즈 :", word_size)

from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, Conv1D

model = Sequential()
model.add(Embedding(25, 10, input_length = 5))  # 전체 데이터의 크기, output node, input shape
model.add(Conv1D (10, 3)) 
# Embedding layer 다음에 Conv1D도 사용 가능 (LSTM 처럼)
# input_shape는 Embedding 에서 써 줬기 때문에 output_node와 kernel_size만 입력하면 된다
# Conv1D기 때문에 Dense layer 바로 못 받으므로 Flatten layer 를 써 준다
model.add(Flatten())
model.add(Dense(1, activation = 'sigmoid'))

model.summary()

# 3. 컴파일, 훈련
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])

model.fit(pad_x, labels, epochs = 30)

# 4. 평가, 예측
acc = model.evaluate(pad_x, labels)[1]
# evluate 반환값 : 첫 번째 [0] = loss, 두 번째 [1] = metrics 지표(여기서는 acc)

print("ACC :", acc)
