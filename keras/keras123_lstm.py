# 122 file cpoy, 123 완성
# embedding을 빼고 LSTM으로 완성
# 자연어 처리 시, embedding 없이도 가능은 하다!

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

# lstm 모델에 넣기 위해 reshape
pad_x = pad_x.reshape(pad_x.shape[0], pad_x.shape[1], 1)
print("pad_x.reshape :", pad_x.shape) # (12, 5, 1)

word_size = len(token.word_index) + 1
print("전체 토큰 사이즈 :", word_size)

from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, LSTM

model = Sequential()
model.add(LSTM(10, input_shape = (5, 1)))
model.add(Dense(1, activation = 'sigmoid'))

model.summary()

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])

model.fit(pad_x, labels, epochs = 30)

# 4. 평가, 예측
acc = model.evaluate(pad_x, labels)[1]
# evluate 반환값 : 첫 번째 [0] = loss, 두 번째 [1] = metrics 지표(여기서는 acc)

print("ACC :", acc)
