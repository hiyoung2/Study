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
from keras.layers import Dense, Embedding, Flatten, LSTM

model = Sequential()
# model.add(Embedding(word_size, 10, input_length = 5)) 
# model.add(Embedding(25, 10, input_length = 5)) # (None, 5, 10) -> LSTM, Conv1D 가능하다
# model.add(Embedding(25, 10))
model.add(Embedding(25, 10, input_length = 5))  # 전체 데이터의 크기, output node, input shape

# input_length를 빼버림 -(Flatten - Dense 모델로 이어지는 상태에서)
# ValueError: The shape of the input to "Flatten" is not fully defined (got (None, 10)).
# Make sure to pass a complete "input_shape" or "batch_input_shape" argument to the first layer in your model.
# 에러 발생 -> shape 틀렸다는 말 -> x에 대한 input이 없다는 에러가 떠야 하는데 shape가 안 맞다는 에러가 뜸 엥???
# Flatten 주석 처리 후 LSTM을 붙여보자 -> 실행이 된다
# x data에 대한 input 값이 없는데도 훈련이 된다
model.add(LSTM(3))
# model.add(Flatten())
model.add(Dense(1, activation = 'sigmoid'))

model.summary()

# Embedding 다음에 LSTM을 결합하면 input_length를 명시하지 않아도 된다(문법적?)
# input_length를 명시하는 것도 상관없다
# Embedding LSTM을 가장 많이 쓴다(자연어처리에서 LSTM을 가장 많이 사용)

# Embedding을 안 넣어도 된다?
# shape을 (12, 5, 1)로 reshape 해서 바로 LSTM 모델로 돌릴 수 있다? -> keras123_LSTM.py 파일로
'''
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding_1 (Embedding)      (None, None, 10)          250
_________________________________________________________________
lstm_1 (LSTM)                (None, 3)                 168
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 4
=================================================================
Total params: 422
Trainable params: 422
Non-trainable params: 0
_________________________________________________________________

embedding param : 250
word_size * output = 25 * 10 = 250

LSTM Param : 168
4 * (input_dim + bias + output) * output
4 * (10 + 1 + 3) * 3
input_dim = 10


# input_length 를 명시했을 때 -> 파라미터 변함X
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding_1 (Embedding)      (None, 5, 10)             250
_________________________________________________________________
lstm_1 (LSTM)                (None, 3)                 168
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 4
=================================================================
Total params: 422
Trainable params: 422
Non-trainable params: 0
_________________________________________________________________
'''

# 3. 컴파일, 훈련
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])

model.fit(pad_x, labels, epochs = 30)

# 4. 평가, 예측
acc = model.evaluate(pad_x, labels)[1]
# evluate 반환값 : 첫 번째 [0] = loss, 두 번째 [1] = metrics 지표(여기서는 acc)

print("ACC :", acc)
