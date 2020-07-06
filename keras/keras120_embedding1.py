# 2020.07.06

# embedding 하기전에 Tokenizer 먼저 쓴다
# 버스 탈 때 토큰 ㅋㅋ
# 토큰 : 프로그래밍 언어에서의 토큰은, 문법적으로 더 이상 나눌 수 없는 기본적인 언어요소
from keras.preprocessing.text import Tokenizer

text = "나는 맛있는 밥을 먹었다"

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index)
# {'나는': 1, '맛있는': 2, '밥을': 3, '먹었다': 4}
# 단어별로(조사포함) 잘라서 인덱스를 걸어줌
# 띄어쓰기 단위로 잘라주는 듯

x = token.texts_to_sequences([text]) # 문자를 순서로 정한다?
print(x) 
# [[1, 2, 3, 4]]

# 모든 문자를 수치화 시키면? 모델을 돌릴 수 있다
# 맛있는 은 나는 에 비해 2배의 가치? 밥을 은 나는 에 비해 3배의 가치? NOPE
# 만 개의 단어를 수치화 했다
# 예를 들어 '오므라이스' 10000 번째, '카레' 100번째, 100배의 가치 차이?
# NOPE 
# 이럴 때 '원 핫 인코딩' 필요

from keras.utils import to_categorical # 0부터 시작
# 사이킷런의 one hot encoding 써도 됨

word_size = len(token.word_index) +1  # 1부터 시작하므로 +1을 해 준다
x = to_categorical(x, num_classes=word_size)
print(x)

# 지금은 4개의 단어지만, 10000개가 된다면?
# 데이터 양이 너무 많아진다
# 어떻게?
# 압축! -> embedding!

# embedding : 자연어처리에서 상당히 많이 쓰고, 시계열에서도 많이 쓰인다

