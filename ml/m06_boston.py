# 1. 데이터
from sklearn.datasets import load_boston
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score


boston = load_boston()
x = boston['data']
y = boston['target']

print(type(boston))

print(x)
print(y)

print("x.shape : ", x.shape) # (506, 13)
print("y.shape : ", y.shape) # (506,)


# train_test splie
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state= 1, shuffle = True
)

# scaler
scaler = StandardScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성

# model = LinearSVC()
# ValueError: Unknown label type: 'continuous'
# 실행 되지 않음

# model = SVC()
# ValueError: Unknown label type: 'continuous'
# 실행 되지 않음

# model = KNeighborsRegressor(n_neighbors=2)

# 회귀모델이라 acc를 포함시키면 실행X
# n = 1
# SCORE :  0.8358206726828242
# R2 :  0.8358206726828242

# SCORE :  0.7207459880168837

# n = 2
# SCORE :  0.8697469031710714
# R2 :  0.8697469031710714

# n = 3
# R2 :  0.8751729641382422
# SCORE :  0.8751729641382422

# model = KNeighborsClassifier(n_neighbors=2)
# acc를 포함 안 시키고 r2, score만 설정해서 시켜도 실행 X
# 아예 전체 다 실행 X
# n = 1
# ValueError: Unknown label type: 'continuous'

# n = 2
# ValueError: Unknown label type: 'continuous'

model = RandomForestRegressor()
# acc 포함 안 시키면 실행O
# SCORE :  0.9124557015299433
# R2 :  0.912455701529943

# model = RandomForestClassifier()
# ValueError: Unknown label type: 'continuous'
# 전체적으로 다 실행 X

# 3. 훈련
model.fit(x_train, y_train)

score = model.score(x_test, y_test)

# 4. 평가 , 예측'
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
# acc = accuracy_score(y_test, y_predict)


print(x_test, "의 예측 결과 : ", y_predict)
print("SCORE : ", score)
print("R2 : ", r2)
# print("ACC : ", acc)


