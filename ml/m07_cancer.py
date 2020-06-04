from sklearn.datasets import load_breast_cancer
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score


cancer = load_breast_cancer()
x = cancer['data']
y = cancer['target']

print(x)
print(y)

print("x.shape : ", x.shape) # (569, 30)
print("y.shape : ", y.shape) # (569,)

# train_test splie
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8
)

# scaler
scaler = StandardScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)

scaler.fit(x_test)
x_test = scaler.transform(x_test)

# 2. 모델 구성
# model = LinearSVC()
# SCORE :  0.9912280701754386
# ACC :  0.9912280701754386
# R2 :  0.9623015873015873

# model = SVC()
# SCORE :  0.9824561403508771
# ACC :  0.9824561403508771
# R2 :  0.922972972972973


# model = KNeighborsRegressor(n_neighbors=2)
#  acc 포함하면 실행 X

# n = 1 / ? 여기서는 acc 넣어도 돌아감
# SCORE :  0.9824561403508771
# ACC :  0.9824561403508771
# R2 :  0.9246031746031746

# n = 2 / 여기서는 acc 넣으면 안 돌아감
# SCORE :  0.9173913043478261
# R2 :  0.9173913043478261

# model = KNeighborsClassifier(n_neighbors=2)

# n = 1
# SCORE :  0.9649122807017544
# ACC :  0.9649122807017544
# R2 :  0.8492063492063492

# n = 2
# SCORE :  0.9736842105263158
# ACC :  0.9736842105263158
# R2 :  0.8879790370127744

# model = RandomForestRegressor()
# acc, r2 포함 안 시키면 실행 O
# SCORE :  0.8665711636828644

model = RandomForestClassifier()

# SCORE :  0.9649122807017544
# ACC :  0.9649122807017544
# R2 :  0.8441025641025641


# 3. 훈련
model.fit(x_train, y_train)
score = model.score(x_test, y_test)

# 4. 평가 , 예측'
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

# print(x_test, "의 예측 결과 : ", y_predict)
print("SCORE : ", score)
print("ACC : ", acc)
print("R2 : ", r2)
