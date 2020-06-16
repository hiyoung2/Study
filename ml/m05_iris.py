# 1. 데이터 

# 1-1. 데이터 불러오기
from sklearn.datasets import load_iris

iris = load_iris()
x = iris['data']
y = iris['target']


print(type(iris)) # <class 'sklearn.utils.Bunch'>

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, r2_score


# logistic regressor : 분류

print(x)
print(y)
print("x.shape : " , x.shape) # (150, 4)
print("y.shape : ", y.shape)  # (150,)

# one hot encoder X -> ok (in ml)

# scaler
scaler = StandardScaler()

scaler.fit(x)
x = scaler.transform(x)

# train_test splie
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, shuffle = True # default : True
)

# 2. 모델 구성

# model = SVC() 
# SCORE :  1.0
# ACC :  1.0
# R2 :  1.0

# model = LinearSVC() 
# SCORE :  0.9
# ACC :  0.9
# R2 :  0.8646616541353382

# model = KNeighborsRegressor(n_neighbors=2) 

# n = 1
# SCORE :  0.9333333333333333
# ACC :  0.9333333333333333
# R2 :  0.8588235294117648

# n = 2
# SCORE :  0.9333333333333333
# ACC :  0.9333333333333333
# R2 :  0.8993288590604027


# model = KNeighborsClassifier(n_neighbors=2)

# n = 1  
# SCORE :  0.9666666666666667
# ACC :  0.9666666666666667
# R2 :  0.9440298507462687

# n = 2
# SCORE :  1.0
# ACC :  1.0
# R2 :  1.0

# model = RandomForestRegressor()
# 회귀 모델이라 acc를 지표로 쓰면 실행이 안 된다
# score만 썼을 때 실행O 
# SCORE :  0.9645362318840581

# score, r2 썼을 때 실행O
# SCORE :  0.8487742537313433
# R2 :  0.8487742537313433

model = RandomForestClassifier() 
# SCORE :  1.0
# ACC :  1.0
# R2 :  1.0

# 3. 훈련
model.fit(x_train, y_train)
score = model.score(x_test, y_test) 
# model.score 은 keras에서 model.evaluate와 같다
# 회귀건, 분류건 상관없이 쓸 수 있고
# 회귀면 r2, 분류면 acc로 자동으로 알아서 처리한다


# 4. 평가, 예측
y_predict = model.predict(x_test)

print("===================================")
print("y_predict.shape :", y_predict.shape)
print("y_test.shape :", y_test.shape)
print("===================================")



acc = accuracy_score(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

# print(x_test, "의 예측 결과 : ", y_predict)
print("SCORE : ", score)
print("ACC : ", acc)
print("R2 : ", r2)


# in classifier : score = accuracy_score
# in regressor : score = r2_score

