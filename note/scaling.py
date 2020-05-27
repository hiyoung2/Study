# 한 쪽으로 치우친 데이터, 평균값 내면 이상함
# 데이터 폭이 너무 큰 이러한 데이터는 scaling을 통해 해결할 수 있다 (앞에서 train_test_split 쓴 것처럼 항상 쓰게 될 것)

# 찾아보기 : MinMaxScaler, StandardScaler (공식, 계산 알아야함)
# 스케일링에는 
# 1. StandardScaler
# 2. MinMaxScaler
# 3. MaxAbsScaler
# 4. RobustScalr 이렇게 4가지가 있는데
# MinMaxScaler, StandardScaler가 가장 대표적으로 잘 쓰임
# 모두 사이킷런에서 제공

#### 1. StandardScaler(표준화, Standardization) #####
# : 기본 스케일. 평균과 표준편차를 사용
# 평균을 제거하고 데이터를 단위 분산으로 조정한다.
# 그러나 이상치가 있다면 평균과 표준편차에 영향을 미쳐 변환된 데이터의 확산은 매우 달라지게 된다. 
# 즉, 이상치가 있는 경우 균형 잡힌 척도가 보장X

#### 2. MinMaxScaler(정규화, Normalization) #####
# : 최대/최솟값이 각각 1, 0이 되도록 스케일링
# 모든 feature 값이 0~1 사이에 있도록 데이터를 재조정한다. 
# 다만 , 이상치가 있는 경우 변환된 값이 매우 좁은 범위로 압출될 수 있다.
# 역시 이상치(outlier)의 존재에 매우 민감하다

# 3. MaxAbsScaler
# : 절댓값이 0 ~ 1 사이에 mapping 되도록 한다. 
# 즉 -1 ~ 1 사이로 재조정한다.  
# 양수 데이터로만 구성된 특징 dataset에서는 minimaxScaler와 유사하게 동작
# 큰 이상치에 민감할 수 있다

# 4. RobustSclaer
# : 이상치(outlier)의 영향을 최소하한 기법
# 중앙값과 IQR(interquartile range)을 사용
# 표준화 후 동일한 값을 더 넓게 분포

# IQR = Q3 - Q1 : 즉 25퍼센타일과 75퍼센타일의 값들을 다룬다
# 중요한 게 아니라 당연히 하는 것이라고 한다,,,,,,,,,,,,
