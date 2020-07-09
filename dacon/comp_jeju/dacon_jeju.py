# 1. Import Library

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

# 2. Data Cleansing & Pre-Processing
def grap_year(data) :
    data = str(data)
    return int(data[:4]) # 현재 데이터에서 날짜가 201904 이런 식이기 때문에 [:4] -> 연도만 남음

def grap_month(data) :
    data = str(data)
    return int(data[4:]) # 인덱스 4번부터면 월만 남음

# 날짜 처리
data = pd.read_csv('./dacon/comp_jeju/data/201901-202003.csv')
data = data.fillna('')
data['year'] = data['REG_YYMM'].apply(lambda x: grap_year(x)) 
data['month'] = data['REG_YYMM'].apply(lambda x: grap_month(x))
data = data.drop(['REG_YYMM'], axis = 1)

# 데이터 정제
df = data.copy()
df = df.drop(['CARD_CCG_NM', 'HOM_CCG_NM'], axis = 1) # 카드 이용지역_시군구, 거주지역_시군구(고객집주소기준)은 사용하지 않음 

columns = ['CARD_SIDO_NM', 'STD_CLSS_NM', 'HOM_SIDO_NM', 'AGE', 'SEX_CTGO_CD', 'FLC', 'year', 'month']
df = df.groupby(columns).sum().reset_index(drop=False)

# 인코딩
# 라벨인코더,, 공부 필요
dtypes = df.dtypes
encoders = {}
for column in df.columns :
    if str(dtypes[column]) == 'object' :
        encoder = LabelEncoder()
        encoder.fit(df[column])
        encoders[column] = encoder

df_num = df.copy()
for column in encoders.keys() :
    encoder = encoders[column]
    df_num[column] = encoder.transform(df[column])

# 3. 탐색적 자료분석 , Exploratory Data Analysis
# ? describe 이런 거 쓰면 되나


# 4. 변수 선택 및 모델 구축, Feature Engineering & Initial Modeling
# feature, target 설정
train_num = df_num.sample(frac = 1, random_state = 0)
train_features = train_num.drop(['CSTMR_CNT', 'AMT', 'CNT'], axis = 1) # 이용고객수, 이용금액, 이용건수
train_target = np.log1p(train_num['AMT']) # 이용금액


print("train_features.shape :", train_features.shape) # train_features.shape : (1057394, 8)
print("train_target.shape :", train_target.shape) # train_target.shape : (1057394,)




# 5. 모델 학습 및 검증, Model Tuning & Evaluation

# 훈련
# model = RandomForestRegressor(n_estimators = 500, max_depth = 5, min_samples_split = 7,
#                              max_leaf_nodes = 5, max_samples = 5, n_jobs = -1, random_state = 0)

# model = LGBMRegressor(n_estimators = 500, max_depth = 7, learning_rate = 0.09, max_bin = 10, num_leaves = 10, n_jobs = -1)

model = XGBRegressor(n_estimators = 500, max_depth = 8, learning_rate = 0.01, max_bin = 150, 
                    colsample_bytree = 0.7, colsample_bylevel = 0.7, n_jobs = -1)


model.fit(train_features, train_target)

# 6. 결과 및 결언, Conclusion & Discussion

# 예측 템플릿 만들기
CARD_SIDO_NMs = df_num['CARD_SIDO_NM'].unique()
STD_CLSS_NMs = df_num['STD_CLSS_NM'].unique()
HOM_SIDO_NMs = df_num['HOM_SIDO_NM'].unique()
AGEs = df_num['AGE'].unique()
SEX_CTGO_CDs = df_num['SEX_CTGO_CD'].unique()
FLCs = df_num['FLC'].unique()
years = [2020]
months = [4, 7]

temp = []
for CARD_SIDO_NM in CARD_SIDO_NMs :
    for STD_CLSS_NM in STD_CLSS_NMs :
        for HOM_SIDO_NM in HOM_SIDO_NMs :
            for AGE in AGEs :
                for SEX_CTGO_CD in SEX_CTGO_CDs :
                    for FLC in FLCs :
                        for year in years :
                            for month in months :
                                temp.append([CARD_SIDO_NM, STD_CLSS_NM, HOM_SIDO_NM, AGE, SEX_CTGO_CD, FLC, year, month])

temp = np.array(temp)
temp = pd.DataFrame(data=temp, columns=train_features.columns)                        

# 예측
pred = model.predict(temp)
pred = np.expm1(pred)
temp['AMT'] = np.round(pred, 0)
temp['REG_YYMM'] = temp['year'] * 100 + temp['month']
temp = temp[['REG_YYMM', 'CARD_SIDO_NM', 'STD_CLSS_NM', 'AMT']]
temp = temp.groupby(['REG_YYMM', 'CARD_SIDO_NM', 'STD_CLSS_NM']).sum().reset_index(drop=False)

# 디코딩
temp['CARD_SIDO_NM'] = encoders['CARD_SIDO_NM'].inverse_transform(temp['CARD_SIDO_NM'])
temp['STD_CLSS_NM'] = encoders['STD_CLSS_NM'].inverse_transform(temp['STD_CLSS_NM'])

# 제출 파일 만들기
submission = pd.read_csv('./dacon/comp_jeju/data/submission.csv', index_col=0)
submission = submission.drop(['AMT'], axis=1)
submission = submission.merge(temp, left_on=['REG_YYMM', 'CARD_SIDO_NM', 'STD_CLSS_NM'], right_on=['REG_YYMM', 'CARD_SIDO_NM', 'STD_CLSS_NM'], how='left')
submission.index.name = 'id'
submission.to_csv('./dacon/comp_jeju/0709/submission_0709_1.csv', encoding='utf-8-sig')
submission.head()
print(submission.head())


'''
submission_0630 
default

submission_0630_1
model = RandomForestRegressor(n_estimators = 300, max_depth = 8, min_samples_split = 3,
                             max_leaf_nodes = 10, max_samples = 10, n_jobs = -1, random_state = 0)

submission_0630_2
model = RandomForestRegressor(n_estimators = 500, max_depth = 5, min_samples_split = 7,
                             max_leaf_nodes = 5, max_samples = 5, n_jobs = -1, random_state = 0)

submission_0630_3
model = LGBMRegressor(n_estimators = 500, max_depth = 7, learning_rate = 0.09, max_bin = 10, num_leaves = 10, early_stopping_round = 50, n_jobs = -1)
'''

'''
submission_0709_1

model = XGBRegressor(n_estimators = 500, max_depth = 8, learning_rate = 0.01, max_bin = 150, 
                    colsample_bytree = 0.7, colsample_bylevel = 0.7, n_jobs = -1)

'''