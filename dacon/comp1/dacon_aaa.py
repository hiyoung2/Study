import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts, GridSearchCV  # as로 간단하게 이름 만들어주고, 다른 것 이어서 import 가능
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, mean_absolute_error as mae # 꿀팁이다, 바로 mae로 별칭 부여
from lightgbm import LGBMRegressor, plot_importance

for size in np.arange(0.91, 0.911, 0.01) : # tts 할 때 train(or test) size를 다양하게 설정하여 돌려볼 수 있다

    # 데이터 전처리, 준비
    train = pd.read_csv("./data/dacon/comp1/train.csv", index_col = 0, header = 0, encoding = "cp949") # sep은 default : , / encoding은 일단 기본적으로?
    test = pd.read_csv("./data/dacon/comp1/test.csv", index_col = 0, header = 0, encoding = "cp949")
    submission = pd.read_csv("./data/dacon/comp1/sample_submission.csv", index_col = 0, header = 0, encoding = "cp949")

    # shape 확인
    print('train.shape :', train.shape) # (10000, 75) -> x_train, x_test, y_train, y_test 로 사용할 데이터
    print('test.shape :', test.shape) # (10000, 71) -> x_pred, 최종 submit 할 데이터에 사용할 input data
    print('submission.shape :', submission.shape) # (10000, 4) -> 최종 submit 할 데이터

    # print(train.isnull().sum()) # 컬럼별로 결측치의 총 개수를 알 수 있다
    # 그런데 컬럼이 많아서 중간에 ... 으로 생략이 되는데 생략 되는 부분의 null의 개수는 알 수 없는 건가? (여기선 없어서 그런 건가)

    '''
    rho           0
    650_src       0
    660_src       0
    670_src       0
    680_src       0
            ...
    990_dst    1987 # null이 1987개 있다
    hhb           0
    hbo2          0
    ca            0
    na            0
    Length: 75, dtype: int64
    '''

    print("===========================================================================================")
    # print(train.notnull().sum()) # 컬럼별 결측값이 아닌 값(결측치가 아닌)의 갯수를 구할 때 사용

    # train = train.interpolate() # 보간법 - 선형보간법 : 첫 번째에 결측치가 있으면 적용이 안 됨
    '''
    rho        10000
    650_src    10000
    660_src    10000
    670_src    10000
    680_src    10000
            ...
    990_dst     8013
    hhb        10000
    hbo2       10000
    ca         10000
    na         10000
    Length: 75, dtype: int64
    '''


    # print(dir(train)) -> 써 준 이유??
    # dir : 파이썬의 내장함수
    # 어떤 객체를 인자로 넣어주면 해당 객체가 어떤 변수와 method를 가지고 있는지 나열해준다
    print("===========================================================================================")
    # print(train.isnull().sum())
    '''
    rho        0
    650_src    0
    660_src    0
    670_src    0
    680_src    0
            ..
    990_dst    0
    hhb        0
    hbo2       0
    ca         0
    na         0
    Length: 75, dtype: int64
    '''

    # print(train.info()) # 데이터프레임 자료구조를 설명해주는 함수
    # train 변수에 저장된 자료 구조 : <class 'pandas.core.frame.DataFrame'>
    # index의 범위 : Int64Index: 10000 entries, 0 to 9999 -> 총 10000 개의 데이터를 가지고 있음
    # 컬럼의 수 : Data columns (total 75 columns):
    '''
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 10000 entries, 0 to 9999
    Data columns (total 75 columns):
    #   Column   Non-Null Count  Dtype
    ---  ------   --------------  -----
    0   rho      10000 non-null  int64
    1   650_src  10000 non-null  float64
    2   660_src  10000 non-null  float64
    3   670_src  10000 non-null  float64
    4   680_src  10000 non-null  float64
    5   690_src  10000 non-null  float64
    6   700_src  10000 non-null  float64
    7   710_src  10000 non-null  float64
    8   720_src  10000 non-null  float64
    9   730_src  10000 non-null  float64
    10  740_src  10000 non-null  float64
    11  750_src  10000 non-null  float64
    12  760_src  10000 non-null  float64
    13  770_src  10000 non-null  float64
    14  780_src  10000 non-null  float64
    15  790_src  10000 non-null  float64
    16  800_src  10000 non-null  float64
    17  810_src  10000 non-null  float64
    18  820_src  10000 non-null  float64
    19  830_src  10000 non-null  float64
    20  840_src  10000 non-null  float64
    21  850_src  10000 non-null  float64
    22  860_src  10000 non-null  float64
    23  870_src  10000 non-null  float64
    24  880_src  10000 non-null  float64
    25  890_src  10000 non-null  float64
    26  900_src  10000 non-null  float64
    27  910_src  10000 non-null  float64
    28  920_src  10000 non-null  float64
    29  930_src  10000 non-null  float64
    30  940_src  10000 non-null  float64
    31  950_src  10000 non-null  float64
    32  960_src  10000 non-null  float64
    33  970_src  10000 non-null  float64
    34  980_src  10000 non-null  float64
    35  990_src  10000 non-null  float64
    36  650_dst  10000 non-null  float64
    37  660_dst  10000 non-null  float64
    38  670_dst  10000 non-null  float64
    39  680_dst  10000 non-null  float64
    40  690_dst  10000 non-null  float64
    41  700_dst  10000 non-null  float64
    42  710_dst  9999 non-null   float64
    43  720_dst  10000 non-null  float64
    44  730_dst  9998 non-null   float64
    45  740_dst  10000 non-null  float64
    46  750_dst  10000 non-null  float64
    47  760_dst  9999 non-null   float64
    48  770_dst  10000 non-null  float64
    49  780_dst  9999 non-null   float64
    50  790_dst  10000 non-null  float64
    51  800_dst  10000 non-null  float64
    52  810_dst  9998 non-null   float64
    53  820_dst  10000 non-null  float64
    54  830_dst  10000 non-null  float64
    55  840_dst  10000 non-null  float64
    56  850_dst  10000 non-null  float64
    57  860_dst  10000 non-null  float64
    58  870_dst  9999 non-null   float64
    59  880_dst  10000 non-null  float64
    60  890_dst  10000 non-null  float64
    61  900_dst  9999 non-null   float64
    62  910_dst  10000 non-null  float64
    63  920_dst  9999 non-null   float64
    64  930_dst  10000 non-null  float64
    65  940_dst  9999 non-null   float64
    66  950_dst  10000 non-null  float64
    67  960_dst  9999 non-null   float64
    68  970_dst  10000 non-null  float64
    69  980_dst  10000 non-null  float64
    70  990_dst  10000 non-null  float64
    71  hhb      10000 non-null  float64
    72  hbo2     10000 non-null  float64
    73  ca       10000 non-null  float64
    74  na       10000 non-null  float64
    dtypes: float64(74), int64(1)
    memory usage: 5.8 MB
    None
    '''


    print("===========================================================================================")
    # print(train.describe())
    # 숫자 데이터 (Numeric Data)가를 가지고 있는 column 별 데이터의 개수, 평균, 표준편차, 최솟값, 최댓값 (2, 3분위)과 같은 통계값을 요약해서 보여준다
    '''
                    rho       650_src       660_src       670_src       680_src       690_src       700_src  ...       970_dst       980_dst       990_dst           hhb          hbo2            ca            na
    count  10000.000000  10000.000000  10000.000000  10000.000000  10000.000000  10000.000000  10000.000000  ...  1.000000e+04  1.000000e+04  1.000000e+04  10000.000000  10000.000000  10000.000000  10000.000000
    mean      17.568000      0.180212      0.203529      0.229804      0.259158      0.289975      0.322244  ...  4.087505e-11  1.131517e-10  2.415016e-10      7.990686      4.009146      9.019226      3.042651
    std        5.595847      0.272859      0.288661      0.306340      0.324849      0.342362      0.357700  ...  2.028121e-10  4.866314e-10  1.018758e-09      2.970818      0.997828      2.979453      1.881872
    min       10.000000      0.000000      0.000000      0.000000      0.000000      0.000000      0.000000  ...  0.000000e+00  0.000000e+00  0.000000e+00      0.000000      0.080000      0.000000      0.000000
    25%       15.000000      0.007318      0.009520      0.011270      0.013728      0.016350      0.019510  ...  2.583714e-20  1.577532e-17  1.598346e-16      5.990000      3.330000      7.000000      1.640000
    50%       20.000000      0.052025      0.064600      0.078565      0.095355      0.119635      0.148590  ...  4.922208e-16  9.710900e-15  4.835940e-14      8.010000      4.010000      8.990000      2.980000
    75%       25.000000      0.221117      0.276952      0.339075      0.417478      0.510915      0.607150  ...  9.249667e-13  4.963447e-12  1.306097e-11     10.010000      4.700000     11.020000      4.330000
    max       25.000000      1.019990      1.019970      1.019930      1.019880      1.020000      1.019970  ...  5.529940e-09  1.013262e-08  2.682289e-08     21.590000      7.690000     20.070000     10.310000

    [8 rows x 75 columns]
    '''
    # print(train.columns.values) # column 명칭 확인
    '''
    ['rho' '650_src' '660_src' '670_src' '680_src' '690_src' '700_src'
    '710_src' '720_src' '730_src' '740_src' '750_src' '760_src' '770_src'
    '780_src' '790_src' '800_src' '810_src' '820_src' '830_src' '840_src'
    '850_src' '860_src' '870_src' '880_src' '890_src' '900_src' '910_src'
    '920_src' '930_src' '940_src' '950_src' '960_src' '970_src' '980_src'
    '990_src' '650_dst' '660_dst' '670_dst' '680_dst' '690_dst' '700_dst'
    '710_dst' '720_dst' '730_dst' '740_dst' '750_dst' '760_dst' '770_dst'
    '780_dst' '790_dst' '800_dst' '810_dst' '820_dst' '830_dst' '840_dst'
    '850_dst' '860_dst' '870_dst' '880_dst' '890_dst' '900_dst' '910_dst'
    '920_dst' '930_dst' '940_dst' '950_dst' '960_dst' '970_dst' '980_dst'
    '990_dst' 'hhb' 'hbo2' 'ca' 'na']
    '''
    print()
    # print(train.index.values) # index 명칭 확인
    # [   0    1    2 ... 9997 9998 9999]

    # 불필요한 컬럼(column) 삭제하기 -> drop() 함수를 사용하여 삭제할 수 있다
    # 함수 인자로 [COLUMN 명칭]으로 구성되는 리스트를 전달, 삭제가 이루어지는 축방향(axis)를 1로 설정한다 
    # (판다스 axis = 1 : 열, column / aixs = 0 : 행, index)
    # inplace 옵션을 True로 지정해야만, df(데이터 이름)라는 변수가 가리키는 원본 데이터 프레임이 수정이 된다
    # ex) train.drop(['650_src', '660_src'], axis = 1, inplace = True) 이런 식으로 사용한다
    # 불필요한 데이터를 drop 했다면, 작업한 결과를 새로운 파일로 저장해야한다
    # new_train.to_csv("./파일경로/파일명.csv")


    # 현재 이 데이터는 transpose로 행과 열을 바꿔서 interpolate 하는 게 맞다는 의견(?)이 있어서 일단 그렇게 적용
    train = train.transpose()
    test = test.transpose()

    train = train.interpolate()
    test = test.interpolate()

    train = train.transpose() # 다시 원상복귀
    test = test.transpose()

    # 혹시 모를 결측치를 채워주기?
    # (1) 0으로 채워주기
    # train = train.fillna(0)
    # test = test.fillna(0)

    # (2) 평균으로 채워주기
    # train = train.fillna(data.mean())
    # test = test.fillna(data.mean())

    # 한 번 더 확인
    # print(train.info())
    # print(test.info())

    x = train.values[:, :-4] # x 데이터 준비, values로 수치들만 남겨준다, train에서 슬라이싱으로 x 준비(마지막 4 column을 빼고 모두 x 데이터가 됨)
    y = train.values[:, -4:] # y 데이터 준비, 역시 values로 수치들만 남겨주고, 마지막 4개의 column을 y 데이터로 쓴다

    test = test.values

    # 슬라이싱 후 shape 확인
    print("x.shape :", x.shape) # (10000, 71)
    print("y.shape :", y.shape) # (10000, 4)

    # train_test_split
    x_train, x_test, y_train, y_test = tts(x, y, train_size = size, random_state = 66)
    # train_test_split을 tts라는 별칭으로 만들어줬기 때문에 간편하게 쓸 수 있다!
    # train_size를 평소 했던 것처럼 0.8 or 0.7 과 같이 고정시키지 않고 size라는 변수를 사용하여 여러 비율로 테스트 해 볼 수 있다(상위에서 for문을 사용하여)

    # 현재, y data의 column 4개
    # boost 계열의 모델은 일반 모델과 달리 컬럼이 하나를 초과하면 컬럼별로 하나씩 돌려주거나 multioutput을 사용해야 함
    # multioutput 사용하니까 feature_importance를 쓰기가 까다로움
    # 그래서 아예 y data를 column 별로 하나씩 분리해서 4번을 돌린다(for문을 사용하거나 하나씩 쪼개서 모델을 돌려도 된다)

    model_0 = LGBMRegressor() # 파라미터는 그리드서치를 통해 최적 파라미터를 넣으면 된다
    model_1 = LGBMRegressor()
    model_2 = LGBMRegressor()
    model_3 = LGBMRegressor()

    y_train_0 = y_train[:, 0] # 행은 다 가져오고 첫 번째 컬럼(index로는 0) 준비(4개의 컬럼을 하나씩 떼어서 y_train을 4개 만든다)
    y_train_1 = y_train[:, 1]
    y_train_2 = y_train[:, 2]
    y_train_3 = y_train[:, 3]

    y_test_0 = y_test[:, 0]
    y_test_1 = y_test[:, 1]
    y_test_2 = y_test[:, 2]
    y_test_3 = y_test[:, 3]

    parameters = {}

    # LGBM 모델 fit
    # verbose = 0 : 훈련과정 아무것도 안 보여줌
    # 4개씩 쪼갠 것을 하나씩 훈련 시킨다
    model_0.fit(x_train, y_train_0, verbose = 0, eval_metric = ['mae'] , eval_set = [(x_train, y_train_0, x_test, y_test_0)], early_stopping_rounds = 10)
    model_1.fit(x_train, y_train_0, verbose = 0, eval_metric = ['mae'] , eval_set = [(x_train, y_train_1, x_test, y_test_1)], early_stopping_rounds = 10)
    model_2.fit(x_train, y_train_0, verbose = 0, eval_metric = ['mae'] , eval_set = [(x_train, y_train_2, x_test, y_test_2)], early_stopping_rounds = 10)
    model_3.fit(x_train, y_train_0, verbose = 0, eval_metric = ['mae'] , eval_set = [(x_train, y_train_3, x_test, y_test_3)], early_stopping_rounds = 10)

    score_0 = model_0.score(x_test, y_test_0)
    score_1 = model_1.score(x_test, y_test_1)
    score_2 = model_2.score(x_test, y_test_2)
    score_3 = model_3.score(x_test, y_test_3)

    print('r2_0 :', score_0)
    print('r2_1 :', score_1)
    print('r2_2 :', score_2)
    print('r2_3 :', score_3)

    print(model_0.feature_importances_)
    print(model_1.feature_importances_)
    print(model_2.feature_importances_)
    print(model_3.feature_importances_)

#     plot_importance(model_0)
#     plot_importance(model_1)
#     plot_importance(model_2)
#     plot_importance(model_3)

#     plt.show()

    thresholds_0 = np.sort(model_0.feature_importances_)
    thresholds_1 = np.sort(model_1.feature_importances_)
    thresholds_2 = np.sort(model_2.feature_importances_)
    thresholds_3 = np.sort(model_3.feature_importances_)

    print(thresholds_0)
    print(thresholds_1)
    print(thresholds_2)
    print(thresholds_3)


'''
    thresholds_0 = np.sort(model_0.feature_importances_)
    thresholds_1 = np.sort(model_1.feature_importances_)
    thresholds_2 = np.sort(model_2.feature_importances_)
    thresholds_3 = np.sort(model_3.feature_importances_)

    # selection_n : selecfrommodel 적용
    selection_0 = SelectFromModel(model_0, threshold = thresholds_0[-1], prefit = True)
    selection_1 = SelectFromModel(model_1, threshold = thresholds_1[-1], prefit = True)
    selection_2 = SelectFromModel(model_2, threshold = thresholds_2[-1], prefit = True)
    selection_3 = SelectFromModel(model_3, threshold = thresholds_3[-1], prefit = True)
    
    # selection_model_n(최종 사용 모델) : LGBMRegessor 적용
    selection_model_0 = LGBMRegressor() # 파라미터 넣으면 됨
    selection_model_1 = LGBMRegressor()
    selection_model_2 = LGBMRegressor()
    selection_model_3 = LGBMRegressor()
    
    # 최종 사용 모델에 GridSearchCV 사용
    selection_model_0 = GridSearchCV(selection_model_0, {}, cv = 5)
    selection_model_1 = GridSearchCV(selection_model_1, {}, cv = 5)
    selection_model_2 = GridSearchCV(selection_model_2, {}, cv = 5)
    selection_model_3 = GridSearchCV(selection_model_3, {}, cv = 5)

    # 최종 사용 모델에 넣기 위한 x_train 준비(selectfrommodel 적용한 것을 사용하기 위해 transform 해 줘야 한다)
    selection_x_train_0 = selection_0.transform(x_train)
    selection_x_train_1 = selection_1.transform(x_train)
    selection_x_train_2 = selection_2.transform(x_train)
    selection_x_train_3 = selection_3.transform(x_train)

    # test도 transform
    selection_x_test_0 = selection_0.transform(x_test)
    selection_x_test_1 = selection_1.transform(x_test)
    selection_x_test_2 = selection_2.transform(x_test)
    selection_x_test_3 = selection_3.transform(x_test)

    # 예측 하기 위해 사용할 test data도 transform
    test_0 = selection_0.transform(test)
    test_1 = selection_1.transform(test)
    test_2 = selection_2.transform(test)
    test_3 = selection_3.transform(test)

    # 훈련
    # 준비된 train, test 데이터들을 최종 모델에 fit 해 준다
    selection_model_0.fit(selection_x_train_0, y_train_0, verbose = 0, eval_metric = ['mae'], eval_set = [(selection_x_train_0, y_train_0, selection_x_test_0, y_test_0)], early_stopping_rounds = 10)
    selection_model_1.fit(selection_x_train_1, y_train_1, verbose = 0, eval_metric = ['mae'], eval_set = [(selection_x_train_1, y_train_1, selection_x_test_1, y_test_1)], early_stopping_rounds = 10)
    selection_model_2.fit(selection_x_train_2, y_train_2, verbose = 0, eval_metric = ['mae'], eval_set = [(selection_x_train_2, y_train_2, selection_x_test_2, y_test_2)], early_stopping_rounds = 10)
    selection_model_3.fit(selection_x_train_3, y_train_3, verbose = 0, eval_metric = ['mae'], eval_set = [(selection_x_train_3, y_train_3, selection_x_test_3, y_test_3)], early_stopping_rounds = 10)
    
    # 예측


'''







#     print('thresholds_0 :', thresholds_0)
#     print(thresholds_0[0])
#     print('thresholds_1 :', thresholds_1)
#     print(thresholds_1[38])






    