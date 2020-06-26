import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts,GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score,mean_absolute_error as mae
# from xgboost import LGBMRegressor,plot_importance
from lightgbm import LGBMRegressor

for size in np.arange(0.91,0.911,0.01):
    #0~1 데이터 전처리, 데이터 입력

    train = pd.read_csv("./dacon/comp1_bio/csv/train.csv",index_col=0,header=0,encoding="cp949")
    test = pd.read_csv("./dacon/comp1_bio/csv/test.csv",index_col=0,header=0,encoding="cp949")
    submission = pd.read_csv("./dacon/comp1_bio/csv/sample_submission.csv",index_col=0,header=0,encoding="cp949")

    # for i in range(len()):
        
        
    # print(train.shape)#(10000, 75) train,test
    # print(test.shape)#(10000, 71) x_predict
    # print(submission.shape)#(10000, 4) y_predict


    # print(train.isnull().sum())
                                        
    # train = train.interpolate()#보간법 // 선형보간

    # print(dir(train))
    # print(train.isnull().sum())


    # print(train.info())
    # print(test.info())

    train=train.transpose()
    test=test.transpose()

    train=train.interpolate()
    test=test.interpolate() #보간법 // 선형보간

    train=train.transpose()
    test=test.transpose()

    
    # train= train.fillna(0)
    # train= train.fillna(0)

    # print(train.info())
    # print(test.info())

    test=test.values

    # print(train.shape)

    x = train.values[:,:-4]
    y = train.values[:,-4:]


    # print(x.shape)
    # print(y.shape)

    x_train, x_test, y_train, y_test = tts(x,y,train_size=size,random_state=66)

    # print(x_train.shape)
    # print(x_test.shape)
    # print(y_train.shape)
    # print(y_test.shape)

    #2.모델 구성-0
    # 33
    # 47
    # 49
    # 5

    
    xgb_0 = LGBMRegressor(n_estimators=200,learning_rate=0.2,max_depth= 7)
    xgb_1 = LGBMRegressor(n_estimators=200,learning_rate=0.1,max_depth= 4)
    xgb_2 = LGBMRegressor(n_estimators=200,learning_rate=0.12,colsample_bytree=0.8,colsample_bylevel=0.4,max_depth=3)
    xgb_3 = LGBMRegressor(n_estimators=200,learning_rate=0.13,colsample_bytree=0.8,colsample_bylevel=0.9,max_depth=4)

    y_train_0=y_train[:,0]
    y_train_1=y_train[:,1]
    y_train_2=y_train[:,2]
    y_train_3=y_train[:,3]

    y_test_0 = y_test[:,0]
    y_test_1 = y_test[:,1]
    y_test_2 = y_test[:,2]
    y_test_3 = y_test[:,3]
    
    parameters = {
    # "n_estimators" : np.arange(100,301,100),
    # "learning_rate" : np.arange(0.01,0.03,0.01),
    # "colsample_bytree":np.arange(0.6,1,0.1),
    # "colsample_bylevel":np.arange(0.6,1,0.1),
    # "max_depth" : [4,5,6]
    }
    xgb_0.fit(x_train,y_train_0,verbose=0,eval_metric=['mae'],eval_set=[(x_train,y_train_0,x_test,y_test_0)],early_stopping_rounds=30)
    xgb_1.fit(x_train,y_train_1,verbose=0,eval_metric=['mae'],eval_set=[(x_train,y_train_1,x_test,y_test_1)],early_stopping_rounds=30)
    xgb_2.fit(x_train,y_train_2,verbose=0,eval_metric=['mae'],eval_set=[(x_train,y_train_2,x_test,y_test_2)],early_stopping_rounds=30)
    xgb_3.fit(x_train,y_train_3,verbose=0,eval_metric=['mae'],eval_set=[(x_train,y_train_3,x_test,y_test_3)],early_stopping_rounds=30)
    
    # r2_0
    # -1
    # 0.8481549266897241

    # r2_1
    # 38
    # 0.34601582660837094

    # r2_2
    # 36
    # 0.3112930550861994

    # r2_3
    # 12
    # 0.2119125826057462

    thresholds_0 = np.sort(xgb_0.feature_importances_)
    thresholds_1 = np.sort(xgb_1.feature_importances_)
    thresholds_2 = np.sort(xgb_2.feature_importances_)
    thresholds_3 = np.sort(xgb_3.feature_importances_)

       
    selection_0=SelectFromModel(xgb_0,threshold=thresholds_0[0],prefit=True)#median +  GridSearch까지 할 것.데이콘 적용해라.
    selection_1=SelectFromModel(xgb_1,threshold=thresholds_1[38],prefit=True)#median +  GridSearch까지 할 것.데이콘 적용해라.
    selection_2=SelectFromModel(xgb_2,threshold=thresholds_2[36],prefit=True)#median +  GridSearch까지 할 것.데이콘 적용해라.
    selection_3=SelectFromModel(xgb_3,threshold=thresholds_3[12],prefit=True)#median +  GridSearch까지 할 것.데이콘 적용해라.
        
    selection_xgb_0 = LGBMRegressor(n_estimators=2000, learning_rate=0.2,max_depth= 7)
    selection_xgb_1 = LGBMRegressor(n_estimators=2000, learning_rate=0.1,max_depth= 4)
    selection_xgb_2 = LGBMRegressor(n_estimators=2000, learning_rate=0.12,colsample_bytree=0.8,colsample_bylevel=0.4,max_depth=3)
    selection_xgb_3 = LGBMRegressor(n_estimators=2000, learning_rate=0.13,colsample_bytree=0.8,colsample_bylevel=0.9,max_depth=4)
    
    selection_xgb_0 = GridSearchCV(selection_xgb_0,{},cv=5)
    selection_xgb_1 = GridSearchCV(selection_xgb_1,{},cv=5)
    selection_xgb_2 = GridSearchCV(selection_xgb_2,{},cv=5)
    selection_xgb_3 = GridSearchCV(selection_xgb_3,{},cv=5)
    
    selection_x_train_0 = selection_0.transform(x_train)
    selection_x_train_1 = selection_1.transform(x_train)
    selection_x_train_2 = selection_2.transform(x_train)
    selection_x_train_3 = selection_3.transform(x_train)

    selection_x_test_0 = selection_0.transform(x_test)
    selection_x_test_1 = selection_1.transform(x_test)
    selection_x_test_2 = selection_2.transform(x_test)
    selection_x_test_3 = selection_3.transform(x_test)

    test_0 = selection_0.transform(test)
    test_1 = selection_1.transform(test)
    test_2 = selection_2.transform(test)
    test_3 = selection_3.transform(test)
            
    selection_xgb_0.fit(selection_x_train_0,y_train_0,verbose=0,eval_metric=['mae'],eval_set=[(selection_x_train_0,y_train_0,selection_x_test_0,y_test_0)],early_stopping_rounds=20)
    selection_xgb_1.fit(selection_x_train_1,y_train_1,verbose=0,eval_metric=['mae'],eval_set=[(selection_x_train_1,y_train_1,selection_x_test_1,y_test_1)],early_stopping_rounds=20)
    selection_xgb_2.fit(selection_x_train_2,y_train_2,verbose=0,eval_metric=['mae'],eval_set=[(selection_x_train_2,y_train_2,selection_x_test_2,y_test_2)],early_stopping_rounds=20)
    selection_xgb_3.fit(selection_x_train_3,y_train_3,verbose=0,eval_metric=['mae'],eval_set=[(selection_x_train_3,y_train_3,selection_x_test_3,y_test_3)],early_stopping_rounds=20)
    
    y_pre_0=selection_xgb_0.predict(selection_x_test_0)
    y_pre_1=selection_xgb_1.predict(selection_x_test_1)
    y_pre_2=selection_xgb_2.predict(selection_x_test_2)
    y_pre_3=selection_xgb_3.predict(selection_x_test_3)

    r2_0= selection_xgb_0.score(selection_x_test_0,y_test_0)
    r2_1= selection_xgb_1.score(selection_x_test_1,y_test_1)
    r2_2= selection_xgb_2.score(selection_x_test_2,y_test_2)
    r2_3= selection_xgb_3.score(selection_x_test_3,y_test_3)

    mae_0= mae(y_test_0,y_pre_0)
    mae_1= mae(y_test_1,y_pre_1)
    mae_2= mae(y_test_2,y_pre_2)
    mae_3= mae(y_test_3,y_pre_3)

    # m=list(mae_0,mae_1,mae_2,mae_3)
    mae_result = (mae_0+mae_1+mae_2+mae_3)/4
    r2_result = (r2_0+r2_1+r2_2+r2_3)/4

    print(__file__)
    print(size)
    
    print(f"mae_0 : {mae_0}")
    print(f"mae_1 : {mae_1}")
    print(f"mae_2 : {mae_2}")
    print(f"mae_3 : {mae_3}")

    print(f"mae_result : {mae_result}")

    print(f"r2_0 : {r2_0}")
    print(f"r2_1 : {r2_1}")
    print(f"r2_2 : {r2_2}")
    print(f"r2_3 : {r2_3}")

    print(f"r2_result : {r2_result}")


    y_pre_0=selection_xgb_0.predict(test_0)
    y_pre_1=selection_xgb_1.predict(test_1)
    y_pre_2=selection_xgb_2.predict(test_2)
    y_pre_3=selection_xgb_3.predict(test_3)

    y_pre=[y_pre_0,y_pre_1,y_pre_2,y_pre_3]
    y_pre=np.array(y_pre)
    y_pre=y_pre.transpose()

    numbering=1
    submission = pd.DataFrame(y_pre,np.arange(10000,20000))
    path=__file__.split('\\')[-1][:-3]
    submission.to_csv(f"dacon\comp1_bio\csv\sample_submission_{path}_{numbering}_{size}.csv", index = True, header=['hhb','hbo2','ca','na'],index_label='id')

    submit = pd.DataFrame({})

# 0.7999999999999999
# mae_0 : 0.8393683517098367
# mae_1 : 0.6263995930744612
# mae_2 : 1.9697889218003912
# mae_3 : 1.2677638610924917
# mae_result : 1.1758301819192953
# r2_0 : 0.8512587841567868
# r2_1 : 0.38295273778728733
# r2_2 : 0.327317419449797
# r2_3 : 0.25651064121537137
# r2_result : 0.4545098956523106

# 0.8999999999999999
# mae_0 : 0.8635152189612388
# mae_1 : 0.6352934117078781
# mae_2 : 1.842055838499069
# mae_3 : 1.2479011308515073
# mae_result : 1.1471914000049233
# r2_0 : 0.8410615436630472
# r2_1 : 0.3693466710432993
# r2_2 : 0.37717609070339586
# r2_3 : 0.2658631704427781
# r2_result : 0.46336186896313014

# 0.91
# mae_0 : 0.8437401030513973
# mae_1 : 0.6066785793516372
# mae_2 : 1.8661167492018806
# mae_3 : 1.2152536158654426
# mae_result : 1.1329472618675895
# r2_0 : 0.8455843061642199
# r2_1 : 0.42587330328190665
# r2_2 : 0.3617241139523344
# r2_3 : 0.31227113664177586
# r2_result : 0.4863632150100592
