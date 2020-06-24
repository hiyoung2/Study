import numpy as np
import pandas as pd

# 이상치의 위치를 알려주는 함수

def outliers_loc(data_out):
    outliers = []
    for i in range(len(data_out.columns)):
        data = data_out.iloc[:, i]
        quartile_1 = data.quantile(.25)
        quartile_3 = data.quantile(.75)
        print("1사 분위 : ",quartile_1)                                       
        print("3사 분위 : ",quartile_3)                                        
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        out = np.where((data > upper_bound) | (data < lower_bound))
        outliers.append(out)
    return outliers

# 이상치 값이 무엇인지 알려주는 함수

def outliers_pd_idx(data_out):
    outliers = []
    for i in range(len(data_out.columns)):
        data = data_out.iloc[:, i]
        quartile_1 = data.quantile(.25)
        quartile_3 = data.quantile(.75)
        print("1사 분위 : ",quartile_1)                                       
        print("3사 분위 : ",quartile_3)                                        
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        outliers.append(list(filter(lambda x: (x > upper_bound) | (x < lower_bound), data )))
    return outliers