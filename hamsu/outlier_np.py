import numpy as np

# 이상치의 위치를 알려주는 함수
def outliers_loc(data_out):
    outliers = []
    for i in range(data_out.shape[1]):
        data = data_out[:, i] 
        quartile_1, quartile_3 = np.percentile(data, [25, 75]) 
        print("1사분위 : ",quartile_1)                                       
        print("3사분위 : ",quartile_3)                                        
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        out = np.where((data_out > upper_bound) | (data_out < lower_bound))
        outliers.append(out)
    return outliers

# 이상치 값이 무엇인지 알려주는 함수
def outliers_np_idx(data_out):
    outliers = []
    for i in range(data_out.shape[1]):
        data = data_out[:, i]
        quartile_1, quartile_3 = np.percentile(data, [25, 75])
        print("1사 분위 : ",quartile_1)                                       
        print("3사 분위 : ",quartile_3)                                        
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        outliers.append(list(filter(lambda x: (x > upper_bound) | (x < lower_bound), data )))
    return outliers