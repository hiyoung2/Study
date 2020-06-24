def outliers(data_out):
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