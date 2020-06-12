# 14.2 DataFrame 복습
# DataFrame은 9.2절 'DataFrame 연결'에서 이미 설명했다
# 여기서는 간단한 문제로 복습

import pandas as pd
from pandas import Series, DataFrame

attri_data1 = {"ID" : ["100", "101", "102", "103", "104", "106", "108", 
                      "110", "111", "113"],
               "city" : ["서울", "부산", "대전", "광주", "서울", "서울", "부산",
                         "대전", "광주", "서울"],
               "birth_year" : [1990, 1989, 1992, 1997, 1982, 1991, 1988, 
                              1990, 1995, 1981],
               "name" : ["영이", "순돌", "짱구", "태양", "션", "유리", "현아",
                         "태식", "민수", "호식"]}

attri_data_frame1 = DataFrame(attri_data1)
print(attri_data_frame1)
print()

attri_data2 = {"ID" : ["107", "109"],
               "city" : ["봉화", "전주"],
               "birth_year" : [1994, 1988]}

attri_data_frame2 = DataFrame(attri_data2)
print(attri_data_frame2)
print()

attri_data_frame1.append(attri_data_frame2).sort_values(by = "ID", ascending = True).reset_index(drop = True)

print("append :", attri_data_frame1)

