# 리스트 안의 리스트 각 인덱스의 요소를 한 번에 삭제하기
import numpy as np

a = [[1,2,3,4],[1,2,3,4],[1,2,3,4]]

# 여기서 각 리스트의 인덱스0의 자리인 1만 삭제하자

# delete column at index 1
a = np.delete(a, 1, axis = 1)
print(a)

''' 
출력
[[1 3 4]
 [1 3 4]
 [1 3 4]]
'''