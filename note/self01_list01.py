# python list

a = [1, 2, 3, 4, 5]
print(a[:2])                     # [1, 2]
print(a[1:2])


import numpy as np

aaa = [[1,2,3],[4,5,6]]
print(type(aaa)) 
x = np.array(aaa)
print(x)


c = [1, 2, 3]
c.append([4])                        # data가 추가 되었을 때 쓸 수 있다, 정말 많이 쓴다
                            
print(c)                           # 출력 : [1, 2, 3, [4]]