# range 함수(Class)
a = range(10)
print(a)                                # 출력 : range(0, 10)

b = range(1, 11)
print(b)

for i in a:
    print(i)                            # 0 1 2 3 ... 9 까지 출력

for i in b:                             
    print(i)                            # 1 2 3 ... 10 까지 출력

print(type(a))                          # 출력 : <class 'range'>

sum = 0
for i in range(1, 11):
    sum = sum + i
print(sum)                              # 출력 : 55 1부터 10까지 더한 값


sum = 0
for i in range(1, 11):
    sum = sum + i
    print(sum)                          # 출력 : 1, 3, 6, 10, ...., 55



