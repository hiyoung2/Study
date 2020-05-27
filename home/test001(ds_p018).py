
for i in [1, 2, 3, 4, 5]: 
    print(i)                 # 'for i' 단락의 첫 번째 줄
    for j in [1, 2, 3, 4, 5]:
        print(j)             # 'for j' 단락의 첫 번째 줄
        print(i + j)         # 'for j' 단락의 마지막 줄
    print(i)                 # 'for i' 단락의 마지막 줄
print("done looping")

'''
출력
1
1
2
2
3
3
4
4
5
5
6
1
2
1
3
2
4
3
5
4
6
5
7
2
3
1
4
2
5
3
6
4
7
5
8
3
4
1
5
2
6
3
7
4
8
5
9
4
5
1
6
2
7
3
8
4
9
5
10
5
'''