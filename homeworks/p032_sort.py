# 2.16 정렬

# 파이썬의 모든 리스트에는 리스트를 자동으로 정렬해주는 sort method가 있다
# 만약 이미 만든 리스트를 망치고 싶지 않다면 sorted 함수를 통해 새롭게 정렬된 리스트를 생성할 수 있다

x = [4, 1, 2, 3]
y = sorted(x) # y는 [1, 2, 3, 4], but x는 변하지 않음
x.sort() # 이제는 x도 [1, 2, 3, 4]로 재정렬 됨

# 기본적으로 sort method와 sorted 함수는 리스트의 각 항목을 일일이 비교해서 오름차순으로 정렬해준다
# 만약 리스트를 내림차순으로 정렬하고 시다면 인자에 revese=True를 추가!
# 이 말은 default는 오름차순?!
# 그리고 리스트의 각 항목끼리 서로 비교하는 대신 key를 사용하면 지정한 함수의 결괏값을 기준으로 리스트를 정렬할 수 있다

# 절댓값의 내림차순으로 리스트를 정렬
x = sorted([-4, 1, -2, 3], key=abs, reverse = True)
print(x) # [-4, 3, -2, 1]

# 빈도의 내림차순으로 단어와 빈도를 정렬
# wc = sorted(word_counts.items(), 
#            key=lambda word_and_count: word_and_count[1],
#            reverse = True)