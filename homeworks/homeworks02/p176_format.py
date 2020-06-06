# 6.4 문자열 포맷 지정

# 문자열형 메서드(format) 에서 foramt() 메서드를 사용하여 문자열의 포맷(형식)을 지정했다
# 파이썬에서 포맷을 지정하는 방법에는 여러 가지가 있다

# % 연산자를 사용하는 방법을 알아보자
# 큰따옴표 혹은 작은따옴표로 둘러싸인 문자열에 %를 기술하여 문자열 뒤에 나열된 객체를 넘겨줄 수 있다

# %d : 정수로 표시
# %f : 소수로 표시
# %.2f : 소수점 이하 두 자리까지 표시
# %s : 문자열로 표시

pai = 3.14592
print("원주율은 %f" % pai)   # 원주율은 3.145920
print("원주율은 %.2f" % pai) # 원주율은 3.15 (반올림 해서 나옴)

# 문제 
def bmi(height, weight) :
    return weight / height**2

print("bmi는 %.4f 입니다" % bmi(183, 72))

# 연습문제
# object 중에서 character를 포함한 요소 수를 세는 함수를 작성
# 인수 object, character를 취하는 함수 check_character를 작성
# count() 메서드로 문자열과 리스트 안의 요소 수를 반환
# ex) check_character([1, 2, 4, 5, 5, 3], 5) # 출력 결과 : 2
# 함수 check_character에 '임의의 문자열(또는 리스트)'과 '개수를 조사할 요소'를 입력


def check_character(object, character) :
    return object.count(character)

print(check_character([1, 2, 4, 5, 5, 3], 5)) # 2
print(check_character(("lovewillfindaway"), "l")) # 3

# 종합문제
# 이진 검색 알고리즘을 이용하여 검색하는 프로그램을 만들자
# 알고리즘은 문제를 푸는 절차이다
# 검색 데이터가 커질수록 선형 검색 알고리즘(맨 앞부터 끝까지 차례대로 찾는 방법)에 비해
# 검색 시간이 압도적으로 짧다
# 이진 검색 알고리즘은 다음과 같다
# * 데이터의 중앙값을 구한다
# * 중앙값이 찾는 값과 일치하는 경우 종료
# * 찾는 값이 중앙값보다 크면 탐색 범위의 최솟값을 중앙값에 1을 더한 값으로 변경하고
#   찾는 값이 중앙값보다 작으면 탐색 범위의 최댓값을 중앙값에서 1을 뺀 값으로 변경

# 문제
# 함수 binary_search에 이진 검색 알고리즘을 사용, 리스트 numbers에서 target_number를 찾아내는 프로그램 작성
# 함수를 실행했을 때 '11은(는) 10번째에 있습니다.' 라고 출력
# 변수 target_number를 변경하고 자신의 프로그램이 제대로 동작하는지 확인

def binary_search(numbers, target_number) :
    # 최솟값을 임시로 결정
    low = 0
    # 범위 내의 최댓값
    high = len(numbers)
    # 목적지를 찾을 때까지 루프
    while low <= high :
        # 중앙값 구하기
        middle = (low + high) // 2
        # numbers의 중앙값과 target_number가 같은 경우
        if numbers[middle] == target_number :
            print("{1}은(는) {0}번째에 있습니다.".format(middle, target_number))
            break
        elif numbers[middle] < target_number :
            low = middle + 1
        else :
            high = middle -1

numbers = [1, 2, 3, 4, 5, 6 , 7 , 8, 9, 10, 11, 12, 13]

target_number = 11

binary_search(numbers, target_number)