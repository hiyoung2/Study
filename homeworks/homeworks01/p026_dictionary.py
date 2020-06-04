# 2.11 딕셔너리

# 딕셔너리(dict, dictionary, 사전)는 파이썬의 또 다른 기본적인 데이터 구조이며
# 특정 값(value)과 연관된 키(key)를 연결해주고
# 이를 사용해 값을 빠르게 검색할 수 있다

empty_dict = {} # 가장 파이썬스럽게 딕셔너리를 만드는 방법
empty_dict2 = dict() # 덜 파이썬스럽게 딕셔너리를 만드는 방법 ,,,, 파이썬스럽게 되게 좋아하네

# {key : value} : key는 호출, value는 값
grades = {"Joel" : 80, "Tim" : 95} # 딕셔너리 예시 

# 대괄호를 사용해 키의 값을 불러올 수 있다
joels_grade = grades["Joel"]

print(grades["Joel"]) # 80 
print(joels_grade) # 80

# 만약 딕셔너리에 존재하지 않는 키를 입력하면? KeyError 발생!
try:
    kates_grade = grades["Kate"]
except KeyError:
    print("no grade for Kate!")
# no grade for Kate! 출력

# 연산자 in을 사용하면 키의 존재 여부 확인 가능

joel_has_grade = "Joel" in grades
print("Joel" in grades) # True
Kate_has_grade = "Kate" in grades
print("Kate" in grades) # False

# 크기가 굉장히 큰 딕셔너리에서도 키의 존재 여부를 빠르게 확인 할 수 있다
# 딕셔너리에서 get 메서드(method)를 사용하면 입력한 키가 딕셔너리에 없어도
# 에러를 반환하지 않고 기본값을 반환해 준다
# method는 class에 속한 함수이다

joels_grade = grades.get("Joel", 0)
print(joels_grade) # 80
kates_grade = grades.get("Kate", 0)
print(kates_grade) # 0
no_ones_grade = grades.get("No One")
print(no_ones_grade) # 기본값으로 None을 반환

# 또한 대괄호를 사용해서 키와 값을 새로 지정해 줄 수 있다
grades["Tim"] = 99 # 기존의 값을 대체
print(grades) # {'Joel': 80, 'Tim': 99}
grades["Kate"] = 100 # 드디어 케이트가 포함 되었다 ㅋㅋ
print(grades) # {'Joel': 80, 'Tim': 99, 'Kate': 100}

num_students = len(grades)
print(num_students) # 3

# 1장, '들어가기'에서 봤다시피 정형화된 데이터를 간단하게 나타낼 때는 주로 딕셔너리가 사용된다
tweet = {
    "user" : "joelgrus",
    "text" : "Data Science is Awesome",
    "retweet_count" : "100",
    "hashtags" : ["#data", "#science", "#datascience", "#awesome", "#yolo"]
}

print(tweet)

# 특정 키 대신 딕셔너리의 모든 키를 한 번에 살펴볼 수 있다

tweet_keys = tweet.keys()      # key에 대한 리스트
print(tweet_keys) 
# 출력 : dict_keys(['user', 'text', 'retweet_count', 'hashtags'])
tweet_values = tweet.values()  # value에 대한 리스트
tweet_items = tweet.items()    # (key, value) 튜플에 대한 리스트

# 딕셔너리의 key는 수정할 수 없으며 리스트를 키로 사용할 수 없다
# 만약 다양한 값으로 구성된 key가 필요하다면 tuple이나 string을 key로 사용하도록!

