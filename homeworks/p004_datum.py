'''
# 핵심 인물 찾기

# 딕셔너리 형태로 구성된 사용자 명단
# 각 사용자의 숫자로 된 고유 번호인 id와 이름을 나타내는 nmae으로 구성
users = [
    {"id": 0, "name" : "Hero"},
    {"id": 1, "name" : "Hero"},
    {"id": 2, "name" : "Sue"},
    {"id": 3, "name" : "Chi"},
    {"id": 4, "name" : "Thor"},
    {"id": 5, "name" : "Clive"},
    {"id": 6, "name" : "Hicks"},
    {"id": 7, "name" : "Devin"},
    {"id": 8, "name" : "Kate"},
    {"id": 9, "name" : "Klein"},
]

# id의 쌍으로 구성된 친구 관계 data인 friendship_pairs도 있다


friendship_pairs = [
    (0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
    (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)
]

# 친구 관계를 위와 같은 쌍의 list로 표현하는 것이 이 데이터를 다루는 가장 쉬운 방법은 아님
# 가령 id가 1인 사용자의 모든 친구르 찾으려면 모든 쌍을 순회하여 1이 포함되어 있는 쌍을 구해야 함
# 만약 엄청나게 많은 쌍이 주어졌다면 특정 사용자의 모든 친구를 찾기 위해 굉장히 오랜 시간이 걸릴 것
# 대신! 사용자의 id를 키(key)로 사용하고 해당 사용자의 모든 친구 목록을 값(value)으로 구성한 딕셔너리를 생성해보자

# 사용자의 친구 목록을 딕셔너리로 생성하기 위해서는 여전히 모든 쌍을 탐색해야 하지만
# 처음에 단 한번만 탐색하면 이후에는 훨씬 빠르게 각 사용자별 친구 목록을 찾아볼 수 있다

# 사용자별로 비어 있는 친구 목록 리스트를 지정하여 딕셔너리를 초기화
friendships = {user["id"]: [] for user in users}

#friendship_pairs 안의 쌍을 차례대로 살펴보면서 딕셔너리 안에 추가
for i, j in friendship_pairs:
    friendships[i].append(j)
    friendships[j].append(i)

# 이렇게 각 사용자의 친구 목록을 딕셔너리로 만들면 '네트워크상에서 각 사용자의 평균 연결 수는 몇 개인가?'
# 와 같이 네트워크 특성에 관한 질문에 답할 수 있다
# 이 질문에 답하기 위해 먼저 friendships 안 모든 리스트의 길이를 더해서 총 연결 수를 구해보자

def number_of_friends(user):
    """user의 친구는 몇 명일까?"""
    user_id = user["id"]
    friend_ids = friendships[user_id]
    return len(friend_ids)

total_connections = sum(number_of_friends(user) for user in users) # 24

num_users = len(users)                           # 총 사용자 리스트의 길이
avg_coonections = total_connections / num_users  # 24 / 10 == 2.4

# (user_id, number_of_friends)로 구성된 리스트 생성
num_friends_by_id = [(user["id"], number_of_friends(user)) for user in  users]


# 다음으로 연결 수가 가장 많은 사람, 즉 친구가 가장 많은 사람이 누군지 알아보자
# 친구가 제일 많은 사람부터 적은 사람 순으로 사용자 정리

# (user_id, number_of_friends)로 구성된 리스트 생성
num_friends_by_id.sort(                                           # 정렬하기  
    key=lambda id_and_friends : id_and_friends[1],                # num_friends 기준으로
    reverse = True                                                # 제일 큰 숫자부터 제일 작은 숫자순으롱
)                           

# 데이터 과학자 추천하기
# 친구 추천 기능 설계
# 사용자에게 친구의 친구를 소개
# 각 사용자의 친구에 대해 그 친구의 친구들을 살펴보고, 사용자의 모든 친구에 대해 똑같은 작업을 반복하고 결과를 저장

def foaf_ids_bad(user): # "foaf"는 친구의 친구("friends of friend"를 의미하는 약자)
    return [foaf_id
            for friend_id in friendships[user["id"]]
            for foaf_id in friendships[friend_id]]

# user[0], Hero에게 함수 실행
hero = foaf_ids_bad(users[0])
print(hero)                   # [0, 2, 3, 0, 1, 3]
# Hero도 자신의 친구의 친구이므로 사용자 0(자기 자신)이 두 번 포함 되어 있음
# 그리고 이미 Hero와 친구인 사용자 1과 사용자 2도 포함되어 있음
# 사용자 3인 Chi는 두 명의 친구와 친구이기 때문에 두 번 포함

print(friendships[0])  # [1, 2]
print(friendships[1])  # [0, 2, 3]
print(friendships[2])  # [0, 1, 3]

# 서로의 함께 아는 친구(mutual friends)가 몇 명일까?
# 동시에 사용자가 이미 아는 사람을 제외하는 함수 만들기

from collections import Counter # 별도로 import 필요

def friends_of_friends(user):
    user_id = user["id"]
    return Counter(
        foaf_ids_bad
        for friend_id in friendships[user_id]       # 사용자의 친구 개개인에 대해
        for foaf_id in friendships[friend_id]       # 그들의 친구들을 세어보고
        if foaf_id != user_id                       # 사용자 자신과
        and foaf_id not in friendships[user_id]     # 사용자의 친구는 제외
    )

print(friends_of_friends(users[3]))                 # Counter({0: 2, 5: 1})
# Chi(id:3)는 Hero(id:0)와 함께 아는 친구 2명, Clive(id:5)와 함께 아는 친구는 1명

# 비슷한 관심사를 가진 사람 소개하기
# 관심사 데이터 interests를 받았음
# 사용자 고유번호 user_id, 관심사 interest의 쌍(user_id, interest)로 구성

interests = [
    (0, "Hadoop"), (0, "Big Data"), (0, "HBase"), (0, "Java"), (0, "Spark"), (0, "Storm"), (0, "Cassandra"),
    (1, "NoSQL"), (1, "MongoDB"), (1, "Cassandra"), (1, "HBase"), (1, "Postgres"), 
    (2, "Python"), (2, "scikit-leran"), (2, "scipy"), (2, "numpy"), (2, "statsmodelsn"), (2, "pandas"), 
    (3, "R"), (3, "Python"), (3, "statistics"), (3, "regression"), (3, "probability"), 
    (4, "mechine learning"), (4, "regression"), (4, "decision tress"), (4, "libsvm"), 
    (5, "Python"), (5, "R"), (5, "Java"), (5, "C++"), (5, "Haskell"), (5, "programming languages"),
    (6, "statistics"),  (6, "probability"),  (6, "mathematics"),  (6, "theory"),
    (7, "machine learning"), (7, "scikit-leran"), (7, "Mahout"), (7, "neural networks"),
    (8, "Big Data"), (8, "artificial intelligence"), 
    (9, "Hadoop"), (9, "Java"), (9, "MapReduce"), (9, "Big Data")
]

# 특정 관심사를 갖고 있는 모든 사용자 id 반환
def data_scientists_who_like(target_interest):
    return [user_id for user_id, user_interest in interests 
            if user_interest == target_interest]
# 이 코드는 호출할 때마다 관심사 데이터를 매번 처음부터 끝까지 훑어야 한다는 단점이 있다
# 사용자 수가 많고 그들의 관심사가 많다면(또는 데이터를 여러 번 훑을 거라면)
# 각 관심사로 사용자 인덱스(index)를 만드는 것이 나을지도,,,

from collections import defaultdict

# key : interest, value :user_id

user_ids_by_interest = defaultdict(list)

for user_id, interest in interests:
    user_ids_by_interest[interest].append(user_id)

# 더불어 각 사용자에 관한 관심사 인덱스도 만들어주기
# key : user_id, value :interest

interests_by_user_id = defaultdict(list)

for user_id, interest in interests:
    interests_by_user_id[user_id].append(interest)

# 이제 특정 사용자가 주어졌을 때, 사용자와 가장 유사한 관심사를 가진 사람이 누구인지
# 다음의 3단계로 알 수 있다
# 1. 해당 사용자의 관심사들을 훑는다
# 2. 각 관심사를 가진 다른 사용자들이 누구인지 찾아본다
# 3. 다른 사용자들이 몇 번이나 등장하는지 센다
# 위의 과정들을 다음과 같은 코드로 구현

def most_common_interests_with(user):
    return Counter(
        interested_user_id
        for interest in interests_by_user_id[user["id"]]
        for interested_user_id in user_ids_by_interest[interest]
        if interested_user_id != user["id"]
    )


# 연봉과 경력
# 연봉은 민감한 데이터, 데이터를 익명화
# 데이터에는 각 사용자의 연봉이 달러로, 근속 기간(tenure)이 연 단위로 표기

salaries_and_tenures = [
    (83000, 8.7), (88000, 8.1), (48000, 0.7), (76000, 6),
    (69000, 6.5), (76000, 7.5), (60000, 2.5), (83000, 10),
    (48000, 1.9), (63000, 4.2)
]

# 근속 연수에 따라 평균 연봉이 어떻게 달라질까?

# key : 근속 연수, value : 근속 연수에 대한 연봉 목록
salary_by_tenure = defaultdict(list)

for salary, tenure in salaries_and_tenures:
    salary_by_tenure[tenure].append(salary)

# key : 근속 연수, value : 해당 근속 연수의 평균 목록

average_salary_by_tenure = {
    tenure : sum(salaries) / len(salaries)
    for tenure, slaries in salary_by_tenure.items()
}

# but, 근속 연수가 같은 사람이 한 명도 없어서 결과가 쓸모 있어 보이지 않음
# 사용자 개개인의 연봉을 보여주는 것과 다르지 않기 때문

# 아래와 같이 경력을 몇 개의 구간으로 나누자

def tenure_bucket(tenure):
    if tenure < 2:
        return "less than two"
    elif tenure < 5:
        return "between two and five"
    else:
        return "more than five"

# 각 연봉을 해당 구간에 대응
# key : 근속 연수 구간, value : 해당 구간에 속하는 사용자들의 연봉
salary_by_tenure_bucket = defaultdict(list)

for salary, tenure in salaries_and_tenures:
    bucket = tenure_bucket(tenure)
    salary_by_tenure_bucket[bucket].append(salary)

# 각 구간의 평균 연봉 구하기
# key : 근속 연속구간, value : 해당 구간에 속하는 사용자들의 평균 연봉
average_salary_by_bucket = {
    tenure_bucket: sum(salaries) / len(salaries)
    for tenure_bucket, salaries in salary_by_tenure_bucket.items()
} 

# 유료 계정
# 어떤 사용자들이 유료 계정으로 전환하는지 파악
# 0.7 paid
# 1.9 unpaid
# 2.5 paid
# 4.2 unpaid
# 6.0 unpaid
# 6.5 unpaid
# 7.5 unpaid
# 8.1 unpiad
# 8.7 paid
# 10.0 paid


# 데이터를 살펴보니 서비스를 이요한 기간이 유료 계정 사용 여부와 상관 있어 보임
# 서비스 이용 기간이 매우 짧거나 아주  긴 경우에는 유료 계정 사용하는 경향
# 기간이 평균치 내외인 경우에는 그렇지 않은 듯
# 비록 데이터가 부족하기는 해도 기간에 따라 유료 계정 사용 여부를 예측할 수 있는 간단한 모델 만들기 가능

def predict_paid_or_unpaid(yesrs_experience):
    if years_experiece < 3.0:
        return "paid"
    elif years_experience < 8.5:
        return "unpiad"
    else:
        return "paid"

# 여기서 분류의 기준이 되는 임계치들은 대충 감으로 정한 것
# 더 많은 데이터가 있다면 (더불어 약간의 수학을 가미한다면)
# 사용자의 서비스 이용 기간에 따라 사용자가 유료 계정으로 전환할 가능성을 계산할 수 있게 됨
# 이런 종류의 문제는 로지스틱 회귀 분석에서 다룸

# 관심 주제
# 사용자들이 주로 어떤 관심사를 가지고 있는가?
# 간단하게 단어의 갯수를 세서 가장 인기가 많은 관심사를 찾을 수 있음

words_and_counts = Counter(word
                            for user, interest in interests
                            for word in interest.lower().split())

# 이 중에서 한 번을 초과해서 등장하는 단어들만 출력하면
for word, count in words_and_coutns.most_common():
    if count > 1:
        print(word, count)

# 원하는 결과를 얻을 수 있다 
# scikit-learn 이 두 개의 단어로 쪼개지길 원했다면 위의 방법은 조금 아쉬움

'''
