# 2. 12 Counter

# Counter는 연속된 값을 defaultdict(int)와 유사한 객체로 변환해주며
# 키와 값의 빈도를 연결시켜 준다

from collections import Counter
c = Counter([0, 1, 2, 0])
print(c) # Counter({0: 2, 1: 1, 2: 1})

# 게다가 특정 문서에서 단어의 갯수를 셀 때도 유용하다

# document가 단어의 list임을 상기하자
document = ["look", "in", "my", "eyes", "Can", "you", "see", "my", "heartbeat", 
"swirling", "everywhere", "i", "go", "look", "in", "my" "eyes", "i", "will", "be",
"next", "to", "you", "you", "let", "me" , "be", "on", "the", "line", "step", "ste", 
"where", "do", "we", "go", "now", "it", "feels", "like", "lost", "in", "a", "wrong",
"maze", "step", "step", "here", "we", "are", "half", "way", "home", "ane", "we", "will",
"never", "give", "up", "hus", "you", "can", "find", "your", "stars", "someday", "in", 
"your", "dream", "yes", "i", "know", "it's", "not", "so", "easy", "yeah", "but","don't", 
"you", "ever", "let", "go", "i", "know", "it's", "all", "or", "nothing", "but", "we", "will",
"make", "it", "all", "right", "oh", "so", "live", "your", "life", "shining", "like", "a",
"star", "and", "i", "will", "be", "your", "moonlight"]

word_counts = Counter(document)

# Counter 객체에는 자주 나오는 단어 10개와 이 단어들의 빈도수 출력
for word, count in word_counts.most_common(10):
    print(word, count)
##출력##
# you 5
# i 5
# in 4
# will 4
# we 4
# your 4
# go 3
# be 3
# step 3
# look 2

