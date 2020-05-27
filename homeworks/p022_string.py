# 2. 7 문자열
# 문자열, string은 작은 따옴표 '', 큰 따옴표 ""로 묶어 나타냄 (다만 앞뒤로 동일한 기호 사용해야함)

single_quoted_string = 'data science'
souble_quoted_string = "data science"

# 파이썬은 몇몇 특수 문자를 인코딩할 때 역슬래시를 사용
tab_string = "\t"  # tab을 의미하는 문자열
len(tab_string)    
print(len(tab_string)) # 1

# 만약 역슬래시를 역슬래시로 보이는 문자로 사용하고 싶다면
# (특히 윈도우 directory 이름이나 정규표현식에서 사용하고 싶을 때)
# 문자열 앞에 r을 붙여 raw string(가공되지 않은 문자열)이라고 명시하면 도니다

not_tab_string = r"\t"    # 문자 '\'와 't'를 나타내는 문자열
len(not_tab_string)
print(len(not_tab_string)) # 2

multi_line_string = """This is the first line.
and this is the second line
and this is the third line"""
print(multi_line_string)

# This is the first line.
# and this is the second line
# and this is the third line

# 파이썬 3.6부터는 문자열 안의 값을 손쉽게 추가할 수 있는 f-string 기능이 추가 되었다
# 가령, 다음과 같이 각각 다른 변수로 주어진 성과 이름을 합쳐서

first_name = "inyoung"
last_name = "ha"

# 전체 이름을 의미하는 full_name 변수를 만들 수 있는 방법은 다양
full_name1 = first_name + " " + last_name # 문자열 합치기
full_name2 = "{0} {1}".format(first_name, last_name) # .format을 통한 문자열 합치기

# 하지만 f-string을 사용하면 훨씬 간편하게 두 문자열을 합칠 수 있다
full_name3 = f"{first_name} {last_name}"

print(full_name1)
print(full_name2)
print(full_name3)