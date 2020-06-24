import p31_sample

x = 222 # 변수를 사용

def main_func() :
    print('x :', x)


# import 된 파일이 있으므로 실행이 됨
p31_sample.test() # x : 111

# 현재 위의 main_func이 실행 됨
main_func() # x : 222