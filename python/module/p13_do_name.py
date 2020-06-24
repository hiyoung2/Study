import p11_car
import p12_tv  
# import로 가져온 거에서는
# import 된 파일들에서 적어놓은 __name__ 이
# 다른 파일(현재 이 파일)에서는 main이 아니라 import 된 파일명이 뜬다


print("====================================")
print("do.py의 module 이름은 ", __name__) # 실행 시킨 파일, 즉 ctrl+F5를 한 파일의 이름이 뜬다
print("====================================")

p11_car.drive() 
p12_tv.watch()

'''
운전하다
car.py의 module 이름은  p11_car
시청하다
tv.py의 module 이름은  p12_tv
====================================
do.py의 module 이름은  __main__
====================================
운전하다
시청하다
'''