import sys # 시스템을 import

print(sys.path) # 현재 작업 공간이랑 ???

from test_import import p62_import
p62_import.sum2()

print("==============================")

from test_import.p62_import import sum2
sum2()

'''
이 import는 아나콘다 폴더에 들어있습니다!
작업그룹 임포트 썸탄다
==============================
작업그룹 임포트 썸탄다
'''