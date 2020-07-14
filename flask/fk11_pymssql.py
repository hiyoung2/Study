import pymssql as ms # pymssql 설치
# print("잘 접속 됐나?") # 출력 잘 됨

# ms에 접속을 하겠다 : ms.connect() 
# # 내 주소 : 127.0.0.1, 사용자명, 패스워드, 데이터베이스 차례로 넣어주고 conn 이라는 변수에 대입
conn = ms.connect(server = '127.0.0.1', user = 'bit2', 
                  password = '1234', database = 'bitdb')

# cursor = 지정한다
cursor = conn.cursor()

# cursor를 excute, 실행하겠다
# cursor.execute("SELECT * FROM iris2;") # SQL 문법으로 iris data를 탐색
# cursor.execute("SELECT * FROM sonar;") # SQL 문법으로 iris data를 탐색
cursor.execute("SELECT * FROM wine;") # SQL 문법으로 iris data를 탐색

# cursor에 있는 것을 한 줄씩 생성(fetchone)해서 row에 대입
row = cursor.fetchone() # 150줄이면 복붙해서 150개 만들기? nope
# 반복문을 사용!
while row :
    print("first column : %s, second column : %s" %(row[0], row[1])) # 첫 번째, 두 번째 컬럽을 가져옴
    row = cursor.fetchone()

# connect 하고 작업이 끝나면 끊어줘야 함 (session.close()와 같이)
conn.close() 