# # web 상에서 데이터 수정을 하기 위한 소스

# from flask import Flask, render_template, request
# import sqlite3

# app = Flask(__name__)

# # 데이터 베이스
# conn = sqlite3.connect("./data/wanggun.db")
# cursor = conn.cursor()
# cursor.execute("SELECT * FROM general;") # sql의 문법을 대문자로 적는 것이 암묵적 규칙
# print(cursor.fetchall()) # 전체 출력

# @app.route('/')
# def run():
#     conn = sqlite3.connect('./data/wanggun.db')
#     c = conn.cursor()
#     c.execute("SELECT * FROM general;")
#     rows = c.fetchall();
#     return render_template("board_index.html", rows=rows)


# @app.route('/modi')
# def modi():
#     ids = request.args.get('id')
#     conn = sqlite3.connect('./data/wanggun.db')
#     c = conn.cursor()
#     c.execute('SELECT * FROM general WHERE id = ' + str(ids))
#     rows = c.fetchall()
#     return render_template("board_modi.html", rows=rows)


# @app.route('/addrec', methods=['POST', 'GET'])
# def addrec() :
#     if request.method =='POST':
#         try:
#             conn = sqlite3.connect('./data/wanggun/db')
#             war = request.form['war']
#             ids = request.form['id']
#             c = conn.cursor()
#             c.execute('UPDATE general SET war = '+ str(war) + ' WHERE id = '+str(ids))
#             conn.commit()
#             msg = '정상적으로 입력되었습니다.'
#         except : # try 부분이 실행함에 있어서 에러가 발생한다면 그에 대한 대응, 처리
#             conn.rollback() # rollback : 원래대로 돌린다
#             msg = '입력과정에서 에러가 발생하였습니다'
#         finally :
#             conn.close()
#             return render_template("board_result.html", msg=msg)
# app.run(host = '127.0.0.1', port = 5111, debug = False)





from flask import Flask, render_template, request
import sqlite3

app = Flask(__name__)

# 데이터베이스 만들기
conn = sqlite3.connect("./data/wanggun.db")
cursor = conn.cursor()
cursor.execute("SELECT * FROM general;")
print(cursor.fetchall())

@app.route('/')
def run():
    conn = sqlite3.connect('./data/wanggun.db')
    c = conn.cursor()
    c.execute("SELECT * FROM general;")
    rows = c.fetchall()
    return render_template("board_index.html", rows=rows)

@app.route('/modi')
def modi():
    ids = request.args.get('id')
    conn = sqlite3.connect('./data/wanggun.db')
    c = conn.cursor()
    c.execute('SELECT * FROM general where id = ' + str(ids))
    rows = c.fetchall()
    return render_template('board_modi.html', rows=rows)

@app.route('/addrec', methods=['POST', 'GET'])
def addrec():
    if request.method == 'POST':
        try:
            conn = sqlite3.connect('./data/wanggun.db')
            war = request.form['war']
            ids = request.form['id']
            c = conn.cursor()
            c.execute('UPDATE general SET war = '+ str(war) + " WHERE id = "+str(ids))
            conn.commit()
            msg = '정상적으로 입력되었습니다.'
        except:
            conn.rollback()
            msg = '에러가 발생하였습니다.'
        finally:
            conn.close()
            return render_template("board_result.html", msg=msg)

app.run(host='127.0.0.1', port=5001, debug=False)



