# flask 첫 수업 2020.07.13

from flask import Flask

app = Flask(__name__) # Flask를 땡겨온다!

@app.route('/')
def hello333() :
    return "<h1>hello inyoung world</h1>"

@app.route('/bit')
def hello334() :
    return "<h1>hello bit computer world</h1>"

@app.route('/gema')
def hello335() :
    return "<h1>hello GEMA world</h1>"


@app.route('/bit/bitcamp')
def hello336() :
    return "<h1>hello bitcamp world</h1>"


if __name__ == '__main__' :
    app.run(host = '127.0.0.1', port = 8888, debug = True) # port는 5000번대 이후로 아무거나 쓰면 된다

# host : 자기 자신 컴퓨터
# 남들이 볼 때는 192.168.0.1
