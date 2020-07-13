from flask import Flask
from flask import redirect

# from flask import Flask, redirect 

app = Flask(__name__)

@app.route('/')
def index() :
    return redirect('http://www.naver.com')

if __name__ == '__main__' :
    app.run(host = '127.0.0.1', port = 5000, debug = False)

# redirect : 도메인 주소로 바로 연결