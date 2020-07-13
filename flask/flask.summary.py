# # Flask란?
# 플라스크(Flask)란 파이썬 웹 어플리케이션을 만드는 프레임 워크
# 다양한 프레임워크들 중에서 플라스크는 심플하고 가벼운 느낌을 가지고 있으면서
# 가벼움 속에서도 중요하고 핵심적인 내용과 기능을 갖고 있다

# from flask import Flask : 
# flask에서 Flask라는 Class를 불러온다

# app = Flask(__name__) : 
# Flask라는 Class의 객체를 생성하고 인수로서 __name__을 입력한다
# 해당 객체의 이름은 app으로 설정된 것
# app이라는 변수에 flask 프로젝트를 초기화 시켜서 실행하겠다는 코드이다

# @app.route('/') : 
# 생성한 객체 app에 대해 route를 설정한다, 즉 URL을 설정해주는 것이다
# '/' 는 http://<기본 주소> 와 같은 경로이다
# ex) @app.route('/route_test')라고 한다면,
# http://<기본 주소>/route_test와 같은 경로가 된다


# def hello_world() :
#     return 'Hello World!'

# 그리고 함수를 만들고 함수의 기능을 설명한다
# 바로 위에서 설정한 경로에 사용자가 요청을 보냈을 때 실행되는 것이다
# hello_world라는 함수를 실행할 것이고, 그 함수의 내용이 바로 return에 담긴 것
# 'Hello World'를 반환하도록 함수를 정의하였다
# 즉, 'Hello World:라는 문자열을 보여달라고 한 것

# if__name__ == '__main__' :
#     app.run()
# 여기에서는 객체의 run 함수를 이용하여 로컬 서버에서 어플리케이션을 실행하도록 한다
# 단순하게, 해당 플라스크 프로젝트를 실행시키는 코드라고 이해하면 된다

# # Cookie
# 쿠키는 클라이언트의 PC에 텍스트 파일 형태로 저장되는 것으로 일반적으로는 시간이 지나면 소멸한다
# 보통 세션과 더불어 자동 로그인, 팝업 창에서 "오늘은 이 창을 더 이상 보지 않기" 등의 기능을
# 클라이언트에 저장해놓기 위해 사용한다!
# -> 오늘 이 창을 안 본다고 클릭하면 그 내용을 저장해둔다는 말인 듯, 
# 즉, 쿠키를 사용하여 저장해두면 오늘 동안은 그 창이 안 뜨게 

# make_response() 
# : 이 함수는 사용자에게 반환할 veiw 함수를 생성한 후, 그대로 묶어두는 역할을 한다

# set_cookie()
# : 이 함수는 쿠키를 생성하는 함수이다 

# request.cookies.get() 
# : 이 함수는 사용자의 페이지 요청과 함께 전송된 쿠키들을 가져올 수 있게 한다
# 쿠키는 사용자가 별도로 뭘 하지 않아도 요청과 동시에 자동으로 전송된다

# # Session
# Session은 Cookie와 다르게 관련된 데이터들이 서버에 저장된다
# 서버에서 관리할 수 있다는 점에서 '안전성'이 좋아서 보통 로그인 관련으로 사용되고 있다
# 플라스크에서 Session은 딕셔너리 형태로 저장되고 키를 통해 해당 값을 불러올 수 있다

# Session을 사용하기 위해서는 해당 값을 암호하기 위한 Key 값을 코드에서 지정해줘야 한다

# ex)
# from flask import Flaask, request, session, redirect, url_for
# app = Flask(__name__)
# app.secret_key = 'any random string'

# # redirect
# : 특정 URL로 강제로 이동 시키고 싶을 때 사용하는 함수, redirect()

###################################################################

# # Flask
# Flask는 WSGI 라이브러리인 Werkzeug를 만들기도 한 Armin Ronacher가 만든 프레임워크
# "마이크로"라는 수식어에 어울리게 아주 핵심적인 부분만을 구현하지만
# 유연하게 확장이 가능하게 설계된 것이 특징이다

# Flask는(정확히는 Werkzeug) 테스트를 위해 간단한 WSGI 서버를 자체 내장하고 있기 때문에
# app.run 을 통해 어플리케이션을 직접 실행할 수 있다

# # Route
# @app.route('/')  
# 이 메서드는 URL 규칙을 받아 해당하는 규칙의 URL로 요청이 들어온 경우,
# 등록한 함수를 실행하게끔 설정한다
# 규칙을 URL로부터 변수도 넘겨 받을 수 있다

# 이렇게 URL을 통해 처리할 핸들러를 찾는 것을 일밙거으로 URL 라우팅(Routing)이라고 한다
# 이런 URL Routing에서 중요한 기능 중 하나는 핸들러에서 해당하는 URL을 생성하는 기능인데,
# Flask는 이를 url_for 이란 메서드를 통해 지원한다

# # Template
# 일반적으로 웹 어플리케이션의 응답은 단순한 문자열보다 대부분이 훨씬 더 복잡하다
# 이를 보다 쉽게 작성할 수 있게끔 도와주는 것이 바로 Flask의 Template 이다

# 기본적으로 템플릿엔진은 별도의 규칙에 맞게 작성된 템플릿 파일을 읽어
# 환경(Context)에 맞게 적용한 결과물을 돌려준ㄴ데, 이 과정을 Flask에서는
# render_template()가 담당하고 있다

# 다음의 코드는 hello.html 이라는 템플릿 파일을 읽어서 이름을 적용한 뒤에 돌려주는 코드이다
# from flask import render_template

# @app.route('/hello/')
# @app.route('/hello/<namae>')
# def hello(name = None) :
#     return render_template('hello.html', name = name)

# # Request
# HTTP 요청을 다루기 위해서는 때로는 environ의 내용은 너무 원시적일 때가 있다
# HTML, Form 으로부터 입력받는 값이 종은 예이다
# Flaks에서는 request 라는 객체를 통해 보다 다루기 쉽게 해 준다

# 다음은 HTML 폼으로부터 입력받은 message 라는 값을 뒤집어서 출력하는 코드이다

# from flask import request

# @app.route("./reverse/")
# def reverse() :
#     message = request.values["message"]
#     return "".join(reversed(message)


# # Session
# 로그인으로 대표되는 요청 간의 상태를 유지해야하는 처리에 사용하는 객체
