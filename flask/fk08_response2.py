from flask import Flask, Response, make_response

app = Flask(__name__)

@app.route('/')
def response_test() :
    custom_response = Response('Custom Response', 200, 
                               {"Program" : "Flask Web Application"})
    return make_response(custom_response)

# @route의 친구들

@app.before_first_request
def before_first_request() :
    print("[1] 앱이 기동되고 나서 첫 번째 HTTP 요청에만 응답합니다.") # 앱이 실행되면 가장 먼저 실행
                                                                  # print문이 먼저 생성된 후 웹이 생성됨
    # print("이 서버는 개인 자산이니 건드리지 말 것")
    # print("곧 자료를 전송합니다,")


@app.before_request 
def before_reques() :
    print("[2] 매 HHTP 요청이처리되기 전에 실행됩니다") # 매번 실행된다

@app.after_request
def after_request(response) :
    app.run(host = '127.0.0.1')
    print("[3] 매  HTTP 요청이 처리되고 나서 실행됩니다") # 역시 매번 실행
    return response

@app.teardown_request
def teardown_request(exception) :
    print("[4] 매 HTTP 요청의 결과가 브라우저에 응답하고 나서 호출된다.")

@app.teardown_appcontext
def teardown_appcontext(exception) :
    print("[5] HTTP 요청의 어플리케이션 컨텍스트가 종료할 때 실행된다")


if __name__ == "__main__" :
    app.run(host = "127.0.0.1") # port 따로 표시 안 함 -> 실행 되고 터미널에 5000으로 뜸
                                # default : 5000

