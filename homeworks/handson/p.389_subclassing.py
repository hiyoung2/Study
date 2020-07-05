# 10.2.5 서브클래싱 API로 동적 모델 만들기

# 시퀀셜 API와 함수형 API는 모두 선언적, declarative 이다
# 사용할 층과 연결 방식을 먼저 정의해야 한다
# 그 다음 모델에 데이터를 주입하여 훈련이나 추론을 시작할 수 있다
# 이 방식에는 장점이 많다
# 모델을 저장, 복사, 공유하기가 쉽고 모델의 구조를 출력하거나 분석하기 좋다
# 프레임워크가 크기를 짐작하고 타입을 확인하여 에러를 일찍 발견할 수 있다
# (모델에 데이터가 주입되기 전에)
# 전체 모델이 층으로 구성된 정적 그래프이므로 디버깅하기도 쉽다
# 하지만! 정적이라는 것이 단점도 된다
# 어떤 모델은 바녹문을 포함하고 다양한 크기를 다루어야 하며, 조건문을 가지는 등의 여러 가지
# 동적인 구조를 필요로 한다
# 이런 경우에 조금 더 명령형, imperative 프로그래밍 스타일이 필요하다면, 서브클래싱 subclassing API가 정답이다

# 간단히 Model 클래스를 상속한 다음 생성자 안에서 필요한 층을 만든다
# 그 다음 call() 메서드 안에 수행하려는 연산을 기술한다
# 예를 들어 다음의 WideAndDeepModel 클래스의 인스턴스는 앞서 함수형 API로 만든 모델과 동일한 기능을 수행한다
# 이전에 한 것처럼 이 인스턴스를 사용해 모델 컴파일, 훈련, 평가, 예측을 수행할 수 있다
import keras

class WideAndDeepModel(keras.model) :
    def __init__(self, units = 30, activation = 'relu', **kwargs) :
        super().__init__(**kwargs) # 표준 매개변수를 처리한다(예를 들면, name)
        self.hidden1 = keras.layers.Dense(units, activation = activation)
        self.hidden2 = keras.layers.Dense(units, activation = activation)
        self.main_output = keras.layers.Dense(1)
        self.aux_output = keras.layers.Dense(1)

    def call(self, inputs) :
        input_A, input_B = inputs
        hidden1 = self.hidden1(input_B)
        hidden2 = self.hidden2(hidden1)
        concat = keras.layers.concatenate([input_A, hidden2])
        main_output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return main_output, aux_output

model = WideAndDeepModel()

# 이 예제는 함수형 API와 비슷하지만 Input 클래스의 객체를 만들 필요가 없다
# 대신 call() 메서드의 input 매개변수를 사용한다
# 생성자에 있는 층 구성과 call()에 있는 정방향 계산을 분리했다
# 주된 차이점은 call() 메서드 안에서 원하는 어떤 계산도 사용할 수 있다는 점이다
# for문, if문, 텐서플로 저수준 연산을 사용할 수 있다

# 유연성이 높아지면 그에 따른 비용이 발생하는 법
# 모델 구조가 call() 메서드 안에 숨겨져 있기 때문에 케라스가 쉽게 이를 분석할 수가 없다
# 즉, 모델을 저장하거나 복사할 수가 없다
# summary() 메서드를 호출하면 층의 목록만 나열되고 층 간의 연결 정보를 얻을 수 없다
# 또한 케라스가 타입과 크기를 미리 확인할 수 없어 실수가 발생하기 쉽다
# 높은 유연성이 필요하지 않는다면 시퀀셜 API와 함수형 API를 사용하는 것이 좋다

# TIP : 케라스 모델은 일반적인 층처럼 사용할 수 있다, 따라서 모델을 연결하여 더 복잡한 구조를 만들 수 있다

