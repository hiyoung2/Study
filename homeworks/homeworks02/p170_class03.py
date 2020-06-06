# 6.3.3 클래스(메서드)

# 앞서 정의한 클래스에는 메서드가 없었다
# 이번에는 MyProduct 클래스에 다음과 같이 메서드를 정의

# * 메서드
# - 상품을 n개 구매하고, 재고률 갱신 : buy_up(n)
# - 상품을 n개 판매하고, 재고와 매출을 갱신 : sell(n)
# - 상품의 개요를 출력 : summary()

# MyProduct 클래스 정의에 위 세 메서드를 추가

# class MyProduct :
#     def __init__(self, name, price, stock) :
#         self.name = name
#         self.price = price
#         self.stcok = stock
#         self.sales = 0
#     # 구매 메서드
#     def buy_up(self, n) :
#         self.stock += n
#     # 판매 메서드
#     def sell(self, n) :
#         self.stock -= n
#         self.sales += n*self.price
#     def summary(self) :
#         message = "called summary().\n name: " + self.name + \
#             "\n price : " + str(self.price) + \
#             "\n stock : " + str(self.stock) + \
#             "\n sales : " + str(self.sales)
#     print(message)

# 메서드를 정의할 때는 생성자와 마찬가지로 첫 번째 인수로 self를 지정해야 하며
# 멤버 앞에 self. 을 붙여야 한다
# 다른 부분은 일반 함수처럼 정의하면 된다
# 메서드를 호출할 때는 '객체.메서드명'을 사용한다
# 멤버는 직접 참조할 수도 있지만, 객체 지향으로는 바람직하지 않다
# 멤버가 간단히 변경되지 않도록 하는 것이 좋은 클래스 설계의 기본이며
# 객체 지향 언어를 사용하는 이상 가급적 이를 따라야 한다
# 따라서 멤버의 참조와 변경을 위한 전용 메서드를 준비하는 것이 좋다

# 문제 
# - MyProduct에 다음 메서드를 추가
# * name의 값을 취득해서 반환 : get_name()
# * price를 n만큼 낮춤 : discount(n)
# - product_2의 price를 5,000만큼 낮추고 , summary() 메서드로 요약 정보를 출력

class MyProduct :
    def __init__(self, name, price, stock) :
        self.name = name
        self.price = price
        self.stock = stock
        self.sales = 0
        #요약 메서드
        # 문자열과 '자신의 메서드'나 '자신의 멤버'를 연결하여 출력
    def summary(self) :
            message = "called summary()." + \
                "\n name: " + self.get_name() + \
                "\n price: " + str(self.price) + \
                "\n stock: " + str(self.stock) + \
                "\n sales: " + str(self.sales) 
            print(message)
    def get_name(self) :
            return self.name
    def discount(self, n) :
            self.price -= n
product_2 = MyProduct("phone", 30000, 100)
product_2.discount(5000)
product_2.summary()