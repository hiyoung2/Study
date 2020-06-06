# 6.3.4 클래스(상속, 오버라이드, 슈퍼)

# 다른 사람이 만든 클래스에 기능을 추가하고 싶을 땐 어떻게?
# 클래스를 직접 변경할 수도 있겠지만, 그 클래스를 사용 중인 다른 프로그램에 영향을 줄 지도
# 해당 소스를 복사하여 새 클래스를 만드는 것도 가능하지만 유사한 프로그램이 두 개가 되어버린다
# 수정할 경우 같은 작업을 두 번 해야 한다

# 이럴 때를 대비해 객체 지향 언어는 상속, inheritance이라는 스마트한 구조를 제공한다
# 기존의 클래스를 바탕으로 메서드나 멤버를 추가하거나 일부 변경하여 새로운 클래스를 만들 수 있다
# 바탕이 되는 클래스는 부모 클래스, 슈퍼 클래스, 기저 클래스 등으로 부르고
# 새로 만든 클래스는 자식 클래스, 서브 클래스, 파생 클래스 등으로 부른다
# 자식 클래스에서는 다음과 같은 일이 가능하다

# * 부모 클래스의 메서드와 멤버를 그대로 사용할 수 있다
# * 부모 클래스의 메서드와 멤버를 덮어쓸 수 있다(오버라이드)
# * 자기 자신의 메서드와 멤버를 자유롭게 추가할 수 있다
# * 자식 클래스에서 부모 클래스의 메서드와 멤버를 호출할 수 있다(슈퍼)

# 앞의 MyProduct를 상속하여 10%의 소비세를 적용한 MyProductSalesTax를 새로 만들자

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

# class MyProductSalesTax(MyProduct) : # MyProduct 클래스를 상속하는 MPST를 정의
#     def __init__(self, name, price, stock, tax_rate) :
#         # super()를 사용하면 부모 클래스의 메서드를 호출할 수있다
#         # 지금은 MyProduct 클래스의 생성자를 호출
#         super().__init__(name, price, stock)
#         self.tax_rate = tax_rate
#     # MYST에서 MP의 get_name을 재정의(오버라이드) 한다
#     def get_name(self) :
#         return self.name + "(소비세 포함)"
#     # MYST에서 get_price_with_tax를 새로 구현
#     def get_price_with_tax(self) :
#         return int(self.price* ( 1 + self.tax_rate))

# product_3 = MyProductSalesTax("phone", 30000, 100, 0.1)
# print(product_3.get_name())
# print(product_3.get_price_with_tax())

# product_3.summary()

'''
phone(소비세 포함)
33000
called summary().
 name: phone(소비세 포함)
 price: 30000
 stock: 100
 sales: 0
'''

# summary() 메서드로 호출한 결과 price가 소비세를 포함하지 않은 가격을 반환
# 즉, 새로 구현한 get_name() aptjemdhk get_price_with_tax() 메서드는 예상대로 작동
# but, MyProduct로 상속한 summary() 메서드가 소비세를 포함하지 않는 가격을 반환한느 버그 발생

# 어떻게 해결?

class MyProductSalesTax(MyProduct) : # MyProduct 클래스를 상속하는 MPST를 정의
    def __init__(self, name, price, stock, tax_rate) :
        # super()를 사용하면 부모 클래스의 메서드를 호출할 수있다
        # 지금은 MyProduct 클래스의 생성자를 호출
        super().__init__(name, price, stock)
        self.tax_rate = tax_rate
    # MYST에서 MP의 get_name을 재정의(오버라이드) 한다
    def get_name(self) :
        return self.name + "(소비세 포함)"
    # MYST에서 get_price_with_tax를 새로 구현
    def get_price_with_tax(self) :
        return int(self.price* ( 1 + self.tax_rate))
    
    def summary(self) :
        message = "called summary().\n name: " + self.get_name() + \
            "\n price: " + str(self.get_price_with_tax()+0) + \
            "\n stock: " + str(self.stock) + \
            "\n sales: " + str(self.sales)
        print(message)

product_3 = MyProductSalesTax("phone", 30000, 100, 0.1)
print(product_3.get_name())
print(product_3.get_price_with_tax())
product_3.summary()

'''
phone(소비세 포함)
33000
called summary().
 name: phone(소비세 포함)
 price: 33000
 stock: 100
 sales: 0
'''