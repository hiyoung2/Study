# 6.3.2 클래스(멤버와 생성자)
# 각각의 객체는 어떤 값을 가질지, 어떻게 처리할지 결정하기 위해 객체의 구조를 결정하는 '설계도'가 필요하다
# 이 설계도를 '클래스 , class' 라고 한다
# list 객체는 list 클래스에 설계된 대로 처리를 할 수 있다

# * 객체의 내용
# - 상품
# * 멤버
# - 상품명 : name
# - 가격 : price
# - 재고 : stock
# - 매출 : sales
# 위 상품 객체를 정의하려면 아래처럼 클래스를 정의한다

# MyProduct 클래스를 정의
# class MyProduct :

# 생성자를 정이
    # def __init__(self, name, price) :
    # # 인수를 멤버에 저장한다
    #     self.name = name
    #     self.price = price
    #     self.stock = 0
    #     self.sales = 0

# 정의된 클래스는 설계도일 뿐이므로 객체를 만들려면 클래스를 호출해야 한다

# MyProduct를 호출하여 객체 product1을 만들자
# product1 = MyProduct("cake", 500)


# 클래스를 호출할 때 작동하는 메서드를 생성자라고 한다
# 생성자는 __init__()로 정의하며, self를 생성자의 첫 번째 인수로 지정해야 한다
# 클래스 내 멤버는 self.price처럼 변수명 앞에 self.를 붙인다
# 위 예에서는 MyProduct가 호출되면 name = "cake", price = "500"으로 생성자가 작동하고
# 각 인수에 의해 멤버 name, price가 초기화된다
# 생성된 객체의 멤버를 참조할 때는 '객체.변수명'으로 직접 참조할 수 있다
# 직접 참조에서는 멤버의 변경이 가능하다

class MyProduct :

    def __init__(self, name, price, stock) :
        self.name = name
        self.price = price
        self.stock = stock

product_1 = MyProduct("cake", 500, 20)

print(product_1.stock) # 20


