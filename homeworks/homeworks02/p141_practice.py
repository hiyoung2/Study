# 연습문제
# 상품의 가격, 개수, 지불금액, 거스름돈을 표시하는 프로그램을 작성
# 변수 items를 for문으로 루프시킨다, 변수는 item으로 한다
# for문의 처리
# - '**은(는) 한 개에 **원이며, **개 구입합니다.'라고 출력하라
# - 변수 totla_price에 가격x수량을 더해서 저장
# '지불해야 할 금액은 **원입니다.'라고 출력
# 변수 money에 임의의 값을 대입
# money > total_price일 때, '거스름돈은 **원입니다.' 라고 출력
# money == total_price일 때, '거스름돈은 없습니다.' 라고 출력
# money < total_price일 때, '돈이 부족합니다.' 라고 출력

items = {"지우개" : [100, 2], "펜" : [200, 3], "노트" : [400, 5]}
total_price = 0

for item in items :
    print(item + "은(는) 한 개에 " + str(items[item][0]) + "원이며, " + str(items[item][1]) + "개 구입합니다.")
    total_price += items[item][0] * items[item][1]


print("지불해야 할 금액은 " + str(total_price) + "원입니다.")

money = 3000

if money > total_price :
    print("거스름돈은 " + str(money - total_price) + "원입니다.")
elif money == total_price :
    print("거스름돈은 없습니다.")
else :
    print("돈이 부족합니다.")

# 'items["키"]'로 리스트의 내용을 꺼낼 수 있다
# 'items["키"][인덱스 번호]'로 값을 추출할 수 있다
# 따라서 가격 및 수량은 items[item][0], items[item][1]로 찾을 수 있다
# 또한 str()의 ()안에 연산자를 사용할 수 있다