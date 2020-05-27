# 2.18 자동 테스트와 assert

# 코드가 제대로 작성되었는지 어떻게 확인?
# 타입(type)이나 자동 테스트(automated test)를 통해 코드 작성 상태 확인이 가능
# 다양한 테스팅 프레임워크가 존재, 일단 assert문만 사용
# assert는 지정된 조건이 충족되지 않는다면 AssertionError를 반환함

assert 1 + 1 == 2
assert 1 + 1 == 2, "1 + 1 should equal 2 but didin't"

# 위의 두 번째 예시처럼 조건이 충족되지 않을 때 출력하고 싶은 문구를 추가할 수 있다

# 함수 테스팅

def smallest_item(xs):
    return min(xs)

assert smallest_item([10, 20, 5, 40]) == 5
assert smallest_item([1, 0, -1, 2]) == -1

# 자주 사용하는 방식은 아니지만 assert로 함수의 인자를 검증할 수도 있다

# def smallest_item(xs):
#     assert xs, "empty list has no smallest item"
#     return min(xs)

