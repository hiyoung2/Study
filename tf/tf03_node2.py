# 사 칙 연 산 을 해 보 시 오.
# 3 + 4 + 5
# 4 - 3
# 3 * 4
# 4 / 2

import tensorflow as tf

node1 = tf.constant(3.0, tf.float32) # 숫자 입력, 타입 지정
node2 = tf.constant(4.0) 
node3 = tf.constant(5.0)
node4 = tf.constant(2.0)

# sess 도 변수명? 맘대로 해 줘도 될 듯하다
sess = tf.Session()
print(sess.run([node1, node2, node3, node4]))

# 텐서플로 덧셈
# tf.add_n?
# 많은 양의 텐서를 한 번에 처리할 때 사용
# tf.add_n은 텐서들을 반드시 대괄호 안에 넣어줘야 한다(그렇지 않으면 TypeError 발생)
# 3개 이상이면 [] 리스트로 묶어주나 봄
# and, tf.add(node1, variable2) 처럼 constant 텐서와 variable 텐서 간의 연산도 가능하다는 점 참고!

# 텐서플로 곱셈
# tf.multiply vs tf.matmul
# tf.multiply는 원소들의 곱 -> 우리가 알고 있는 행렬의 곱셈 방식이 아님
'''
1 2  1 0  1*1 2*0
3 4  0 1  3*0 4*1
'''
# 행렬의 곱셈은
# tf.matmul 이라는 함수를 통해 구현한다
'''
1 2  1 0  (1*1)+(2*0) (1*0)+(2*1)
3 4  0 1  (3*1)+(4*0) (3*0)+(4*1)
'''

# 텐서플로 나눗셈
# tf.divide and tf.mod
# tf.divide : ~를 ~로 나누면?
# tf.mod : ~를 ~로 나눈 나머지?

node_add = tf.add_n([node1, node2, node3]) # 그냥 add(~) 해서 에러 발생함! 위 텐서플로 덧셈 참고
node_sub = tf.subtract(node2, node1)
node_mul = tf.multiply(node1, node2)
node_div = tf.divide(node2, node4)

print(node_add)
# Tensor("AddN:0", shape=(), dtype=float32) # Session 통과 시키지 않았으므로 이렇게 출력됨
# 이름이 AddN 이라고 뜸! 
# 이름을 node_add 라고 줬는데

# print(sess.run(node_add)) # for test

print(sess.run([node_add, node_sub, node_mul, node_div]))

print("sess.run.node_add :", sess.run(node_add))
print("sess.run.node_sub :", sess.run(node_sub))
print("sess.run.node_mul :", sess.run(node_mul))
print("sess.run.node_div :", sess.run(node_div))
# sess.run.node_add : 12.0
# sess.run.node_sub : 1.0
# sess.run.node_mul : 12.0
# sess.run.node_div : 2.0