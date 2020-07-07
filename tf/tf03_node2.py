# 3 + 4 + 5
# 4 - 3
# 3 * 4
# 4 / 2

import tensorflow as tf

node1 = tf.constant(3.0, tf.float32) # 숫자 입력, 타입 지정
node2 = tf.constant(4.0) 
node3 = tf.constant(5.0)
node4 = tf.constant(2.0)

sess = tf.Session()
print(sess.run([node1, node2, node3, node4]))

# tf.add_n?
# 많은 양의 텐서를 한 번에 처리할 때 사용
# tf.add_n은 텐서들을 반드시 대괄호 안에 넣어줘야 한다(그렇지 않으면 TypeError 발생)
# and, tf.add(node1, variable2) 처럼 constant 텐서와 variable 텐서 간의 연산도 가능하다는 점 참고!
node_add = tf.add_n([node1, node2, node3])
node_sub = tf.subtract(node2, node1)
node_mul = tf.multiply(node1, node2)
node_div = tf.divide(node2, node4)
print(sess.run(node_add))

print(sess.run([node_add, node_sub, node_mul, node_div]))

print("sess.run.node_add :", sess.run(node_add))
print("sess.run.node_sub :", sess.run(node_sub))
print("sess.run.node_mul :", sess.run(node_mul))
print("sess.run.node_div :", sess.run(node_div))
