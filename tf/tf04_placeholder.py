import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) 
node3 = tf.add(node1, node2)

sess = tf.Session() # tensor machine을 만든 상태라고 생각

# placeholder 준비
a = tf.placeholder(tf.float32) # 'a'라는  placeholder를 지정해준다
b = tf.placeholder(tf.float32)

# adder_node 이라는 node 생성
adder_node = a + b # adder_node 라는 변수에 a+b를 대입

# feed_dict : 집어 넣는다는 의미 -> 쉽게 input이라고 생각
# placeholder에 값을 집어 넣으려면 feed_dict 를 써야 한다!!
# sess.run 할 때에 feed_dict를 쓰면 된다

# tensorflow machine에서 sess.run을 하면 결괏값이 나온다
# tensor machine 에서 연산을 꼭 거친다고 생각하면 된다
print(sess.run(adder_node, feed_dict={a:3, b:4.5})) # 7.5
print(sess.run(adder_node, feed_dict={a:[1,3], b:[2,4]})) # [3. 7.] # 리스트 형태도 가능하다 # 넘파이 연산

# add_and_triple이라는 또 다른  node 생성
add_and_triple = adder_node * 3
print(sess.run(add_and_triple, feed_dict = {a:3, b:4.5})) # 22.5

# 전체적으로 봤을 때 무엇과 비슷? -> 인공신경망 레이어 구성하는 것과 비슷
# placeholder로 준비한 a, b -> input layer 처럼 보임
# sess.run을 거친 print문으로 출력된 예를 들어 7.5, [3. 7.], 22.5 이런 것들은 -> output layeer (또는 hidden layer가 될 수도) 처럼 보임
