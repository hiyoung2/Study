# 2020.07.08

# x_train 과  y_train에  placeholer 를 적용 시켜 소스 완성
# feed_dict를  with 문 안에 있는 sees.run 부분에 넣으면 된다!
# tf04_placeholder 파일에 보면 
# >>> sess.run 할 때에 feed_dict를 쓰면 된다
# 라고 했으므로!

import tensorflow as tf
tf.set_random_seed(777)

# x_train = [1, 2, 3]
# y_train = [3, 5, 7]

# placeholder로 선언
x_train = tf.placeholder(tf.float32, shape = [None])
y_train = tf.placeholder(tf.float32, shape = [None])

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = x_train * W + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))

train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)

# with tf.compat.v1.Session() as sess : # warning 문제 해결 # 아래보다 이것을 권장한다는 warning임(compile에는 문제가 없음)
with tf.Session() as sess : # with 문을 쓴 이유 
                            # Session을 쓰면 닫아줘야하는데(-> 무슨 의미?)
                            # with 문을 쓰면 sess를 닫아준다
    # sess.run(tf.compat.v1.global_variables_initializer()) # 이렇게 쓰면 warning 해결
    sess.run(tf.global_variables_initializer()) # '변수를 선언하겠다'로 이해
                                                # 초기화는 한 번만 이루어진 것(for문 영역에 속하지 않음!)

    for step in range(2001) : 
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b], feed_dict = {x_train:[1, 2, 3], y_train:[3, 5, 7]}) 

        if step % 20 == 0 : 
            print(step, cost_val, W_val, b_val) 

    # predict !
    print("예측(4) :", sess.run(hypothesis, feed_dict = {x_train :[4]})) # with문 안으로 들어가야 run 작동
    print("예측(5, 6) :", sess.run(hypothesis, feed_dict = {x_train : [5, 6]}))
    print("예측(6, 7, 8) :", sess.run(hypothesis, feed_dict = {x_train : [6, 7, 8]}))
    # 예측(4) : [9.000078]
    # 예측(5, 6) : [11.000123 13.000169]
    # 예측(6, 7, 8) : [13.000169 15.000214 17.000257]