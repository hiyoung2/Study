import tensorflow as tf
tf.set_random_seed(777)

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

print("W :", W) # W : <tf.Variable 'weight:0' shape=(1,) dtype=float32_ref>

# 변수는 feed_dict 할 필요가 없다(placeholder는 항상 feed_dict 해 줘야 함)

# W , weight 값에 변화룰 줌
W = tf.Variable([0.3], tf.float32)

# W를 보려면
sess = tf.Session() # 1. sess 설정
sess.run(tf.global_variables_initializer()) # 2. 변수 초기화 
aaa = sess.run(W)
print("aaa :", aaa)   # [0.3]
sess.close() # 원래는 이렇게 명시를 해 줘야 한다
             # sess : 메모리를 열어서 작업을 함
             # 작업이 끝나면 메모리를 닫아줘야하므로 sess.close()
             # 작은 소스는 상관 없지만 큰 소스에서는 close 해 주지 않으면 파일 간 엉키는 문제가 발생할 수 있다
             # 그것을 방지하기 위해서 우리는 앞에서 with로 close()를 대신 했었음

# Session의 친구들이 있다, New~ Session!
sess = tf.InteractiveSession() # 상호작용세션?
sess.run(tf.global_variables_initializer())
bbb = W.eval() # InteractiveSession을 사용하면 sess.run(W) 하지 않고 W.eval() 하면 된다
print("bbb :", bbb) # [0.3]
sess.close()


sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = W.eval(session = sess) #  일반 Session 에서도 eval이 먹히는데, 이 때는 session = sess 임을 명시해줘야 한다
print("ccc :", ccc) # [0.3]
sess.close()



# 정리
# 우리들이 선택할 수 있다 : 1) tf.session() 2) tf.InteractiveSession() 3) tf.session() + eval
# 문법적으로 다른 부분이 있고 기능은 똑같다
# 1) tf.session()
# sess = tf.Session()
# sess.run(tf.global_variabls_initializer())
# 변수명 = sess.run(Variable값)
# sess.close()

# 2) tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# 변수명 = Variable값.eval()
# sess.close()

# 3) tf.Session()
# sess.run(tf.global_variables_initializer())
# 변수명 = Variable값.eval(session = sess)
# sess.close()
