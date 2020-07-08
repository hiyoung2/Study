# hypothesis를 구하시오
# H = Wx + b
# aaa, bbb, ccc 자리에 각 hypothesis를 구하시오

import tensorflow as tf
tf.set_random_seed(777)

x = [1.0, 2.0, 3.0]
W = tf.Variable([0.3])
b = tf.Variable([1.0])

hypothesis = W * x  + b

print(x)
print(W)
print(b)


# 1) 
sess = tf.Session() 
sess.run(tf.global_variables_initializer()) 
aaa = sess.run(hypothesis)
print("hypothesis :", aaa)   
sess.close() 

# 2)
sess = tf.InteractiveSession() 
sess.run(tf.global_variables_initializer())
bbb = hypothesis.eval() 
print("hypothesis :", bbb) 
sess.close()

# 3)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = hypothesis.eval(session = sess) 
print("hypothesis :", ccc)
sess.close()


