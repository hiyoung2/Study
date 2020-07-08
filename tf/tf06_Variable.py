import tensorflow as tf
tf.set_random_seed(777)

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

print("W :", W) # W : <tf.Variable 'weight:0' shape=(1,) dtype=float32_ref>