import tensorflow as tf

node1 = tf.constant(3.0, tf.float32) # 숫자 입력, 타입 지정
node2 = tf.constant(4.0) 
node3 = tf.add(node1, node2)

print("node1 :", node1, "node2 :", node2)
print("node3 :", node3)
# node1 : Tensor("Const:0", shape=(), dtype=float32) node2 : Tensor("Const_1:0", shape=(), dtype=float32)
# node3 : Tensor("Add:0", shape=(), dtype=float32)

sess = tf.Session()
print("sees.run(node1, node2) :", sess.run([node1, node2]))
print("sees.run(node3) :", sess.run(node3))
# sees.run(node1, node2) : [3.0, 4.0]
# sees.run(node3) : 7.0
