import tensorflow as tf
import numpy as np

dataset = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(dataset.shape) # (10,)

# RNN 모델을 구성하라

size = 6
def split_x(seq, size):                            
                                                    
    aaa = []                                      
    for i in range(len(seq) - size + 1):         
        subset = seq[i : (i+size)]                  
        aaa.append([item for item in subset])    
        # == aaa.append(subset) 더 단순하게 표현

    print(type(aaa))                                
    return np.array(aaa)                            

dataset = split_x(dataset, size)
print(dataset)
print()
'''
[[ 1  2  3  4  5  6]
 [ 2  3  4  5  6  7]
 [ 3  4  5  6  7  8]
 [ 4  5  6  7  8  9]
 [ 5  6  7  8  9 10]]
'''

x_data = dataset[:, :-1]
y_data = dataset[:, -1]

print(x_data)
print()
print(y_data)
print()
'''
x_data
[[1 2 3 4 5]
 [2 3 4 5 6]
 [3 4 5 6 7]
 [4 5 6 7 8]
 [5 6 7 8 9]]

y_data
[ 6  7  8  9 10]
'''
print("x_data.shape :", x_data.shape) # (5, 5)
print("y_data.shape :", y_data.shape) # (5,)
print()

x_data = x_data.reshape(1, 5, 5)
y_data = y_data.reshape(1, 5)

print("x_data")
print(x_data)
print("y_data")
print(y_data)
print()

print("x_data.shape :", x_data.shape) # (1, 5, 5)
print("y_data.shape :", y_data.shape) # (1, 5)
print()



sequence_length = 5
input_dim = 5
output = 10 # 10 아래로는 잘못된 인자를 받았다는 InvalidArgumentError 발생함
batch_size = 1 
lr = 0.003

X = tf.compat.v1.placeholder(tf.float32, (None, sequence_length, input_dim))
Y = tf.compat.v1.placeholder(tf.int32, (None, sequence_length))
print(X) # Tensor("Placeholder:0", shape=(?, 5, 5), dtype=float32)
print(Y) # Tensor("Placeholder_1:0", shape=(?, 5), dtype=int32)

cell = tf.nn.rnn_cell.BasicLSTMCell(output) 
# cell = tf.keras.layers.LSTMCell(output)

hypothesis, _states = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32)

print("hypothesis :", hypothesis)
# hypothesis : Tensor("rnn/transpose_1:0", shape=(?, 5, 5), dtype=float32)

weights = tf.ones([batch_size, sequence_length]) 

sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits = hypothesis, targets = Y, weights = weights
)

cost = tf.reduce_mean(sequence_loss)

train = tf.compat.v1.train.AdamOptimizer(learning_rate = lr).minimize(cost)

prediction = tf.argmax(hypothesis, axis = 2)


with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    for i in range(501) :
        loss, _ = sess.run([cost, train], feed_dict = {X:x_data, Y:y_data})

        result = sess.run(prediction, feed_dict = {X:x_data})

        print(i, "loss :", loss, "\nPredict :", result, "\nTrue_Y :", y_data)

'''
sequence_length = 5
input_dim = 5
output = 11 
batch_size = 1 
lr = 0.003

epo = 500

500 loss : 1.2132335
Predict : [[ 6  7  8  9 10]]
True_Y : [[ 6  7  8  9 10]]
'''