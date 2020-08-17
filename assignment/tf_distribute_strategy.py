# tf.distribute.Strategy
# tf.distribute.MirroredStrategy

# distibute : 분산하다, 나누다
# strategy : 전략

# tf.distribute.Strategy란?
# 훈련을 여러 GPU(Graphic Processing Unit) 또는 여러 장비, 여러 TPU(Tensor Processing Unit)로 
# 나누어 처리하기 위한 tensorflow API이다
# 이 API를 사용하면 기존의 모델이나 훈련 코드를 조금만 고쳐서 분산처리를 할 수 있다

# tf.distribute.Strategy에는 아래의 5가지 전략이 있다
# MirroredStrategy	
# TPUStrategy	
# MultiWorkerMirroredStrategy	
# CentralStorageStrategy	
# ParameterServerStrategy

# 그 중 MirroredStrategy는 장비 하나에 다중 GPU를 이용한 동기(synchronous) 분산 훈련을 지원한다
# 각각의 GPU 장치마다 복제본이 만들어진다
# 모델의 모든 변수가 복제본마다 미러링 된다
# 이 미러링된 변수들은 하나의 가상의 변수에 대응되는데, 이를 MirroredVariable이라고 한다
# 이 변수들은 동일한 변경사항이 함께 적용되므로 모두 같은 값ㄷ을 유지한
# 여러 장치에 변수의 변경사항을 전달하기 위해 효율적인 all-reduce algorithms을 사용한다
# all-reduce algorithms 이란, 모든 장치에 걸쳐 tensor를 모은 다음 그 합을 구하여 다시 각 장비에 제공하는 것을 말한다
# 이 통합된 알고리즘은 매우 효율적이어서 동기화의 부담을 많이 덜어낼 수 있다
# 장치 간에 사용 가능한 통신 방법에 따라 다양한 all-reduce algorithms과 구현이 있다
# 기본값으로는 NVIDIA NCCL을 all-reduce algorithms 구현으로 사용한다

# MirroredStrategy를 만드는 방법은 다음과 같다
# mirrored_strategy = tf.distribute.MirroredStrategy()
# 장비의 GPU 중 일부만 사용하고 싶다면, 다음과 같이 하면 된다
# mirrored_strategy = tf.distribute.MirroredStrategy(deviceds=["/gpu:0", "/gpu:1"])


# mnist 예제

# 필요한 packages
import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import train_test_split as tts

# dataset download
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("x_train.shape :", x_train.shape) # (60000, 28, 28)
print("x_test.shape :", x_test.shape) # (10000, 28, 28)
print("y_train.shape :", y_train.shape) # (60000,)
print("y_test.shape :", y_test.shape) # (10000,)

# 분산전략 정의
strategy = tf.distribute.MirroredStrategy()
# print("장치의 수 : {}".format(strategy.num_replicas_in_sync)) # 1

# 입력 파이프라인 구성
# 다중 GPU로 모델을 훈련할 때에는 배치 크기를 늘려야 컴퓨팅 자원을 효과적으로 사용할 수 있다
# 기본적인 GPU 메모리에 맞추어 가능한 가장 큰 배치를 사용하고 이에 맞게 학습률도 조정해야 한다

# buffer_size = 10000
# batch_size_per_replica = 64
# batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255

with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

model.fit(x_train, y_train, epochs = 32, validation_split = 0.3)








################################################################################################################
# from __future__ import absolute_import, division, print_function, unicode_literals

# import tensorflow_datasets as tfds
# import tensorflow as tf
# tfds.disable_progress_bar

# import os

# datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
# mnist_train, mnist_test = datasets['train'], datasets['test']

# # 4. 분산 전략 정의하기
# strategy = tf.distribute.MirroredStrategy()
# print("장치의 수 : {}".format(strategy.num_replicas_in_sync))

# num_train_examples = info.splits['train'].num_examples
# num_test_examples = info.splits['test'].num_examples

# buffer_size = 10000

# batch_size_per_replica = 64
# batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

# def scale(image, label):
#     image = tf.cast(image, tf.float32)
#     image /= 255

#     return image, label

# train_dataset = mnist_train.map(scale).shuffle(buffer_size).batch(batch_size)
# eval_dataset = mnist_test.map(scale).batch(batch_size)

# # 6. 모델 만들기
# with strategy.scope():
#     model = tf.keras.Sequential([
#             tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
#             tf.keras.layers.MaxPooling2D(),
#             tf.keras.layers.Flatten(),
#             tf.keras.layers.Dense(64, activation='relu'),
#             tf.keras.layers.Dense(10, activation='softmax')
#             ])

#     model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])


# # callback 정의
# # checkpoint 저장할 checkpoint directory 지정
# checkpoint_dir = './assignment/training_checkpoints'
# # cehckpoint file name
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

# # 학습률을 점점 줄이기 위한 함수
# def decay(epoch):
#     if epoch < 3:
#         return 1e-3
#     elif epoch>=3 and epoch <7:
#         return 1e-4
#     else:
#         return 1e-5
    
# class PrintLR(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         print("\n에포크 {}의 학습률은 {}입니다.".format(epoch + 1, model.optimizer.lr.numpy()))

# callbacks = [
#     tf.keras.callbacks.TensorBoard(log_dir = './assignment/logs'),
#     tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_prefix, save_weights_only = True),
#     tf.keras.callbacks.LearningRateScheduler(decay),
#     PrintLR
# ]

# # 8. 훈련과 평가
# model.fit(train_dataset, epochs=12, callbacks=callbacks)

# model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

# eval_loss, eval_acc = model.evaluate(eval_dataset)

# print("평가 손실: {}, 평가 정확도: {}".format(eval_loss, eval_acc))


# # path = './assignment/save_model/'
# # tf.keras.experimentalexport_saved_model(model, path)



# print("문제 없음")
