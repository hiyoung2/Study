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
# 이 변수들은 동일한 변경사항이 함께 적용되므로 모두 같은 값을 유지한
# 여러 장치에 변수의 변경사항을 전달하기 위해 효율적인 all-reduce algorithms을 사용한다
# all-reduce algorithms 이란, 모든 장치에 걸쳐 tensor를 모은 다음 그 합을 구하여 다시 각 장비에 제공하는 것을 말한다
# 이 통합된 알고리즘은 매우 효율적이어서 동기화의 부담을 많이 덜어낼 수 있다
# 장치 간에 사용 가능한 통신 방법에 따라 다양한 all-reduce algorithms과 구현이 있다
# 기본값으로는 NVIDIA NCCL을 all-reduce algorithms 구현으로 사용한다

# MirroredStrategy를 만드는 방법은 다음과 같다
# mirrored_strategy = tf.distribute.MirroredStrategy()
# 장비의 GPU 중 일부만 사용하고 싶다면, 다음과 같이 하면 된다
# mirrored_strategy = tf.distribute.MirroredStrategy(deviceds=["/gpu:0", "/gpu:1"])

# 보충
# tf.distribute.MirroredStrategy 어떻게 동작하는가

# 모든 변수와 모델 그래프는 장치(replicas, 다른 문서에서는 replica가 분산 훈련에서 장치 등에 복제된 모델을 의미하는 경우가 있으나 이 문서에서는 장치 자체를 의미합니다)에 복제됩니다.
# 입력은 장치에 고르게 분배되어 들어갑니다.
# 각 장치는 주어지는 입력에 대해서 손실(loss)과 그래디언트를 계산합니다.
# 그래디언트들을 전부 더함으로써 모든 장치들 간에 그래디언트들이 동기화됩니다.
# 동기화된 후에, 동일한 업데이트가 각 장치에 있는 변수의 복사본(copies)에 동일하게 적용됩니다.




import tensorflow as tf
import numpy as np
import os

print(tf.__version__) # 2.0.0


# 1. 데이터 

fashion_mnist = tf.keras.datasets.fashion_mnist # dataset load

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


print("x_train.shape :", x_train.shape) # (60000, 28, 28)
print("x_test.shape :", x_test.shape) # (10000, 28, 28)
print("y_train.shape :", y_train.shape) # (60000,)
print("y_test.shape :", y_test.shape) # (10000,)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[1], 1).astype('float32')/255
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[1], 1).astype('float32')/255


# 2. 모델 구성

# 분산 전략을 정의
strategy = tf.distribute.MirroredStrategy()
print ('장치의 수: {}'.format(strategy.num_replicas_in_sync)) # 1


# 입력 파이프라인 설정
BUFFER_SIZE = len(x_train)
# shuffle()의 parameter : buffer_size
# shuffle()은 해당 dataset의 원소를 랜덤하게 셔플해주는 함수
# 개, 고양이 이미지 분류 같은 경우에 shuffle을 하지 않으면 고양이의 이미지만 train으로 쓰게 된다거나 하는 문제가 발생
# buffer_size는 몇으로 설정?
# buffer_size 는 20000 이상으로 설정하거나 사전에 파일이름과 라벨을 shuffle해서 dataset을 만들어야 한다
# 또한 모든 파일이름과 라벨을 메모리에 저장하는 것은 큰 문제가 아니므로 buffer_size = len(filenmaes)을 사용할 수도 있다
# 여기서 train용 데이터는 x_train이므로 len(x_train)으로 설정해줌
# 이미지를 읽고 프로세싱하고 배치작업 등의 무건운 작업을 하기 전에 tf.data.Dataset.shuffle()을 호출해야하는 것을 명심해야 한다!

BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

EPOCHS = 10

with strategy.scope():

  train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE) 
  # tf.data.Dataset.from_tensor_slices 함수는 tf.data.Dataset을 생성하는 함수로 입력된 텐서로부터 slices를 생성한다
  # 예를 들어 MNIST의 학습데이터(60000, 28, 28)가 입력되면 60000개의 slices로 만들고 각각의 slice는 28*28의 이미지 크기를 갖게 된다
  train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
  
  test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(GLOBAL_BATCH_SIZE) 
  test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)


def create_model():
    model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

    return model


# 체크포인트들을 저장하기 위해서 체크포인트 디렉토리를 생성
checkpoint_dir = './assignment/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")


with strategy.scope():
  # reduction을 `none`으로 설정
  # 축소를 나중에 하고, GLOBAL_BATCH_SIZE로 나눌 수 있다
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
      reduction=tf.keras.losses.Reduction.NONE)
  # 또는 loss_fn = tf.keras.losses.sparse_categorical_crossentropy를 사용해도 된다
  def compute_loss(labels, predictions):
    per_example_loss = loss_object(labels, predictions)
    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)


with strategy.scope():
  test_loss = tf.keras.metrics.Mean(name='test_loss')

  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='train_accuracy')
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='test_accuracy')


# 모델과 옵티마이저는 `strategy.scope`에서 만들어져야한다
with strategy.scope():
    model = create_model()

    optimizer = tf.keras.optimizers.Adam()

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)


with strategy.scope():
  def train_step(inputs):
    images, labels = inputs

    with tf.GradientTape() as tape:
      predictions = model(images, training=True)
      loss = compute_loss(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_accuracy.update_state(labels, predictions)
    return loss 

  def test_step(inputs):
    images, labels = inputs

    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss.update_state(t_loss)
    test_accuracy.update_state(labels, predictions)


with strategy.scope():
  # `experimental_run_v2`는 주어진 계산을 복사하고, 분산된 입력으로 계산을 수행
  
  @tf.function
  def distributed_train_step(dataset_inputs):
    per_replica_losses = strategy.experimental_run_v2(train_step,
                                                      args=(dataset_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                           axis=None)
 
  @tf.function
  def distributed_test_step(dataset_inputs):
    return strategy.experimental_run_v2(test_step, args=(dataset_inputs,))

  for epoch in range(EPOCHS):
    # 훈련 루프
    total_loss = 0.0
    num_batches = 0
    for x in train_dist_dataset:
      total_loss += distributed_train_step(x)
      num_batches += 1
    train_loss = total_loss / num_batches

    # 테스트 루프
    for x in test_dist_dataset:
      distributed_test_step(x)

    if epoch % 2 == 0:
      checkpoint.save(checkpoint_prefix)

    template = ("epoch {}, train_loss: {}, train_acc: {}, test_loss: {}, "
                "test_acc: {}")
    print (template.format(epoch+1, train_loss,
                           train_accuracy.result()*100, test_loss.result(),
                           test_accuracy.result()*100))

    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()    


