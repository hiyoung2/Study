import keras
import tensorflow as tf

# version 확인
print(keras.__version__) # 2.3.1
print(tf.__version__) # 2.0.0

# gpu 사용중인지 확인
# 모두 True가 나오면 된다
print(tf.test.is_built_with_cuda()) # True
print(tf.test.is_gpu_available(cuda_only = False, min_cuda_compute_capability = None)) # True