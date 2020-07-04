# vgg16 : layer 16개
# 이미지 분석 시 가져다 쓸 수 있음
# 이미지넷에서 준우승 한 모델? 
# 우승 모델 가져다 쓰기 파일?
# 전이학습 : 잘 만든 모델 가져와서 쓰기

# keras.applications : 사전 훈련된 여러 네트워크를 제공

from keras.applications import VGG16, VGG19, Xception, ResNet101, ResNet101V2, ResNet152, ResNet152V2, ResNet50, ResNet50V2
from keras.applications import InceptionV3, InceptionResNetV2, MobileNet, MobileNetV2, DenseNet121, DenseNet169, DenseNet201
from keras.applications import NASNetLarge, NASNetMobile

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Activation, Dropout
from keras.optimizers import Adam

# model = VGG16()
# model = VGG19()
# model = Xception()
# model = ResNet101()
# model = ResNet101V2()
# model = ResNet152()
# model = ResNet152V2()
# model = ResNet50()
# model = ResNet50V2()
# model = InceptionV3()
# model = InceptionResNetV2()
# model = MobileNet()
# mdoel = MobileNetV2()
# model = DenseNet121()
# model = DenseNet169()
# model = DenseNet201()
# model = NASNetLarge()
# modle = NASNetMobile()


# # 2. 모델 구성
vgg16 = VGG16(weights = 'imagenet', include_top = False, input_shape = (224, 224, 3)) # (None, 224, 224, 3)
# vgg16.summary()

model = Sequential()

model.add(vgg16)
model.add(Flatten())
model.add(Dense())
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10, activation = 'softmax'))
model.summary()