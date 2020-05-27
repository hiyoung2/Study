# CNN, Convolutional Neural Network

# CNN은 모델이 직접 이미지, 비디오, 텍스트 또는 사운드를 분류하는 머신 러닝의 한 유형인 딥러닝에 가장 많이 사용되는 알고리즘
# CNN은 이미지에서 객체, 얼굴, 장면을 인식하기 위해 패턴을 찾는데 특히 유용하다
# CNN은 데이터에서 직접 학습하며 패턴을 사용하여 이미지를 분류하고 특징을 수동으로 추출할 필요가 없다(오!~)
# 자율 주행 자동차, 얼굴 인식 어프리케이션과 같이 객체 인식과 컴퓨터 비전이 필요한 분야에서 cnn을 많이 사용
# 응용 분야에 따라 cnn을 처음부터 만들 수도 있고 데이터셋으로 사전 학습된 모델을 사용할 수 있다

# CNN은 합성곱, convolution 연산을 사용하는 ann의 한 종류
# convolution을 사용하면 3차원 데이터의 공간적 정보를 유지한 채 다음 레이어로 보낼 수 있다

# 대표적인 cnn으로 LeNet, AlexNet이 있다
# VGG< GoogLeNet, ResNet 등은 층을 더 깊게 쌓은 cnn 기반의 심층 신경망(dnn)이다

# CNN은 기존 fully connected nerual network와 비교하여 다음과 같은 차별성을 가진다
# - 각 레이어의 입출력 형상 유지
# - 이미지의 공간 정보를 유지하면서 인접 이미지와의 특징을 효과적으로 인식
# - 복수의 필터로 이미지의 특징 추출 및 학습
# - 추출한 이미지의 특징을 모으고 강화하는 Pooling 레이어
# - 필터를 공유 파라미터로 사용하기 때문에 일반 신경망과 비교하여 학습 파라미터가 매우 적음

# CNN은 이미지의 특징을 추출하는 부분과 클래스를 분류하는 부분으로 나눌 수 있다
# 특징 추출영역은 필수 요소인 Convolution Layer와 선택요소인 Pooling Layer를 여러겹 쌓는 형태로 구성된다
# cnn 마지막 부분에 이미지 분류를 위한 Fully Connected layer가 추가 된다
# 이미지의 특징을 추출하는 부분과 이미지를 분류하는 부분 사이에 이미지 형태의 데이터를
# 배열 형태로 만드는 Flatten layer가 위치한다

# CNN은 이미지 특징 추출을 위하여 입력 데이터를 필터가 순회하며 합성곱을 계산하고
# 그 계산 결과를 이용하여 Feature map을 만든다
# Convolution Layer는 Filter의 크기, Stride, Padding 적용 여부, Max Pooling 크기에 따라서
# 출력 데이터의 Shape이 변경된다

# 1. 용어 정리
# 1.1 합성곱 Convolutiono
# 합성곱 연산은 두 함수 f, g 가운데 하나의 함수를 반전(reverse), 전이(shift) 시킨 다음,
# 다른 하나의 함수와 곱한 결과를 적분하는 것을 의미한다

# 1.2 채널 Channel
# 이미지 픽셀 하나하나는 실수이다
# 컬러 사진은 천연색을 표현하기 위해 각 픽셀을 RGB 3개의 실수로 표현한 3차원 데이터이다
# 컬러 이미지는 3개의 채널로 구성된다
# 반면 흑백 명암만을 표현하는 흑백 사진은 2차원 데이터로 1개 채널로 구성된다
# 높이가 39픽셀, 폭이 31픽셀인 컬러 사진 데이터의 shape은 (39, 31, 3)으로 표현한다
# 반면 높이가 39픽셀, 폭이 31인 흑백 사진의 데이터 shape은 (39, 31, 1)이다
# Convolution Layer에 유입되는 입력 데이터에는 한 개 이상의 필터가 적용된다
# 1개 필터는 Feature Map의 채널이 된다
# Convolution Layer에 n개의 필터가 적용된다면 출력 데이터는 n개의 채널을 갖게 된다

# 1.3 필터, Filter & Stride
# 필터는 이미지의 특징을 찾아내기 위한 공용 파라미터이다
# Filter를 Kernel이라 부르기도 한다
# CNN에서 Filter와 Kernel은 같은 의미이다
# 필터는 일반적으로 (4, 4)이나 (3, 3)과 같은 정사각 행렬로 정의된다
# CNN에서 학습의 대상은 필터 파라미터이다
# 입력 데이터를 지정된 간격으로 순회하며 채널별로 합성곱을 하고 모든 채널을 합성곱의 합을 Feature Map을 만든다
# 필터는 입력 데이터를 지정한 간격으로 순회하면서 합성곱을 계산한다
# 여기서 지정된 간격으로 필터를 순회하는 간격을 Stride라고 한다
# 입력 데이터가 여러 채널을 갖을 경우 필터는 각 채널을 순회하며 합성곱을 계산한 후 채널별 피처 맵을 만든다
# 그리고 각 채널의 피처 맵을 합산하여 최종 피처 맵으로 변환한다
# 입력 데이터는 채널 수와 상관없이 필터 별로 1개의 피ㅓ 맵이 만들ㅇ러진다
# 하나의 Convolution Layer에 크기가 같은 여러 개의 필터를 적용할 수 있다
# 이 경우에 Feature Map에는 필터 갯수 만큼의 채널이 만들어진다
# Convolution layer의 입력 데이터를 필터가 순회하며 합성곱을 통해서 만든 출력을
# Feature Map 또는 Activation Map이라고 한다
# Feature Map은 합성곱으로 만들어진 행렬이다
# Activation Map은 Feature Map 행렬에 활성화 함수를 적용한 결과이다
# 즉, Convolution layer의 최종 출력 결과가 Activation Map이다

# 1.4 패딩, Padding
# Convolution layer에서 Filter와 Stride의 작용으로 Feature Map 크기는 입력 데이터보다 작다
# Convolution layer의 출력 데이터가 줄어드는 것을 방지하는 방법이 Padding이다
# Padding은 입력 데이터의 외곽에 지정된 픽셀만큼 특정 값을 채워 넣는 것을 의미
# 보통 패딩 값으로 0을 채워 넣는다

# 1.5 Pooling Layer
# Pooling Layer는 Convolution Layer의 출력 데이터를 입력으로 받아서 
# 출력 데이터(Activation Map)의 크기를 줄이거나 특정 데이터를 강조하는 용도로 사용
# Pooling Layer를 처리하는 방법으로는 Max Pooling, Average Pooling, Min Pooling 이 있다
# 정사각 행렬의 특정 영역 안에 값이 최대값을 모으거나 특정 영역의 평균을 구하는 방식으로 동작한다
# 일반적으로 Pooling 크기와 Stride를 같은 크기로 설정하여 모든 원소가 한 번씩 처리 되도록 설정한다
# Pooling Layer는 Convolution Layer와 비교하여 다음과 같은 특징이 있다
# - 학습대상 파라미터가 없다
# - Pooling Layer를 통과하면 행렬의 크기 감소
# - Pooling Layer를 통해서 채널 수 변경 없음
# - 입력 데이터의 변화에 민감하게 반응하지 않는다(Robustness)
# - 채널 수 변경 없음 해석 : 
#   Convolution Layer에서 각 필터 채널을 적용한 결과 채널들을 다 더해야 하나의 output 채널이 되지만
#   Pooling Layer에서는 결과를 더하지 않는다
#   결과 채널이 그대로 output 채널이 되기 때문에 채널 수가 유지된다
# - 입력 데이터의 변화에 민감하게 반응하지 않는다(Robustness)
#   Pooling layer를 적용하는 목적이 여기에 있는데 내가 찾아내고자 하는 특징의 위치를 중요하게 여기기보다는
#   input이 그 특징을 포함하고 있느냐 없느냐를 판단하도록 하기 위해서 주변에 있는 값들을 뭉뚱그려서 보겠다는 것!
# CNN에서는 주로 Max Pooling을 사용한다
# 보통 Pooling의 window size와 stride는 같은 값으로 설정해서 모든 원소가 한 번씩만 연산에 참여하도록 한다
# >>>>흠,, 수업시간에 pooling size 설정할 때 stride 이야기는 없었는데 뭘까

# CNN의 네트워크 구조
# 지금까지 다룬 신경망은 이전 계층의 모든 뉴런과 결합되어 있었고 이를 Affine layer라고 불렀다(그렇구만)
# 이런 식으로 이전 계층의 모든 뉴런과 결합된 형태의 layer를 fully-connected layer(FC layer, 전결합 레이어)
# 또는 Dense layer라고 한다

# CNN에서는 FC 대신 다음 두 레이어를 활성화 함수 앞뒤에 배치한다
# - Convolution layer
# - Pooling layer
# 그러나 모두 이렇게 바뀌는 것은 아니고 출력에 가까운 층에서는 FC layer를 사용할 수 있다
# 또한 마지막 출력 계층에서는 FC-Softmax 그대로이다
# 결과적으로 다음과 같은 형태가 된다
# Conv-ReLU[-Pooling]....반복....FC-ReLu....FC-Softmax(여기서도 Softmax가???)
# * Pooling layer는 생략하기도 한다(??????)

# Convolution layer
# Convolution의 합성곱이라는 뜻이다
# 입력데이터와 가중치들의 집합체 다양한 filter와의 convolution 연산을 통해 
# 입력데이터의 특징(feature)을 추출하는 역할을 수행한다
# 합성곱은 "두 함수 중 하나를 반전(reverse)하고 이동(shift) 시켜가며 다른 하나의 함수와 곱한 결과를 적분해나간다"는 뜻

# - 출력이 Convolved Feature, CNN 에서는 합성곱 계층의 입출력 데이터를 특징맵(Feature Map)이라고 부른다
# - 위 합성곱의 정의에서 두 함수를 이미지, 필터라고 생각하면, 필터를 이동시켜가며 이미지와 곱한 결과를
#   적분(덧셈) 해 나간다는 뜻이 된다
# - 여기서 수행하는 적분(덧셈)은 단일 곱셈-누산(fused multiply-add-FMA)이라 한다
# - 행렬에서 반전(reverse)에 대응하는 것은 플리핑(flipping)인 듯 하나
#   flipping 하면 합성곱 연산이고 하지 않으면 교차상관 연산인데 딥러닝에선 잘 구분하지 않는다
#   flipping 여부를 인수로 받기도 한다
# - 필터를 커널이라 부르기도 한다
# - 편향(bias)는 필터를 적용한 결과 데이터에 더해진다 
# - 편향은 항상 1x1d이며 이 값을 모든 원소에 더한다

# 2. layer 별 출력 데이터 산정
# 2.1 Convolution 레이어 출력 데이터 크기 산정
# 입력 데이터에 대한 필터의 크기와 stride 크기에 따라서 Featrue Map 크기가 결정된다

