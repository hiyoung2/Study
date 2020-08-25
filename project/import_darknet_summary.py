from darknet import darknet

print("complete")


import cv2, numpy as np
from ctypes import *

# load_net : cfg(configure, 이미 구성된 모델, 예를 들면 이미지 분석 모델의 vgg16과 같은 것)파일과 weights 파일을 로드해줌
# load_meta : train or test에 필요한 data 내용(모델에 돌릴 파일 리스트 텍스트파일의 위치, validation 파일 위치, class name을 지정해둔 name 파일의 위치 등 정보)이 담긴 data파일을 로드
# python2에서는 그냥 " " 안에 넣어주고 python3에서는 " "앞에 b를 붙여서 사용한다

net = darknet.load_net(b"C:/Users/bitcamp/anaconda3/Lib/site-packages/darknet/cfg/yolov3.cfg", b"C:/Users/bitcamp/anaconda3/Lib/site-packages/darknet/weight/yolov3.weights", 0) 
meta = darknet.load_meta(b"C:/Users/bitcamp/anaconda3/Lib/site-packages/darknet/data/darknettest.data") 

# cv2.VideoCapture 메서드를 통해 테스트할 동영상을 불러 온다
cap = cv2.VideoCapture("C:/Users/bitcamp/anaconda3/Lib/site-packages/darknet/data/6-1_cam01_assault01_place03_night_spring.mp4") 


print(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # frame의 너비
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # frame의 높이


while(cap.isOpened()):
    ret, image = cap.read() # 영상을 한 프레임씩 읽는다, 제대로 읽으면 ret값이 True, 실패하면 False, image : 읽은 프레임
    image = cv2.resize(image, dsize=(480, 640), interpolation=cv2.INTER_AREA)
    
    # Scaling : 이미지 사이즈가 변하는 것
    # opencv에서는 cv2.resize() 함수를 사용하여 적용할 수 있다
    # 사이즈가 변하면 pixel 사이의 값을 결정해야하는데, 이 때 사용하는 것을 보간법이라고 한다
    # interpolation : 보간법
    # 많이 사용되는 보간법은 사이즈를 줄일 때는 cv.INTER_AREA, 사이즈를 크게 할 때는 cv2.INTER_CUBIC, cv2.INTER_LINEAR을 사용한다
    
    # cv2.resize(img, dsize, fx, fy, interpolation)
    # parameters
    # img : Image
    # dsize : Manual Size / 가로, 세로 형태의 tuple 모양
    # fx : 가로 사이즈의 배수, 2배로 크게 하려면 2. / 반으로 줄이려면 0.5
    # fy : 세로 사이즈의 배수
    # interpolation : 보간법

    print(image.shape)
    if not ret: 
        break 
    frame = darknet.nparray_to_image(image)

    # def nparray_to_image(img):
    #     data = img.ctypes.data_as(POINTER(c_ubyte))
    #     image = ndarray_image(data, img.ctypes.shape, img.ctypes.strides)

    # return image

    # ndarray.ctypes : an object to simplify the interaction of the array with the ctypes module
    # == cytpes 모듈과 배열의 상호작용을 단순화? 쉽게 해주는 객체
    # This attribute creates an object that makes it easier to use arrays when calling shared libraries with the ctypes module. 
    # The returned object has, among others, data, shape, and strides attributes (see Notes below) which themselves return ctypes objects 
    # that can be used as arguments to a shared library.

    # ctypes.data_as(self, obj)


    r = darknet.detect_image(net, meta, frame) 
    print(r)
 
    boxes = [] 
 
    for k in range(len(r)): 
        width = r[k][2][2] 
        height = r[k][2][3] 
        center_x = r[k][2][0] 
        center_y = r[k][2][1] 
        bottomLeft_x = center_x - (width / 2) 
        bottomLeft_y = center_y - (height / 2) 
        x, y, w, h = bottomLeft_x, bottomLeft_y, width, height 
        boxes.append((x, y, w, h))
 
    for k in range(len(boxes)): 
        x, y, w, h = boxes[k] 
        top = max(0, np.floor(x + 0.5).astype(int)) 
        left = max(0, np.floor(y + 0.5).astype(int)) 
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int)) 
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int)) 
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2) 

        # 아래는 사람 잡을 때 축이랑 중심점 잡아주는 것
        # cv2.line(image, (top + int(w / 2), left), (top + int(w / 2), left + int(h)), (0,255,0), 3) 
        # cv2.line(image, (top, left + int(h / 2)), (top + int(w), left + int(h / 2)), (0,255,0), 3) 
        # cv2.circle(image, (top + int(w / 2), left + int(h / 2)), 2, tuple((0,0,255)), 5)

    darknet.free_image(frame) # (c언어 개념 : 동적메모리 할당)할당된 메모리를 해제해주는 코드 -> 쌓이는 메모리를 풀어준다?
    cv2.imshow('frame', image) 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
 
cap.release()


