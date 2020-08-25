from darknet import darknet

print("complete")


import cv2, numpy as np
from ctypes import *

# load_net : cfg(configure, 이미 구성된 모델, 예를 들면 이미지 분석 모델의 vgg16과 같은 것)파일과 weights 파일을 로드해줌
# load_meta : train or test에 필요한 data 내용(모델에 돌릴 파일 리스트 텍스트파일의 위치, validation 파일 위치, class name을 지정해둔 name 파일의 위치 등 정보)이 담긴 data파일을 로드
# python2에서는 그냥 " " 안에 넣어주고 python3에서는 " "앞에 b를 붙여서 사용한다

net = darknet.load_net(b"C:/Users/bitcamp/anaconda3/Lib/site-packages/darknet/cfg/yolov3.cfg", b"C:/Users/bitcamp/anaconda3/Lib/site-packages/darknet/weight/yolov3.weights", 0) 
meta = darknet.load_meta(b"C:/Users/bitcamp/anaconda3/Lib/site-packages/darknet/data/darknettest.data") 
cap = cv2.VideoCapture("C:/Users/bitcamp/anaconda3/Lib/site-packages/darknet/data/6-1_cam01_assault01_place03_night_spring.mp4") 


print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
while(cap.isOpened()):
    ret, image = cap.read() 
    image = cv2.resize(image, dsize=(480, 640), interpolation=cv2.INTER_AREA)
    print(image.shape)
    if not ret: 
        break 
    frame = darknet.nparray_to_image(image)
    r = darknet.detect_image(net, meta, frame) 
 
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
        # cv2.line(image, (top + int(w / 2), left), (top + int(w / 2), left + int(h)), (0,255,0), 3) 
        # cv2.line(image, (top, left + int(h / 2)), (top + int(w), left + int(h / 2)), (0,255,0), 3) 
        # cv2.circle(image, (top + int(w / 2), left + int(h / 2)), 2, tuple((0,0,255)), 5)

    darknet.free_image(frame) # (c언어 개념 : 동적메모리 할당)할당된 메모리를 해제해주는 코드 -> 쌓이는 메모리를 풀어준다?
    cv2.imshow('frame', image) 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
 
cap.release()


