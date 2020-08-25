from flask import Flask, render_template, Response, request
# emulated camera
import cv2, numpy as np
from threading import Thread
from darknet import darknet
 
# net = darknet.load_net(b"C:/Users/bitcamp/anaconda3/Lib/site-packages/darknet/cfg/yolov3.cfg", b"C:/Users/bitcamp/anaconda3/Lib/site-packages/darknet/weight/yolov3.weights", 0) 
# meta = darknet.load_meta(b"C:/Users/bitcamp/anaconda3/Lib/site-packages/darknet/data/coco.data") 
# cap = cv2.VideoCapture("C:/Users/bitcamp/anaconda3/Lib/site-packages/darknet/26-4_cam01_assault01_place01_night_spring.mp4") 
 
net = darknet.load_net(b"C:/Users/bitcamp/anaconda3/Lib/site-packages/darknet/cfg/yolov3.cfg", b"C:/Users/bitcamp/anaconda3/Lib/site-packages/darknet/weight/yolov3.weights", 0) 
meta = darknet.load_meta(b"C:/Users/bitcamp/anaconda3/Lib/site-packages/darknet/data/darknettest.data") 
cap = cv2.VideoCapture("C:/Users/bitcamp/anaconda3/Lib/site-packages/darknet/data/6-1_cam01_assault01_place03_night_spring.mp4") 

class WebcamVideoStream:
       def __init__(self, src=0):
           print("init")
           self.stream = cv2.VideoCapture(src)
           (self.grabbed, self.frame) = self.stream.read()
 
           self.stopped = False
 
       def start(self):
           print("start thread")
           t = Thread(target=self.update, args=())
           t.daemon = True
           t.start()
           return self
 
       def update(self):
           print("read")
           while True:
               if self.stopped:
                   return
 
               (self.grabbed, self.frame) = self.stream.read()
 
       def read(self):
           return self.frame
 
       def stop(self):
           self.stopped = True
 
video = [(1,"C:/Users/bitcamp/anaconda3/Lib/site-packages/darknet/data/6-1_cam01_assault01_place03_night_spring.mp4"),(2,"C:/Users/bitcamp/anaconda3/Lib/site-packages/darknet/data/411-5_cam01_assault01_place08_night_spring.mp4")]
app = Flask(__name__)
 
@app.route('/')
def run():
    # conn = sqlite3.connect('./data/wanggun.db')
    # c = conn.cursor()
    # c.execute("SELECT * FROM general;")
    rows = video#c.fetchall()
    return render_template("./project/template/camera_index.html", rows=rows)
 
@app.route('/index')
def index():
        ids = request.args.get('id')
        print(ids)
        rows = video
        return render_template('./project/template/camera.html', rows=[rows[int(ids)-1]])
 
 
def gen(camera):
        """Video streaming generator function."""
        while True:
            image = camera.read()
            image = cv2.resize(image, dsize=(640, 480), interpolation=cv2.INTER_AREA)
            frame = darknet.nparray_to_image(image)
            r = darknet.detect_image(net, meta, frame, thresh=.5, hier_thresh=.5, nms=.45, debug= False)
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
                cv2.line(image, (top + int(w / 2), left), (top + int(w / 2), left + int(h)), (0,255,0), 3) 
                cv2.line(image, (top, left + int(h / 2)), (top + int(w), left + int(h / 2)), (0,255,0), 3) 
                cv2.circle(image, (top + int(w / 2), left + int(h / 2)), 2, tuple((0,0,255)), 5)
 
            ret, jpeg = cv2.imencode('.jpg', image)
            darknet.free_image(frame)
            # print("after get_frame")
            if jpeg is not None:
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            else:
                print("frame is none")
 
 
 
@app.route('/index/video_feed')
def video_feed():
        ids = request.args.get('id')
        print(ids)
        """Video streaming route. Put this in the src attribute of an img tag."""
        return Response(gen(WebcamVideoStream(src=video[int(ids)-1][1]).start()),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
 
 
if __name__ == '__main__':
       app.run(host='192.168.0.128', debug=True, threaded=True)


#  192.168.0.128