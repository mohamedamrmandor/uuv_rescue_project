#!/usr?bin?env python3
import cv2 as cv
import numpy as np
cap = cv.VideoCapture(r"[the path of the video]/what the auv see.mp4", cv.CAP_GSTREAMER)
whT = 320
confThreshold =0.1
nmsThreshold= 0.2
#### LOAD MODEL
## Coco Names
classesFile = "[the path of the workspace]/catkin_ws/src/uuv_rescue_project/scripts/coco(1).names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('n')
print(classNames)
## Model Files

net = cv.dnn.readNetFromDarknet("[the path of the workspace]/catkin_ws/src/uuv_rescue_project/scripts/yolo_custom.cfg", "[the path of the workspace]/catkin_ws/src/uuv_rescue_project/scripts/yolov3_training_last.weights")
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

def findObjects(outputs,img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    if indices == True:
     for i in indices:
        box = bbox[i[0]]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # print(x,y,w,h)
        cv.rectangle(img, (x, y), (x+w,y+h), (255, 0 , 255), 2)
        cv.putText(img,"Human",
                  (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

while True:
    success, img = cap.read()
    blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    findObjects(outputs,img)

    cv.imshow('Image', img)
    if cv.waitKey(0) and 0xFF ==  ord('q'):
     break
cap.release()
cv.destroyAllWindows()
