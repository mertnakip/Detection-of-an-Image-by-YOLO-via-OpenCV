import cv2
import numpy as np
from Process_Result import Process

# Write down conf, nms thresholds,inp width/height
confThreshold = 0.45
nmsThreshold = 0.45
inpWidth = 416
inpHeight = 416

class_list = [0, 1, 2]

# Load names of classes and turn that into a list
classesFile = 'obj.names'
classes = None

with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Model configuration
modelConf = 'yolov3-tiny_obj.cfg'
modelWeights = 'yolov3-tiny_obj_5000.weights'

net = cv2.dnn.readNetFromDarknet(modelConf, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

proc = Process(confThreshold, class_list, nmsThreshold, classes)

image_folder = 'Image_Folder/'
features = np.zeros((1, 960))
desired = np.zeros((1, 2))

# Read image as gray-scale
img = cv2.imread(image_folder + 'image_name.jpg', cv2.IMREAD_COLOR)

blob = cv2.dnn.blobFromImage(img, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

# Set the input the the net
net.setInput(blob)
outs = net.forward(proc.getOutputsNames(net))

label_list = proc.postprocess(img, outs)

