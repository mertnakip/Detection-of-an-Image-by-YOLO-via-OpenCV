import numpy as np
import cv2 as cv


class Process:
    def __init__(self, conf_threshold, label_classes, nms_threshold, all_classes):
        self.confThreshold = conf_threshold
        self.class_list = label_classes
        self.nmsThreshold = nms_threshold
        self.classes = all_classes

    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        classIDs = []
        confidences = []
        boxes = []

        label_list = list()
        compare_list = list()
        comp_score = list()

        for out in outs:
            for detection in out:

                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > self.confThreshold:
                    centerX = int(detection[0] * frameWidth)
                    centerY = int(detection[1] * frameHeight)

                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)

                    left = int(centerX - width / 2)
                    top = int(centerY - height / 2)

                    classIDs.append(classID)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

                    comp_score.append([classID, confidence])
                    compare_list.append([classID, width * height])
                    label_list.append(
                        str(classID) + ' ' + str(centerX) + ' ' + str(centerY) + ' ' + str(width) + ' ' + str(
                            height) + '\n')

        compare_array = np.array(comp_score)
        if len(compare_array) > 0:
            for clas in self.class_list:
                index = compare_array[:, 0] == clas
                try:
                    ind = np.argmax(compare_array[index], axis=0)
                    a = compare_array[index]
                    a[ind, 1] = 0
                    clas_lab = np.array(label_list)
                    clas_lab = clas_lab[index]

                    # with open(label_name, 'a') as f:
                    #    f.write(clas_lab[ind[1]])
                except:
                    pass
                # ind = np.argmax(a, axis=0)
                # print(ind[1])
                # print(clas_lab[ind[1]])
                # if a.shape[0]>1:
                #    with open(label_name, 'a') as f:
                #        f.write(clas_lab[ind[1]])
        indices = cv.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        # print(indices)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]


            self.drawPred(frame, classIDs[i], confidences[i], left, top, left + width, top + height)
        return label_list

    def getOutputsNames(self, net):
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()

        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 2)

        label = '%.2f' % conf

        # Get the label for the class name and its confidence
        if self.classes:
            assert (classId < len(self.classes))
            label = '%s:%s' % (self.classes[classId], label)

        # A fancier display of the label from learnopencv.com
        # Display the label at the top of the bounding box
        # labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # top = max(top, labelSize[1])
        # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
        # (255, 255, 255), cv.FILLED)
        # cv.rectangle(frame, (left,top),(right,bottom), (255,255,255), 1 )
        # cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def getOutputsNames(self, net):
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()

        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Set up the net

    def decision_maker(self, labels):
        labels_in_im = list()
        names = list()
        for label in labels:
            splitted_label = label.split()
            class_ID = int(splitted_label[0])
            labels_in_im.append(self.classes[class_ID].split('_'))

            if labels_in_im[-1][0] not in names:
                names.append(labels_in_im[-1][0])

        flag = {name: 0 for name in names}

        for name in names:
            check_list = ['0', '0', '0']
            exist_property = list()
            for properties in labels_in_im:
                if (properties[0] == name) and (properties[1] not in exist_property):
                    exist_property.append(properties[1])

                    if properties[1] == 'Plate':
                        check_list[0] = '1'
                    elif properties[1] == 'Face':
                        check_list[1] = '1'
                    elif properties[1] == 'Vehicle':
                        check_list[2] = '1'

            flag[name] = int(check_list[0] + check_list[1] + check_list[2], 2)
        return names, flag
