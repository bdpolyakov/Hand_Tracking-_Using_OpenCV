import cv2
import mediapipe as mp
import time
import os
import numpy as np
import csv
import tensorflow as tf
from timeit import default_timer as timer
import box as bbhelper

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
def get_mediapipe_boundingbox(img):
    pTime = 0
    cTime = 0

    # success, img = cap.read()
    h, w, c = img.shape

    # while True:
    # success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for id, lm in enumerate(handLms.landmark):
                # print(id,lm)
                h, w, c = img.shape

                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y

                # cx, cy = int(lm.x *w), int(lm.y*h)
                # if id ==0:
                #cv2.circle(img, (x, y), 7, (255, 0, 255), cv2.FILLED)

            #cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            #mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        #cv2.imshow("Image", img)
        #cv2.waitKey(0)
        return x_min, y_min, x_max, y_max


def sigm(x):
    return 1 / (1 + np.exp(-x) )

def get_palm_detector_bb(image):
    model = 'C:/development/OCULI/testing/tfliteanalysis/models/palm_detection_full.tflite'
    anchors_name = 'C:/development/OCULI/testing/tfliteanalysis/anchors_new.csv'

    i = 0
    time_list = list()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    start = timer()

    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load anchors
    with open(anchors_name, 'r') as csv_f:
        anchors = np.r_[[x for x in csv.reader(csv_f, quoting=csv.QUOTE_NONNUMERIC)]]

    print(np.shape(image))


    # Skip the resizing if the image has already the correct size
    if (image.shape[0] != 192 and image.shape[1] != 192):

        # Resize image while maintaining aspect ratio
        height, width, _ = image.shape
        aspect_ratio = width / height
        if aspect_ratio > 1:
            target_size = (192, int(192 / aspect_ratio))
        else:
            target_size = (int(192 * aspect_ratio), 192)
        resized_image = cv2.resize(image, target_size)

        # Pad image to square while keeping letterbox padding
        padded_image = np.zeros((192, 192, 3), dtype=np.uint8)
        pad_height = (192 - resized_image.shape[0]) // 2
        pad_width = (192 - resized_image.shape[1]) // 2
        padded_image[pad_height:pad_height + resized_image.shape[0], pad_width:pad_width + resized_image.shape[1],
        :] = resized_image
        padded_image = padded_image.astype('float32')
        padded_image /= 255.0

    else:
        padded_image = image.astype('float32')
        padded_image /= 255.0

    # Sets the value of the input tensor.
    #
    # Note this copies data in value.
    #
    # If you want to avoid copying, you can use the tensor() function
    # to get a numpy buffer pointing to the input buffer in the tflite
    # interpreter.
    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(padded_image, axis=0))

    # Invoke the interpreter.
    #
    # Be sure to set the input sizes, allocate tensors and
    # fill values before calling this. Also, note that this
    # function releases the GIL so heavy computation can be
    # done in the background while the Python interpreter continues.
    # No other function on this object should be called while the
    # invoke() call has not finished.

    interpreter.invoke()

    # Gets the value of the output tensor (get a copy).
    #
    # If you wish to avoid the copy, use tensor(). This
    # function cannot be used to read intermediate results.
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    # Add new inference time
    inference_time = timer() - start
    time_list.append(1000 * inference_time)
    i = i + 1
    print(i)

    print(f'Inference time: {np.average(time_list)} ms')

    # read the outputs using the anchors
    clf = interpreter.get_tensor(output_details[1]['index'])[0, :, 0]
    probabilities = sigm(clf)

    print("prob type", type(probabilities))
    print("max prob:", max(probabilities))
    print("probabilities[np.argmax(probabilities)]", probabilities[np.argmax(probabilities)])
    print("prob > 0.02", True in (probabilities > 0.02))
    #print("probabilities == probabilities[np.argmax(probabilities)]", True in probabilities == probabilities[np.argmax(probabilities)])

    detecion_mask = probabilities > .2 #== probabilities[np.argmax(probabilities)]
    candidate_detect = output_data[detecion_mask]
    candidate_anchors = anchors[detecion_mask]
    probabilities = probabilities[detecion_mask]

    candidate_detect[:, :2] += 192 * candidate_anchors[:, :2]

    print("true in detection mask?:", True in detecion_mask)
    print(candidate_anchors)
    print(list(enumerate(candidate_detect)))

    x_min, y_min, x_max, y_max = 0, 0, 0, 0

    all_palm_detections = []
    for idx, detect in enumerate(candidate_detect):
        (x, y, w, h) = detect[0:4].astype(np.int32)
        print("Pad width:", pad_width, "Pad Height:", pad_height)
        x_min = x - pad_width - w//2
        y_min = y - pad_height - h//2
        x_max = x + w - pad_width
        y_max = y + h - pad_height
        cv2.rectangle(resized_image, (x_min, y_min), (x_min + w, y_min + h), (0, 255, 0), 1)
        #cv2.imshow("Resized Image", resized_image)
        #cv2.waitKey()
        bb = (x_min, y_min, w, h)
        input_shape = (resized_image.shape[0], resized_image.shape[1])
        output_shape = (image.shape[0], image.shape[1])
        (x_min, y_min, new_w, new_h) = bbhelper.scale(bb, input_shape, output_shape)
        x_max = x_min + new_w
        y_max = y_min + new_h
        all_palm_detections.append([x_min, y_min, x_max, y_max])
        #print("bb?:", (x, y, w, h))
        #padded_image = cv2.rectangle(padded_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # cv2.rectangle(padded_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
    #cv2.namedWindow("Resized Image", cv2.WINDOW_NORMAL)
    #cv2.imshow("Resized Image", resized_image)
    #cv2.waitKey()
    return all_palm_detections


def get_closest_kitti_bb(label_name, x_min, y_min, x_max, y_max):
    f = open(label_name, "r")
    best_distance = -1
    for x in f:
        data = x.split()
        distance = abs(int(data[4]) - x_min) + abs(int(data[5]) - y_min) + abs(int(data[6]) - x_max) + abs(
            int(data[7]) - y_max)
        if (best_distance < 0 or distance < best_distance):
            kitti_x_min = int(data[4])
            kitti_y_min = int(data[5])
            kitti_x_max = int(data[6])
            kitti_y_max = int(data[7])
            best_distance = distance
    #print(best_distance)
    return kitti_x_min, kitti_y_min, kitti_x_max, kitti_y_max

def intersection_over_union(x_min, y_min, x_max, y_max, kitti_x_min, kitti_y_min, kitti_x_max, kitti_y_max):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(x_min, kitti_x_min)
    yA = max(y_min, kitti_y_min)
    xB = min(x_max, kitti_x_max)
    yB = min(y_max, kitti_y_max)

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (x_max - x_min + 1) * (y_max - y_min + 1)
    boxBArea = (kitti_x_max - kitti_x_min + 1) * (kitti_y_max - kitti_y_min + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def mean_iou():
    num_files = 0
    total_iou = 0
    img_directory = "C:\\Users\\bpdas\\OneDrive\\Desktop\\OCULI\\testing\\hand-detection-tutorial\\egohands_kitti_formatted\\images"
    label_directory = "C:\\Users\\bpdas\\OneDrive\\Desktop\\OCULI\\testing\\hand-detection-tutorial\\egohands_kitti_formatted\\labels"
    for filename in os.listdir(img_directory):
        img_file = os.path.join(img_directory, filename)
        filebase = filename[:-3]
        labelfilename = filebase+"txt"
        label_file = os.path.join(label_directory, labelfilename)

        # checking if it is a file
        if os.path.isfile(img_file) and os.path.isfile(label_file):

            img = cv2.imread(img_file)
            coords = get_mediapipe_boundingbox(img)
            if(coords):
                x_min, y_min, x_max, y_max = coords
                kitti_x_min, kitti_y_min, kitti_x_max, kitti_y_max = get_closest_kitti_bb(label_file ,x_min, y_min, x_max, y_max)
                iou = intersection_over_union(x_min, y_min, x_max, y_max, kitti_x_min, kitti_y_min, kitti_x_max, kitti_y_max)
                if(iou > 0):
                    total_iou += iou
                    num_files+=1
                    #print(num_files)
                    #print(total_iou)
                else:
                    print("IOU < 1")
    if(num_files > 0):
        print("mean iou:")
        mean_iou = total_iou/num_files
    return mean_iou

def drawbothbb(filename):
    img_directory = "C:\\Users\\bpdas\\OneDrive\\Desktop\\OCULI\\testing\\hand-detection-tutorial\\egohands_kitti_formatted\\images"
    label_directory = "C:\\Users\\bpdas\\OneDrive\\Desktop\\OCULI\\testing\\hand-detection-tutorial\\egohands_kitti_formatted\\labels"
    img_file = os.path.join(img_directory, filename)
    filebase = filename[:-3]
    labelfilename = filebase + "txt"
    label_file = os.path.join(label_directory, labelfilename)

    img = cv2.imread(img_file)
    coords = get_mediapipe_boundingbox(img)
    if (not coords):
        print("error in mediapipe bb!")
    x_min, y_min, x_max, y_max = coords
    kitti_x_min, kitti_y_min, kitti_x_max, kitti_y_max = get_closest_kitti_bb(label_file, x_min, y_min, x_max, y_max)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
    cv2.rectangle(img, (kitti_x_min, kitti_y_min), (kitti_x_max, kitti_y_max), (0, 255, 0), 2)
    coords = get_mediapipe_boundingbox(img)
    if (not coords):
        print("error in mediapipe bb!")
    x_min, y_min, x_max, y_max = coords
    kitti_x_min, kitti_y_min, kitti_x_max, kitti_y_max = get_closest_kitti_bb(label_file, x_min, y_min, x_max, y_max)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
    cv2.rectangle(img, (kitti_x_min, kitti_y_min), (kitti_x_max, kitti_y_max), (0, 255, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(0)

def main():
    #drawbothbb("CARDS_COURTYARD_B_T_frame_0011.jpg")
    img = cv2.imread("C:/development/OCULI/testing/hand-detection-tutorial/egohands_kitti_formatted/images/CARDS_LIVINGROOM_S_H_frame_2348.jpg") #CARDS_COURTYARD_B_T_frame_0011
    palm_x_min, palm_y_min, palm_x_max, palm_y_max = get_palm_detector_bb(img)[0]
    print("palm detector bb", palm_x_min, palm_y_min, palm_x_max, palm_y_max)
    #cv2.rectangle(img, (palm_x_min, palm_y_min), (palm_x_max, palm_y_max), (0, 255, 0), 2)

    #cv2.imshow("Image", img)
    #cv2.waitKey()
    #cv2.imshow("image", img)
    #cv2.waitKey(1)
    x_min, y_min, x_max, y_max = get_mediapipe_boundingbox(img)
    print("mp bb", x_min, y_min, x_max, y_max)
    kitti_x_min, kitti_y_min, kitti_x_max, kitti_y_max = get_closest_kitti_bb("C:/development/OCULI/testing/hand-detection-tutorial/egohands_kitti_formatted/labels/CARDS_LIVINGROOM_S_H_frame_2348.txt",x_min, y_min, x_max, y_max)
    print("label bb",kitti_x_min, kitti_y_min, kitti_x_max, kitti_y_max)
    print("IOU for MP vs label",intersection_over_union(x_min, y_min, x_max, y_max, kitti_x_min, kitti_y_min, kitti_x_max, kitti_y_max))

    cv2.rectangle(img, (palm_x_min, palm_y_min), (palm_x_max, palm_y_max), (255, 0, 0), 2)
    cv2.rectangle(img, (kitti_x_min, kitti_y_min), (kitti_x_max, kitti_y_max), (0, 255, 0), 2)

    print("IOU for palm vs label", intersection_over_union(palm_x_min, palm_y_min, palm_x_max, palm_y_max, kitti_x_min, kitti_y_min, kitti_x_max, kitti_y_max))
    cv2.imshow("Image", img)
    cv2.waitKey()
    #print(mean_iou())
    print("End of program")



main()
