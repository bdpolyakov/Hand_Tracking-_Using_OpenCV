####################################################################
# box.py
#
# Helper functions for bounding boxes.
#
# Copyright: (C) 2019-2023 FRIS Inc. (dba Oculi)
####################################################################

import cv2
import numpy as np

def draw(input, bbox, color=(0,255,0)):
    '''
    Add bounding boxes to a copy of the input image and return.
    
    Args:
        input (3D numpy array): input image
        bbox (tuple): bounding box to draw (x,y,w,h)
        color (tuple): color to use for bounding boxes (B,G,R)

    Returns:
        3D numpy array: copy of the input with bounding boxes drawn
    '''
    # Make a copy of the input
    output = input.copy()

    # Draw bounding box for single bbox tuple, else iterate over list
    # of tuples and draw each bounding box on the output image.
    if type(bbox) == tuple:
        (x,y,w,h) = bbox
        output = cv2.rectangle(output, (x,y), (x+w,y+h), color, 2)
    elif type(bbox) == list:
        for (x,y,w,h) in bbox:
            output = cv2.rectangle(output, (x,y), (x+w,y+h), color, 2)

    return output


def shade(input, bbox, color=(0,255,0)):
    '''
    Add shaded bounding boxes to a copy of the input image and return.
    
    Args:
        input (3D numpy array): input image
        bbox (tuple): bounding box to shade (x,y,w,h)
        color (tuple): color to use for bounding boxes (B,G,R)

    Returns:
        3D numpy array: copy of the input with bounding boxes shaded
    '''
    # Make a copy of the input
    overlay = input.copy()

    # Draw bounding box for single bbox tuple, else iterate over list
    # of tuples and draw each bounding box on the output image. These
    # boxes will be completely filled in with color.
    if type(bbox) == tuple:
        cv2.rectangle(overlay, (x,y), (x+w,y+h), color, -1)
    elif type(bbox) == list:
        for (x,y,w,h) in bbox:
            cv2.rectangle(overlay, (x,y), (x+w,y+h), color, -1)
   
    # Use cv2.addWeighted shade the bounding boxes rather than leave
    # them completely filled in.
    alpha = 0.2
    return cv2.addWeighted(overlay, alpha, input, 1-alpha, 0)

def scale(bbox, input_size, output_size):
    '''
    Function to scale bounding box from model input size (input_size) to 
    desired output image size. The output bounding box will then be able 
    to be used with and image of size output_size.

    Args:
        bbox (tuple): bounding box in the form (x,y,w,h)
        input_size (tuple): input size of model (rows,cols) used to generate bbox
        output_size (tuple): output size of the image (rows,cols)

    Returns:
        tuple: rescaled bounding box in the form (x,y,w,h)

    '''
    (x,y,w,h) = bbox
    scale_x = output_size[1] / input_size[1]
    scale_y = output_size[0] / input_size[0]
    return  int(scale_x * x), int(scale_y * y), int(scale_x * w), int(scale_y * h)

def check_bounds(bbox, rows, cols):
    '''
    Check that bounding box is within specified rows and columns. This
    works by forcing x / y to be greater than or equal to zero and less
    than or equal to cols / rows respectively. Afterwards, w / h are
    resized accordingly.
    
    Args:
        bbox (tuple): bounding box to check (x,y,w,h)
        rows (int): number of rows in the image
        cols (int): number of columns in the image
    Returns:
        tuple of ints: bounding box within rows and cols
    '''
    # Resize bbox to be within rows and cols
    (x,y,w,h) = bbox
    if x < 0:
        w = w + x # remove the negative pixels (x is negative)
        x = 0     # set x to 0
    elif x > cols:
        w = 0
        x = cols
    elif x + w > cols:
        w = cols-x
    
    if y < 0:
        h = h + y # remove the negative pixels (y is negative)
        y = 0     # set to 0
    elif y > rows:
        h = 0
        y = rows
    elif y + h > rows:
        h = rows-y

    return (x,y,w,h)

def check_bounds2(bbox, rows, cols):
    '''
    Check that bounding box is within specified rows and columns. This
    works by forcing x / y to be greater than or equal to zero and less
    than or equal to cols / rows respectively without modifying the 
    width or the height values.
    
    Args:
        bbox (tuple): bounding box to check (x,y,w,h)
        rows (int): number of rows in the image
        cols (int): number of columns in the image
    Returns:
        tuple of ints: bounding box within rows and cols with fixed width and height
    '''
    # Resize bbox to be within rows and cols
    (x,y,w,h) = bbox
    if x < 0:
        x = 0
    elif x + w > cols:
        x = cols - w
    
    if y < 0:
        y = 0
    elif y + h > rows:
        y = rows - h

    return (x,y,w,h)

def add_offset(bbox, offset):
    '''
    Add offset to (x,y) coordinates of the bounding box.
    
    Args:
        bbox (tuple): bounding box (x,y,w,h)
        offset (tuple): offset value (x,y)
    
    Returns:
        tuple: bounding box offset by its original (x,y) coordinates
    '''
    (x,y,w,h) = bbox
    return x+offset[0], y+offset[1], w, h

def largest_width(bbox):
    '''
    Iterate over the list of faces and return the one
    with the largest width. A face consists of (x,y,w,h).
    '''
    largest_width_bbox = (0,0,0,0)
    for (x,y,w,h) in bbox:
        if w > largest_width_bbox[2]:
            largest_width_bbox = (x,y,w,h)
    return largest_width_bbox

def resize_faces(faces, width_scale=1, height_scale=1):
    output = []
    for (x,y,w,h) in faces:
        new_x = int(x + w*(1-width_scale)/2)
        new_y = int(y + h*(1-height_scale)/2)
        new_w = int(w*width_scale)
        new_h = int(h*height_scale)
        output.append( (new_x, new_y, new_w, new_h) )
    return output

def intersection_over_union(bbox1, bbox2):
    '''
    Take two bounding boxes (x,y,w,h) and compute the intersection over
    union between them.
    '''
    # Extract bounding box parameters
    (x1,y1,w1,h1) = bbox1
    (x2,y2,w2,h2) = bbox2

    # Find intersection coordinates (top right and bottom left x/y points)
    x_inter1 = max(x1, x2)
    y_inter1 = max(y1, y2)

    x_inter2 = min(x1+w1, x2+w2)
    y_inter2 = min(y1+h1, y2+h2)

    # Find area of intersection
    width_inter  = max(0, x_inter2 - x_inter1)
    height_inter = max(0, y_inter2 - y_inter1)
    area_inter = width_inter * height_inter

    # Find area of box1 and box2
    area_box1 = w1 * h1
    area_box2 = w2 * h2

    # Find area of the union and then the intersection over union
    area_union = area_box1 + area_box2 - area_inter
    if area_union != 0:
        iou = area_inter / area_union
    else:
        iou = 0

    return iou

# class HaarCascade():
#     def __init__(self):
#         # Face Cascade Classifier
#         self.model = '../data/haarcascade_frontalface_default.xml'
#         self.face_cascade = cv2.CascadeClassifier(self.model)

#     def visualize(self, image, results, box_color=(0,255,0)):
#         output = image.copy()
#         for (x,y,w,h) in results:
#             output = cv2.rectangle(output, (x,y), (x+w,y+h), box_color, 2)
#         return output

#     def find_faces(self, image):
#         '''Run multi-scale Haar Cascade to detect faces.'''
#         return self.face_cascade.detectMultiScale(image, 1.1, 3)

# class DNN():
#     def __init__(self):
#         self.model_size = (300,300)
#         self.model = "../models/res10_300x300_ssd_iter_140000.caffemodel"
#         self.config = "../models/deploy.prototxt"
#         self.net = cv2.dnn.readNetFromCaffe(self.config, self.model)

#     def visualize(self, image, results, box_color=(0,255,0)):
#         output = image.copy()
#         bbox = self.results_to_bbox(results, image_shape=image.shape[0:2], minimum_confidence=0.5)
#         for (x,y,w,h) in bbox:
#             output = cv2.rectangle(output, (x,y), (x+w,y+h), box_color, 2)
#         return output

#     def results_to_bbox(self, results, image_shape, minimum_confidence=0.5):
#         bbox = []
#         rows,cols = image_shape
#         for i in range(results.shape[2]):
#             confidence = results[0, 0, i, 2]
#             if confidence > minimum_confidence:
#                 box = results[0, 0, i, 3:7] * np.array([cols, rows, cols, rows])
#                 (x1, y1, x2, y2) = box.astype(np.int32)
#                 w = x2 - x1
#                 h = y2 - y1

#                 bbox.append( (x1,y1,w,h) )
#         return bbox

#     def find_faces(self, image):
#         # Check image size vs. model size
#         if image.shape[0:2] != self.model_size:
#             print(f'Size mismatch. Resizing to model size {self.model_size}..')
#             image = cv2.resize(image, self.model_size)

#         blob = cv2.dnn.blobFromImage(image, 1.0, self.model_size, (104.0, 117.0, 123.0))
#         self.net.setInput(blob)
#         return self.net.forward()
