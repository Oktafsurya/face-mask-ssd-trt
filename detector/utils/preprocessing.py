import cv2
import numpy as np

def preprocess_img(input_frame):
    input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
    input_frame = cv2.resize(input_frame, (300,300), interpolation = cv2.INTER_AREA)
    input_frame = np.array(input_frame, dtype='float32', order='C')
    input_frame -= np.array([127, 127, 127])
    input_frame /= 128
    input_frame = input_frame.transpose((2, 0, 1))
    input_frame = np.expand_dims(input_frame, axis=0)
    return input_frame