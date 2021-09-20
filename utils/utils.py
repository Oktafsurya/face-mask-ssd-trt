import numpy as np 
import cv2


def preprocess_img(input_frame, input_layer:str):

    preprocessed_input = {input_layer: None}
    input_frame = cv2.resize(input_frame, (300,300), interpolation = cv2.INTER_AREA)
    input_frame = np.array(input_frame, dtype='float32', order='C')
    input_frame -= np.array([127, 127, 127])
    input_frame /= 128
    input_frame = input_frame.transpose((2, 0, 1))

    print('[INFO] input_frame.shape:', input_frame.shape)
    
    preprocessed_input[input_layer] = input_frame
    return {'preprocessed_input': preprocessed_input}

def box_to_xyxy(bboxes, width, heigh):
    #x1 = (bboxes[1] - bboxes[3]/2)*width
    #y1 = (bboxes[0] - bboxes[2]/2)*heigh
    #x2 = (bboxes[3] + bboxes[3]/2)*width
    #y2 = (bboxes[2] + bboxes[2]/2)*heigh

    x1 = bboxes[:,1]*width
    y1 = bboxes[:,0]*heigh
    x2 = bboxes[:,3]*width
    y2 = bboxes[:,2]*heigh

    #w_new = x2 - x1
    #h_new = y2 - y1
    
    #x1 = x1 - w_new/2
    #y1 = y1 + h_new/2
    #x2 = x2 - w_new/2
    #y2 = y2 + h_new/2
    
    return np.array([x1,y1,x2,y2]).T

def nms(bounding_boxes, confidence_score, conf_scores_idx, overlap_thres, conf_thres):
    
    if len(bounding_boxes) == 0:
        return [], []

    boxes = np.array(bounding_boxes)

    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    score = np.array(confidence_score)

    picked_boxes = []
    picked_score = []
    picked_score_idx = []

    areas = (end_x - start_x + 1) * (end_y - start_y + 1)
    order = np.argsort(score)

    while order.size > 0:

        index = order[-1]

        if confidence_score[index] > conf_thres:
            picked_boxes.append(bounding_boxes[index])
            picked_score.append(confidence_score[index])
            picked_score_idx.append(conf_scores_idx[index])

        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h
  
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)
        left = np.where(ratio < overlap_thres)
        order = order[left]     

    return np.array(picked_boxes), np.array(picked_score), np.array(picked_score_idx)


def postprocess_img(img, out_detect, out_scores, label_list):

    out_detect = out_detect.reshape((3000,4))
    out_scores = out_scores.reshape((3000,len(label_list)))
    out_scores_ori = np.amax(out_scores_ori[:,1:], axis=1)
    out_scores_indices = np.argsort(out_scores_ori[:,1:], axis=1)[:, len(label_list)-2]
    label_list = label_list[1:]

    WIDTH = img.shape[0]
    HEIGH = img.shape[1]
    THRESHOLD = 0.6
    OVERLAP_THRES = 0.6

    out_detect = box_to_xyxy(out_detect, WIDTH, HEIGH)
    out_detect, out_scores, out_scores_indices = nms(out_detect, out_scores, out_scores_indices, OVERLAP_THRES,  THRESHOLD)

    for idx in range(out_scores.shape[0]):
        out_detect_idx = out_detect[idx]
        out_scores_idx = out_scores[idx]
        out_scores_max_idx = out_scores_indices[idx]
        label_idx = label_list[out_scores_max_idx]
        print('out_scores_idx:', out_scores_idx)
        print('out_scores_max_idx:', out_scores_max_idx)
        print('label_idx:', label_idx)
	
        if out_scores_idx > THRESHOLD:
            text = '{} - {}%'.format(label_idx, round(out_scores_idx*100., 3))
            print('text:', text)
            start_point = (int(out_detect_idx[0]), int(out_detect_idx[1]))
            pos_txt = (int(out_detect_idx[0]+10), int(out_detect_idx[1]+40))
            end_point = (int(out_detect_idx[2]), int(out_detect_idx[3]))
            color = (255, 0, 0)
            thickness = 2
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontscale = 1
            line = cv2.LINE_AA
            img = cv2.rectangle(img, start_point, end_point, color, thickness)
            img = cv2.putText(img, text, pos_txt, font, fontscale, color, thickness, line)

    frame, inference_data = img, out_detect
    return {'inference_data': {'frame':frame, 'data':inference_data}}




    