import cv2
from detector.utils.helper import predict

def postprocess_img(img, out_detect, out_scores, label_list):
    
    out_detect = out_detect.reshape((3000,4))
    out_scores = out_scores.reshape((3000,len(label_list)))

    WIDTH = img.shape[0]
    HEIGH = img.shape[1]
    IOU_THRESH = 0.7

    boxes, labels, probs = predict(WIDTH, HEIGH, out_scores, out_detect, IOU_THRESH)
    
    label_each_frame = []
    score_each_frame = []
    counter = str(boxes.shape[0])

    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        label = labels[0]
    
        label_each_frame.append(str(label_list[label]))
        score_each_frame.append(str(probs[i]))

        x1, y1, x2, y2 = box
        color = (0,255,0) if label == 1 else (0,0,255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (x1, y2 - 20), (x2, y2), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        text = f"{label_list[label]}: {round(probs[i]*100, 3)}%"
        cv2.putText(img, text, (x1 + 6, y2 - 6), font, 0.3, (255, 255, 255), 1)
    
    json_array = {"class_str": label_each_frame,
                	"score": score_each_frame,
                    "counter": counter}

    postprocess_data = {'inference_data': {'frame':img, 'data':json_array}}
    return postprocess_data