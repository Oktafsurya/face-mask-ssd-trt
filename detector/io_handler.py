import cv2
import numpy as np
from typing import Union

class IOHandler:
    def __init__(self, infer_instance, frame_handler):
        self.infer_instance = infer_instance
        self.frame_handler = frame_handler
        self.source = self.frame_handler.get_source
        self._data = None

    def _jpeg_encoder(cls, img: np.ndarray):
        encode_param = (int(cv2.IMWRITE_JPEG_QUALITY), 90)
        _, encimg = cv2.imencode('.jpg', img, encode_param)
        return encimg.tostring()
    
    def _bgr2rgb(cls, img: Union[np.ndarray, list]):
        ret = None
        if isinstance(img, np.ndarray):
            ret = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            ret = list()
            for im in img:
                if isinstance(im, np.ndarray):
                    ret.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                else:
                   raise TypeError("[BGR2RGB] Image format is not supported")
        return ret

    def _rgb2bgr(cls, img: Union[np.ndarray, list]):
        ret = None
        if isinstance(img, np.ndarray):
            ret = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            ret = list()
            for im in img:
                if isinstance(im, np.ndarray):
                    ret.append(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
                else:
                    raise TypeError("[RGB2BGR] Image format is not supported")
        return ret

    @property
    def info(self):
        return self._data
    
    def inference(self):
        frame = self.frame_handler.frame
        frame = self._bgr2rgb(frame)
        result = self.infer_instance.process(frame)
        self._data = {"data": result["inference_data"]["data"]}
        frame = self._rgb2bgr(result["inference_data"]["frame"])
        frame = self._jpeg_encoder(frame)
        return frame