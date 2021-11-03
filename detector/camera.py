import cv2
from detector.server import RTSP_SCHEME, USB_CAM_SCHEME

class GStreamerString:
    def __call__(self, source: str) -> str:
        if source.startswith(RTSP_SCHEME):
            source = "rtspsrc location={} ! decodebin ! videoconvert ! appsink max-buffers=1 drop=true".format(source)
            error_message = "Could not initialize RTSP"
        elif source.startswith(USB_CAM_SCHEME):
            fmt = "format=YUY2"
            width = "width=640"
            height = "height=480"
            args = "pixel-aspect-ratio=1/1, framerate=30/1 ! videoconvert ! appsink"
            source = f"v4l2src device={source} ! video/x-raw, {fmt}, {width}, {height}, {args}"
            error_message = "Could not initialize Webcam"
        else:
            error_message = "GStreamer for file not supported"
        return source, error_message

class FrameHandler:
    """Handler for image frames.

    :param source: source of image
    :param frame_batch: number of frame batch    
    """
    def __init__(self, source: str):
        error_message = ""
        self._source = source
        try:
            gstream_source, error_message = GStreamerString(source)
            self._initialize(gstream_source)
            source = gstream_source
        except:
            print("Could not utilize GStreamer: {}".format(error_message))
            assert self._initialize(source), "Failed to retrieve frame"
        self.capture = cv2.VideoCapture(source)

    @property
    def get_source(self):
        return self._source

    def _initialize(self, source):
        """Initialize FrameHandler.

        :return: success status of initialization
        """
        try:
            self.capture = cv2.VideoCapture(source)
            ret, self.last_frame = self.capture.read()
            self.capture.release()
            return ret
        except:
            raise Exception("Can't initialize camera device")

    @property
    def frame(self):
        """Obtain frame from source.
    
        :return: frame
        """
        try:
            ret, frame = self.capture.read()
            if ret:
                self.last_frame = frame
                return frame
            else:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return self.last_frame
        except:
            raise Exception("Can't open camera device")