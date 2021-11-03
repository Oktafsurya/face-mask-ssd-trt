import tensorrt as trt
from detector.utils import trt_utils
from detector.utils import preprocessing
from detector.utils import postprocessing

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

class InferenceHandler:
    def __init__(self, engine_path, labels_path):
        self.engine_path = engine_path
        self.labels_path = labels_path
        self.load()

    def load(self):
        self.engine = self.get_engine(self.engine_path)
        self.labels = self.read_label_file(self.labels_path)
        self.context = self.engine.create_execution_context()

    def get_engine(self, engine_file_path):
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    
    def read_label_file(self, labels_file_path):
        print("Reading labels from file {}".format(labels_file_path))
        with open(labels_file_path, "r") as file:
            readline = file.read().splitlines()
            return readline

    def predict(self, input_img):
        inputs, outputs, bindings, stream = trt_utils.allocate_buffers(self.engine)
        inputs[0].host = input_img.ravel()
        outputs = trt_utils.do_inference_v2(
            self.context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        )
        return outputs

class InferenceInstance:
    def __init__(self, infer_handler):
        self.infer_handler = infer_handler       

    def _preprocessing(self, raw_img):
        return preprocessing.preprocess_img(raw_img)
    
    def _predict(self, img):
        out = self.infer_handler.predict(img)
        return out

    def _postprocessing(self, inp, out):
        return postprocessing.postprocess_img(inp, out[1], out[0], self.infer_handler.labels)

    def process(self, inp):
        norm_img = self._preprocessing(inp)
        out = self._predict(norm_img)
        out_data = self._postprocessing(inp, out)
        return out_data


    