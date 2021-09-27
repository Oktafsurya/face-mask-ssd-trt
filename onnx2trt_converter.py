import argparse
import os
import tensorrt as trt

from onnx import ModelProto
from typing import Optional


class ONNX2TRT:
    def __init__(self, onnx_path: str, trt_path: str, precision: str, shape: Optional[str] = None):
        self.shape = shape
        self.input_path = onnx_path
        self.output_path = trt_path
        self.precision = precision
        self.trt_logger = trt.Logger()
        self.explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    def convert(self):
        trt.init_libnvinfer_plugins(self.trt_logger, "")
        builder = trt.Builder(self.trt_logger)
        network = builder.create_network(self.explicit_batch)
        config = builder.create_builder_config()
        parser = trt.OnnxParser(network, self.trt_logger)

        config.max_workspace_size = (256 << 20)
        builder.max_batch_size = 1
        if self.precision == 'fp32':
            print("[INFO] Using default precision: fp32")
        elif self.precision == 'fp16':
            print("[INFO] Using precision: fp16")
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
            config.set_flag(trt.BuilderFlag.FP16)
        elif self.precision == 'int8':
            print("[INFO] Using precision: int8")
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
            config.set_flag(trt.BuilderFlag.INT8)
        else:
            print(TypeError, "[INFO] Expected precision: fp16, fp32 or int8")
            raise Exception

        print("Loading ONNX file from path {}...".format(self.input_path))
        if not os.path.exists(self.input_path):
            print(Exception, "[ERROR]: ONNX file {} not found.".format(self.input_path))

        with open(self.input_path, "rb") as model:
            print("[INFO] Beginning ONNX file parsing")
            if not parser.parse(model.read()):
                print(Exception, "[ERROR]: Failed to parse the ONNX file.")

        model = ModelProto()
        with open(self.input_path, "rb") as f:
            model.ParseFromString(f.read())
        if self.shape is None or self.shape == "":
            for index, inp in enumerate(model.graph.input):
                shape = list()
                for dim in inp.type.tensor_type.shape.dim:
                    shape.append(dim.dim_value)
                try:
                    network.get_input(index).shape = shape
                except:
                    print("[ERROR]: Can not assign shape, use default instead")
        else:
            index = 0
            while index < len(model.graph.input):
                start = self.shape.find("(") + len("(")
                end = self.shape.find(")")
                if self.shape == self.shape[start:end] or self.shape[start:end] == '': break
                shape = self.shape[start:end]
                shape = [int(dim) for dim in shape.split(',')]
                try:
                    network.get_input(index).shape = shape
                except:
                    print("[ERROR]: Can not assign shape, use default instead")
                self.shape = self.shape[end+1:]
                index += 1

        print("[INFO] Completed parsing of ONNX file")
        print("[INFO] Building an engine from file {}; this may take a while...".format(self.input_path))
        engine = builder.build_engine(network, config)
        print("[INFO] Completed creating Engine")
        with open(self.output_path, "wb") as f:
            f.write(engine.serialize())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="File path of onnx model.", required=True)
    parser.add_argument(
        "--output", help="File path to save trt file output", required=True, type=str
    )
    parser.add_argument(
        "--precision", help="precision to be used", required=True, type=str
    )

    args = parser.parse_args()
    converter = ONNX2TRT(args.model,args.output, args.precision)
    converter.convert()