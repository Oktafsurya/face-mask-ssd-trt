import argparse
from detector import (
    FaceMaskServer,
    FrameHandler,
    InferenceHandler,
    InferenceInstance,
    IOHandler
)

class FaceMaskApp:
    def __init__(self, args):
        self.args = args

    def run(self):
        inference_handler = InferenceHandler(self.args.model, self.args.label)
        inference_instance = InferenceInstance(inference_handler)
        frame_handler = FrameHandler(self.args.source)
        _io_handler = IOHandler(inference_instance, frame_handler)
        server = FaceMaskServer(_io_handler)
        server.run()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", help="File path of TensorRT model.", required=True, type=str)
    parser.add_argument(
        "--label", help="File path of label file.", required=True, type=str)
    parser.add_argument(
        "--source", help="File path of input video file.", default='/dev/video0', type=str)
    args = parser.parse_args()

    app = FaceMaskApp(args)
    app.run()

if __name__ == '__main__':
    main()

