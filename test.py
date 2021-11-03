import argparse
import os
import time

import cv2
import numpy as np
import tensorrt as trt

from detector import trt_utils
from detector import preprocessing, postprocessing

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

WINDOW_NAME = "TensorRT detection example."

def get_engine(engine_file_path):
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def read_label_file(file_path):
    with open(file_path, "r") as file:
        readline = file.read().splitlines()
        return readline

def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(
        image, caption, (b[0], b[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
    )
    cv2.putText(
        image, caption, (b[0], b[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="File path of TensorRT model.", required=True)
    parser.add_argument(
        "--label", help="File path of label file.", required=True, type=str
    )
    parser.add_argument(
        "--videopath", help="File path of input video file.", default='/dev/video0', type=str
    )
    parser.add_argument(
        "--output", help="File path of output vide file.", default=None, type=str
    )
    parser.add_argument(
        "--scoreThreshold", help="Score threshold.", default=0.5, type=float
    )
    args = parser.parse_args()

    # Initialize window.
    cv2.namedWindow(
        WINDOW_NAME, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO
    )
    cv2.moveWindow(WINDOW_NAME, 100, 50)

    # Read label
    labels_list = read_label_file(args.label) if args.label else None
    
    # Video capture.
    if args.videopath == '/dev/video0':
        print("open camera.")
        cap = cv2.VideoCapture(args.videopath)
    else:
        print("open video file", args.videopath)
        cap = cv2.VideoCapture(args.videopath)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Input Video (height, width, fps): ", h, w, fps)

    model_name = os.path.splitext(os.path.basename(args.model))[0]

    # Load model.
    engine = get_engine(args.model)
    context = engine.create_execution_context()

    video_writer = None
    if args.output is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        video_writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    elapsed_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("VideoCapture read return false.")
            break

        normalized_img = preprocessing.preprocess_img(frame)

        # inference.
        start = time.perf_counter()
        inputs, outputs, bindings, stream = trt_utils.allocate_buffers(engine)
        inputs[0].host = normalized_img.ravel()
        outputs = trt_utils.do_inference_v2(
            context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        )

        inference_time = (time.perf_counter() - start) * 1000
        frame = postprocessing.postprocess_img(frame, outputs[1], outputs[0], labels_list)
        frame = frame['inference_data']['frame']

        # Calc fps.
        elapsed_list.append(inference_time)
        avg_text = ""
        if len(elapsed_list) > 100:
            elapsed_list.pop(0)
            avg_elapsed_ms = np.mean(elapsed_list)
            avg_text = " AGV: {0:.2f}ms".format(avg_elapsed_ms)

        # Display fps
        fps_text = "Inference: {0:.2f}ms".format(inference_time)
        display_text = model_name + " " + fps_text + avg_text
        draw_caption(frame, (10, 30), display_text)

        # Output video file
        if video_writer is not None:
            video_writer.write(frame)

        # Display
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
