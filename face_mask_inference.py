import jetson.inference
import jetson.utils

import argparse
import sys
import cv2
import zmq
import base64
import json

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

#ZMQ ==========================================================================================================
context_ssd_output = zmq.Context()
context_count = zmq.Context()
context_json = zmq.Context()

#SSD Scene Publisher
footage_socket_ssd_output = context_ssd_output.socket(zmq.PUB)
footage_socket_ssd_output.connect('tcp://localhost:7000')

# Counter socket
count_socket = context_count.socket(zmq.PUB)
count_socket.connect('tcp://localhost:6100')

# JSON socket
json_socket = context_json.socket(zmq.PUB)
json_socket.connect('tcp://localhost:6500')
#ZMQ End =====================================================================================================

output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv+is_headless)
net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
class_list = ['BACKGROUND', 'with_mask', 'without_mask', 'mask_weared_incorrect']

while True:

	try:

		total_person_count = 0
		label_each_frame = []
		score_each_frame = []

		img = input.Capture()
		detections = net.Detect(img, overlay=opt.overlay)
		
		for detection in detections:
			cls_label = class_list[int(detection.ClassID)]
			score =  detection.Confidence
			total_person_count = len(detections)
			label_each_frame.append(cls_label)
			score_each_frame.append(score)

		counter_send = str(total_person_count)
		count_socket.send_string(counter_send) 

		json_array = {"class_str": label_each_frame,
                	"score": score_each_frame}
		data_json = json.dumps(json_array)
		json_socket.send_json(data_json)

		img = jetson.utils.cudaToNumpy(img)
		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		encoded, buffer = cv2.imencode('.jpg', img)
		footage_socket_ssd_output.send(base64.b64encode(buffer))

		# print out performance info
		# net.PrintProfilerTimes()

		cv2.waitKey(1)

		if not input.IsStreaming() or not output.IsStreaming():
			break
		
	except KeyboardInterrupt:
		print("\n\nBye bye\n")
		break
