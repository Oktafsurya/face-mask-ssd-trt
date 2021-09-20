python3 face_mask_inference.py --model=ssd-mobilenet-face-mask.onnx --labels=labels-face-mask.txt --input-blob=input_0 --output-cvg=scores --output-bbox=boxes /dev/video0 &
python3 face_mask_dashboard.py &
