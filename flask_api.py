"""
Run a rest API exposing the yolov5s object detection model
"""
import argparse
import io
from PIL import Image

import torch
from flask import Flask, request
import numpy as np
import cv2

from predict_api import detect

app = Flask(__name__)

DETECTION_URL = "/v1/object-detection/yolov5s"


@app.route(DETECTION_URL, methods=["POST"])
def predict():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    im = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    #saving image for debugging. disable it before deployment
    cv2.imwrite("client_image.jpg",im)

    #passing image to predictor

    outputs = detect("client_image.jpg")
    print(outputs)

    return outputs


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Flask api exposing yolov5 model")
    # parser.add_argument("--port", default=5000, type=int, help="port number")
    # args = parser.parse_args()

    # model = torch.hub.load(
    #     "ultralytics/yolov5", "yolov5s", pretrained=True, force_reload=True
    # ).autoshape()  # force_reload = recache latest code
    # model.eval()
    app.run(host="0.0.0.0", port=5000)  # debug=True causes Restarting with stat