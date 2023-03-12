import argparse
import io
import os

import cv2
from PIL import Image
import datetime
from flask import Flask, render_template, request, redirect
from yolov5 import detect
import torch

app = Flask(__name__)

DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"


@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        # img = cv2.imread(file)
        model.conf = 0.55
        # model.labels = False
        model.hide_labels = True
        model.line_thickness = 1
        results = model(img)

        # results = detect.run(imgsz=640, weights='runs/train/exp2/weights/best.pt',
        #                      source=img,
        #                      conf_thres=0.55,
        #                      hide_labels=True,
        #                      line_thickness=1,
        #                      view_img=True)

        results.render(labels=False)
        # results.ims()# updates results.imgs with boxes and labels
        now_time = datetime.datetime.now().strftime(DATETIME_FORMAT)
        img_savename = f"static/{now_time}.png"
        Image.fromarray(results.ims[0]).save(img_savename)
        return redirect(img_savename)

    return render_template("index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    # parser.add_argument('--hide-labels', default=True, action='store_true', help='hide labels')
    # parser.add_argument('--hide-conf', default=True, action='store_true', help='hide confidences')
    args = parser.parse_args()

    model = torch.hub.load('ultralytics/yolov5', 'custom', 'runs/train/exp2/weights/best.pt', device="0")
    # model.conf = 0.55
    # model.hide_labels = True
    # model.line_thickness = 1
    model.eval()
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
