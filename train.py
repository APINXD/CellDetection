from yolov5 import train, detect

if __name__ == '__main__':
    # train.run(imgsz=640, data='data.yaml', weights='yolov5s.pt', epochs=300, batch=-1, workers=2)
    detect.run(imgsz=640, weights='runs/train/exp2/weights/best.pt',
               source='test1.jpg',
               conf_thres=0.55,
               hide_labels=True,
               line_thickness=1,
               view_img=True)
