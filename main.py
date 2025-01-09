import argparse
import csv
import os
import platform
import sys
from pathlib import Path

###### ADD python packaeg bymyself ####

from utilsbymyself import auto_sending_message, llm_generate_file, inspectionlog
from PySide6 import QtWidgets, QtCore, QtGui
import cv2
from datetime import datetime
import numpy as np
from pathlib import Path
import time
import gc
###### END ####

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode

os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms'


##### Graphical interface development #####
class MWindow(QtWidgets.QMainWindow):

    def __init__(self, opt):

        super().__init__()

        # Define the number of signficant threat
        self.significant_threat_count = 0
        # Define the number of found threat in all image
        self.det_class_count = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0
        }
        # parameter needed for predict part
        self.img_size = (640, 640)  # default image size
        self.stride = 32  # default stride
        self.auto = True  # default auto parameter for letterbox
        # activate yolov5 model
        self.model, self.dt, self.names = self.YOLO(weights=opt.weights, data=opt.data, device=opt.device)

        # set up UI
        self.setupUI()

        self.imageBtn.clicked.connect(self.selectimage)
        self.camBtn.clicked.connect(self.startCamera)
        self.videoBtn.clicked.connect(self.startVideoFile)
        self.stopBtn.clicked.connect(self.stop)

        # set up timer to call show_camera
        self.timer_camera = QtCore.QTimer()
        self.timer_camera.timeout.connect(self.show_camera)

        # set up to update count
        self.fps = 15
        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self.periodic_update)

        # define imagenumber
        self.imagenumber = 0
        # list to store image to be predicted
        self.frameToAnalyze = []
        # inspection log number
        self.lognumber = 0
        # define temp image save path for video
        self.image2video_path = "./results/save"
        # define whether it is image
        self.isimage = False
        # define a signal to judege whether it's the same object.
        self.max_objects = 10
        self.judgethesameobject = {
            0: torch.zeros((self.max_objects, 4), device='cuda:0'),
            1: torch.zeros((self.max_objects, 4), device='cuda:0'),
            2: torch.zeros((self.max_objects, 4), device='cuda:0'),
            3: torch.zeros((self.max_objects, 4), device='cuda:0'),
            4: torch.zeros((self.max_objects, 4), device='cuda:0'),
            5: torch.zeros((self.max_objects, 4), device='cuda:0')
        }

        self.tempjudgethesameobject = {
            0: torch.zeros((self.max_objects, 4), device='cuda:0'),
            1: torch.zeros((self.max_objects, 4), device='cuda:0'),
            2: torch.zeros((self.max_objects, 4), device='cuda:0'),
            3: torch.zeros((self.max_objects, 4), device='cuda:0'),
            4: torch.zeros((self.max_objects, 4), device='cuda:0'),
            5: torch.zeros((self.max_objects, 4), device='cuda:0')
        }
        self.ifwrite2log = 1

        self.clear_folder("./results/log")
        self.clear_folder("./results")

        # # use another thread to run model
        # Thread(target=self.predict, daemon=True).start()

    def setupUI(self):

        self.resize(1200, 800)

        self.setWindowTitle('Construction Inspection Model')

        # central Widget
        centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(centralWidget)

        # central Widget  layout
        mainLayout = QtWidgets.QVBoxLayout(centralWidget)

        # define the top window
        topLayout = QtWidgets.QHBoxLayout()
        self.label_ori_video = QtWidgets.QLabel(self)
        self.label_treated = QtWidgets.QLabel(self)
        self.label_ori_video.setMinimumSize(520, 400)
        self.label_treated.setMinimumSize(520, 400)
        self.label_ori_video.setStyleSheet('border:1px solid #D7E2F9;')
        self.label_treated.setStyleSheet('border:1px solid #D7E2F9;')

        topLayout.addWidget(self.label_ori_video)
        topLayout.addWidget(self.label_treated)

        mainLayout.addLayout(topLayout)

        # define the bottom window
        groupBox = QtWidgets.QGroupBox(self)
        bottomLayout = QtWidgets.QVBoxLayout(groupBox)  # Change to vertical layout

        # Create a horizontal layout for class detection counts
        self.classCountLayout = QtWidgets.QHBoxLayout()
        self.classCountLabels = []

        # Create labels for class counts
        class_titles = [self.names[i] for i in range(6)]

        for i, title in enumerate(class_titles):
            label = QtWidgets.QLabel(f"{title}: {self.det_class_count[i]}")
            label.setStyleSheet("""
                background-color: rgba(255, 255, 255, 0.8);
                border: 1px solid #D7E2F9;
                border-radius: 5px;
                padding: 5px;
                text-align: center;
            """)
            self.classCountLayout.addWidget(label)
            self.classCountLabels.append(label)

        # Add the class count layout to the bottom layout
        bottomLayout.addLayout(self.classCountLayout)

        # Rest of the bottom layout remains the same
        self.textLog = QtWidgets.QTextBrowser()
        bottomLayout.addWidget(self.textLog)

        # Button layout
        btnLayout = QtWidgets.QVBoxLayout()
        self.imageBtn = QtWidgets.QPushButton('Image')
        self.videoBtn = QtWidgets.QPushButton('Video')
        self.camBtn = QtWidgets.QPushButton('Camera')
        self.stopBtn = QtWidgets.QPushButton('Stop and Generate')
        btnLayout.addWidget(self.imageBtn)
        btnLayout.addWidget(self.videoBtn)
        btnLayout.addWidget(self.camBtn)
        btnLayout.addWidget(self.stopBtn)
        bottomLayout.addLayout(btnLayout)

        # Add groupBox to main layout
        mainLayout.addWidget(groupBox)

    # update det_class_number in main window
    def periodic_update(self):
        for i, count in self.det_class_count.items():
            # print(self.det_class_count)
            self.classCountLabels[i].setText(f"{self.names[i]}: {count}")
            self.classCountLabels[i].repaint()

    def clear_folder(self, folder_path):

        folder = Path(folder_path)
        if not folder.exists():
            return

        for file in folder.iterdir():
            if file.is_file() and file.suffix.lower() in ['.jpg', '.png', '.bmp']:
                try:
                    file.unlink()
                    print(f"Deleted file: {file}")
                except Exception as e:
                    print(f"Failed to delete file {file}: {e}")

        for file in folder.iterdir():
            if file.is_file() and file.suffix.lower() in ['.md']:
                try:
                    file.unlink()
                    print(f"Deleted file: {file}")
                except Exception as e:
                    print(f"Failed to delete file {file}: {e}")

    def selectimage(self):
        self.isimage = True

        imagePath, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Image File", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if not imagePath:
            return

        self.textLog.append(f"Selected image: {imagePath}")

        image = cv2.imread(imagePath)
        if image is None:
            self.textLog.append("Failed to load the selected image.")
            return

        self.textLog.append("Image loaded successfully, processing...")

        self.frameToAnalyze.append(image)

        image_resized = cv2.resize(image, (520, 400))
        image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        qImage = QtGui.QImage(
            image_resized.data, image_resized.shape[1], image_resized.shape[0],
            QtGui.QImage.Format_RGB888
        )
        self.label_ori_video.setPixmap(QtGui.QPixmap.fromImage(qImage))
        self.label_ori_video.setAlignment(QtCore.Qt.AlignCenter)

        self.predict()

    def startVideoFile(self):
        self.isimage = False
        # choose video file
        videoPath, _ = QtWidgets.QFileDialog.getOpenFileName(self, "choose video file", "",
                                                             "videofile (*.mp4 *.avi *.mkv)")
        if not videoPath:
            return  # user cancel

        self.cap = cv2.VideoCapture(videoPath)  # open video file
        if not self.cap.isOpened():
            self.textLog.append(f"can not open video file: {videoPath}")
            return

        self.textLog.append(f"Video file is processing: {videoPath}")
        # delete residual files
        self.clear_folder(self.image2video_path)
        if not self.timer_camera.isActive():
            self.timer_camera.start((1 / self.fps) * 1000)
        if not self.update_timer.isActive():
            self.update_timer.start(1000)

    def startCamera(self):

        self.isimage = False
        # windows cv2.CAP_DSHOW
        # Linux/Mac V4L, FFMPEG or GSTREAMER
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L)
        # self.cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("The firsr camera can't be opened!")
            return ()

        # delete residual files
        self.clear_folder(self.image2video_path)
        if self.timer_camera.isActive() == False:
            self.timer_camera.start((1 / self.fps) * 1000)
        # update count
        if not self.update_timer.isActive():
            self.update_timer.start(1000)

    def show_camera(self):

        ret, frame = self.cap.read()  # read image from video
        if not ret:
            return

        im0 = frame.copy()
        frame = cv2.resize(frame, (520, 400))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qImage = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0],
                              QtGui.QImage.Format_RGB888)
        # show original image in window
        self.label_ori_video.setPixmap(QtGui.QPixmap.fromImage(qImage))
        self.label_ori_video.setAlignment(QtCore.Qt.AlignCenter)

        # if no task
        if not self.frameToAnalyze:
            self.frameToAnalyze.append(im0)

        self.predict()

    @smart_inference_mode()
    def YOLO(self,
             weights=ROOT / "yolov5s.pt",  # model path or triton URL
             data=ROOT / "data/custom_data.yaml",  # dataset.yaml path
             imgsz=(640, 640),  # inference size (height, width)
             device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
             half=False,  # use FP16 half-precision inference
             dnn=False,  # use OpenCV DNN for ONNX inference
             ):

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloader
        bs = 1  # batch_size

        # Run inference
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))

        return model, dt, names

    def predict(self,
                conf_thres=0.25,  # confidence threshold
                iou_thres=0.45,  # NMS IOU threshold
                max_det=1000,  # maximum detections per image
                classes=None,  # filter by class: --class 0, or --class 0 2 3
                agnostic_nms=False,  # class-agnostic NMS
                augment=False,  # augmented inference
                visualize=False,  # visualize features
                line_thickness=3,  # bounding box thickness (pixels)
                hide_labels=False,  # hide labels
                hide_conf=False,  # hide confidences
                ):
        with torch.no_grad():
            if not self.frameToAnalyze:
                time.sleep(0.5)
            else:
                # im: image to be processed; im0: original image format
                im0 = self.frameToAnalyze.pop(0)
                # save original image for log
                log_save_path = "./results/log/" + str(self.lognumber) + str(self.imagenumber) + ".jpg"
                # image path needed to write to mdfile, need use relative path
                image_in_log_path = log_save_path.replace("results/", "")
                if not self.isimage:
                    cv2.imwrite(log_save_path, im0)
                # define time to detect this image, used to send message timely
                self.log_time = datetime.now().strftime("%Y %m %d ,%H:%M:%S")
                # inspection log path
                self.logfile_path = "./results/construction_inspection_log" + f"{self.lognumber}" + ".md"

                im = self.letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
                im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                im = np.ascontiguousarray(im)  # contiguous

                

                with self.dt[0]:
                    im = torch.from_numpy(im).to(self.model.device)
                    im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim
                    if self.model.xml and im.shape[0] > 1:
                        ims = torch.chunk(im, im.shape[0], 0)

                # Inference
                with self.dt[1]:

                    if self.model.xml and im.shape[0] > 1:
                        pred = None
                        for image in ims:
                            if pred is None:
                                pred = self.model(image, augment=augment, visualize=visualize).unsqueeze(0)
                            else:
                                pred = torch.cat(
                                    (pred, self.model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                        pred = [pred, None]
                    else:
                        pred = self.model(im, augment=augment, visualize=visualize)
                # NMS
                with self.dt[2]:
                    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

                # judge part
                judgefirstwrite = 0
                if not self.judgethesameobject:
                    judgefirstwrite = 1

                # Process predictions
                det = pred[0]
                image_save_path_for_video = "./results/save/" + str(self.imagenumber) + ".jpg"
                annotator = Annotator(im0, line_width=line_thickness, example=str(self.names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    det_count = 0
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label_log = self.names[c] if hide_conf else f"{self.names[c]}"

                        confidence = float(conf)
                        confidence_str = f"{confidence:.2f}"

                        label = None if hide_labels else (self.names[c] if hide_conf else f"{self.names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))

                        if not self.isimage:

                            # judge whether it is the same thing.
                            xyxy_tensor = torch.tensor(xyxy, device='cuda:0')

                            # judge whether it is the same thing.
                            self.tempjudgethesameobject[c][det_count] = xyxy_tensor

                            if judgefirstwrite:
                                self.judgethesameobject[c][det_count] = xyxy_tensor
                            else:
                                for i in self.judgethesameobject[c]:
                                    if torch.any(i != torch.zeros(4, device='cuda:0')):
                                        # Now both i and xyxy_tensor are on the same device (GPU)
                                        iou = self.calculate_iou(i, xyxy_tensor)  # No need to move tensors to CPU
                                        if iou > 0.4:
                                            self.ifwrite2log = 0
                                            break

                            #### FIND SERIOUS THREAT AND SENDING MESSGAE ####
                            if self.significant_threat_count == 0 and c in [0, 1]:
                                message_content = "Construction Inspection Model notes:" + f"{self.log_time},Found{label_log}, Please login in platform to deal!"
                                # auto_sending_message.main_auto_sending(message_content)

                            if self.ifwrite2log:
                                inspectionlog.append_to_markdown_table(self.logfile_path, label_log, image_in_log_path)
                                self.textLog.append(f"{label_log} recorded!")
                                self.det_class_count[c] += 1

                            self.ifwrite2log = 1
                            det_count += 1

                    self.judgethesameobject = self.tempjudgethesameobject
                    self.tempjudgethesameobject = {
                        0: torch.zeros((self.max_objects, 4), device='cuda:0'),
                        1: torch.zeros((self.max_objects, 4), device='cuda:0'),
                        2: torch.zeros((self.max_objects, 4), device='cuda:0'),
                        3: torch.zeros((self.max_objects, 4), device='cuda:0'),
                        4: torch.zeros((self.max_objects, 4), device='cuda:0'),
                        5: torch.zeros((self.max_objects, 4), device='cuda:0')
                    }

                im0 = annotator.result()
                frame_processed = im0.copy()
                frame_processed = cv2.resize(frame_processed, (520, 400))
                frame_processed = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB)
                qImage_processed = QtGui.QImage(frame_processed.data, frame_processed.shape[1],
                                                frame_processed.shape[0],
                                                QtGui.QImage.Format_RGB888)
                self.label_treated.setPixmap(QtGui.QPixmap.fromImage(qImage_processed))
                self.label_treated.setAlignment(QtCore.Qt.AlignCenter)
                # save image with labels
                # cv2.imwrite(image_save_path_for_video, im0)

                self.imagenumber += 1

                torch.cuda.empty_cache()
                gc.collect()

    def calculate_iou(self, box1, box2):
        # box1 and box2 are assumed to be on the same device 
        device = 'cuda:0'
        
        x1 = torch.max(box1[0], box2[0])
        y1 = torch.max(box1[1], box2[1])
        x2 = torch.min(box1[2], box2[2])
        y2 = torch.min(box1[3], box2[3])

        intersection = torch.max(torch.tensor(0.0, device=device), x2 - x1) * torch.max(
            torch.tensor(0.0, device=device), y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        iou = intersection / union if union != 0 else 0
        return iou

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True,
                  stride=32):
        """Resizes and pads image to new_shape with stride-multiple constraints, returns resized image, ratio, padding."""
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)

    def generate_video(self, folder_path, output_video_path, fps=30, delete_images_after=True):
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            self.textLog.append(f"Folder {folder_path} does not exist or is not a directory.")
            return

        image_files = sorted([f for f in folder.iterdir() if f.suffix in ['.jpg', '.png', '.bmp']])
        if not image_files:
            self.textLog.append(f"No image files found in {folder_path}.")
            return

        first_image = cv2.imread(str(image_files[0]))
        if first_image is None:
            self.textLog.append(f"Failed to load the first image {image_files[0]}.")
            return

        height, width, _ = first_image.shape
        self.textLog.append(f"Generating video from {len(image_files)} images...")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        for image_file in image_files:
            image = cv2.imread(str(image_file))
            if image is None:
                self.textLog.append(f"Failed to load image {image_file}, skipping...")
                continue
            video_writer.write(image)

        video_writer.release()
        self.textLog.append(f"Video saved to {output_video_path}.")

        if delete_images_after:
            for image_file in image_files:
                try:
                    image_file.unlink()
                except Exception as e:
                    self.textLog.append(f"Failed to delete image {image_file}: {e}")

    def llm_generate(self):
        llm_content = {}
        for i, count in self.det_class_count.items():
            llm_content[self.names[i]] = count
        llm_generate_file.generate_mdfile_byLLM(llm_content, self.logfile_path)
        self.textLog.append(f"Construction inspection log {self.logfile_path} genereated!")

    def stop(self):
        # close UI
        self.timer_camera.stop()

        self.label_ori_video.clear()
        self.label_treated.clear()

        if not self.isimage:
            self.cap.release()

            # call llm to generate logfile
            self.textLog.append(f"Construction inspection log is genreating!")
            self.llm_generate()

            # generate recongized video
            # video_output_path = "./results/construction_inspection_video" + str(self.lognumber) + ".mp4"
            # self.generate_video(self.image2video_path, video_output_path, self.fps)
        # Define the number of signficant threat
        self.significant_threat_count = 0
        # Define the number of found threat in all image
        self.det_class_count = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0
        }
        self.imagenumber = 0
        self.isimage = False
        # the number of log
        self.lognumber += 1

        self.judgethesameobject = {
            0: torch.zeros((self.max_objects, 4), device='cuda:0'),
            1: torch.zeros((self.max_objects, 4), device='cuda:0'),
            2: torch.zeros((self.max_objects, 4), device='cuda:0'),
            3: torch.zeros((self.max_objects, 4), device='cuda:0'),
            4: torch.zeros((self.max_objects, 4), device='cuda:0'),
            5: torch.zeros((self.max_objects, 4), device='cuda:0')
        }

        self.tempjudgethesameobject = {
            0: torch.zeros((self.max_objects, 4), device='cuda:0'),
            1: torch.zeros((self.max_objects, 4), device='cuda:0'),
            2: torch.zeros((self.max_objects, 4), device='cuda:0'),
            3: torch.zeros((self.max_objects, 4), device='cuda:0'),
            4: torch.zeros((self.max_objects, 4), device='cuda:0'),
            5: torch.zeros((self.max_objects, 4), device='cuda:0')
        }

        gc.collect()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "best.pt", help="model path or triton URL")
    parser.add_argument("--data", type=str, default=ROOT / "data/custom_data.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--device", default="0", help="0 device, i.e. 0 or 0,1,2,3 or cpu")
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main():
    opt = parse_opt()
    app = QtWidgets.QApplication()
    window = MWindow(opt)
    window.show()
    app.exec()


if __name__ == '__main__':
    main()














