#cameraIp:172.15.88.143
#数采服务器：172.15.88.117：8080
import os.path
import sys
import time
import datetime
import cv2
import re
import logging,logging.config
from cloghandler import ConcurrentRotatingFileHandler
import socket
import requests
import multiprocessing as mp
import ctypes
from pathlib import Path


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from utils import box_in_mask_predict, frame_to_img, in_car_judgement
from app.utils import getPredictor
from yolov5.utils.general import non_max_suppression, scale_coords
# from utils import MultiProcessSafeTimedRotatingFileHandler

maskThreshould = 0.7
boxThreshould = 0.3
IOUinCar = 0.7
weight_mask = '../maskrcnn/model_final.pth'
# weight_box = './fasterrcnn/model_final.pth'
weight_box = '../yolov5/best.pt'


def sendResult(result1, result2):
    #socket
    s = socket.socket()
    host = "172.15.3.248"
    port = 12345  # 设置端口
    s.bind((host, port))
    s.listen(10)

    #日志
    logging.config.fileConfig("../log/logging.ini")
    log = logging.getLogger()

    while True:
        try:
            c, addr = s.accept()
            print("连接地址:", addr)
            while True:
                message = "1" if True in [result1.value, result2.value] else "0"
                c.send(message.encode())
                if message == "1":
                    log.info("服务器消息：{} 时间戳：{}".format(message, time.time()))
                    now = datetime.datetime.now().strftime('%m_%d_%H_%M_%S')
                    print("{}:识别结果:{}".format(now, message))
                time.sleep(0.2)
        except Exception as e:
            print(e)
            c.close()


def weight_get(weight, url="http://172.15.88.145:8080/"):
    while True:
        try:
            # 设置重连次数
            requests.adapters.DEFAULT_RETRIES = 15
            # 设置连接活跃状态为False
            s = requests.session()
            s.keep_alive = False
            response = requests.get(url).text
            weight.value = int("".join(re.findall(r";(.+?);", response)))
            time.sleep(1)
        except Exception as e:
            print(e)
            time.sleep(10)
        # stat = int("".join(re.findall(r";(.+?);",response))) > 500
        # print(stat)
        # requestQ.put(stat)
        # requestQ.get() if requestQ.qsize() > 6 else time.sleep(0.001)
        # print(weight.value)


def image_put(weight, imageQ, user, pwd, ip, processid, channel=3):
    cap = cv2.VideoCapture("rtsp://%s:%s@%s/cam/realmonitor?channel=%d&subtype=0" % (user, pwd, ip, channel))
    print("进程{}启动推流".format(processid))
    while True:
        # print("进程{}开始推流".format(processid))
        # print(cap.read()[0])
        _, frame = cap.read()
        if _:
            imageQ.put(cap.read()[1])
            imageQ.get() if imageQ.qsize() > 1 else time.sleep(0.001)
        else:
            cap = cv2.VideoCapture("rtsp://%s:%s@%s/cam/realmonitor?channel=%d&subtype=0" % (user, pwd, ip, channel))


def predict(weight, q, result, processid):#调用模型
    predictor_mask = getPredictor('mask', weight_mask, maskThreshould)
    predictor_box = getPredictor('box', weight_box)
    print("进程{}启动检测".format(processid))
    while True:
        if weight.value > 500:
            # print("进程{}开始检测".format(processid))
            frame = q.get()
            # frame = imgAug(frame)
            if frame is None:
                print("未获取到视频")
                continue
            # # seg
            # print("开始分割")
            outputs_mask = predictor_mask(frame)
            # print("分割完成")

            # TODO convert framge to img(yolo format)
            img = frame_to_img(frame)
            # # detect
            # print("开始检测")
            outputs_box = predictor_box(img)
            # print("检测完成")
            # NMS
            outputs_box = non_max_suppression(outputs_box, conf_thres=boxThreshould)
            outputs_box = outputs_box[0][:, :4]
            # resize to orignal size
            outputs_box = scale_coords(img.shape[2:], outputs_box, frame.shape).round()

            result.value = in_car_judgement(outputs_mask, outputs_box, IOUinCar)
            # print("进程{}识别结果{}".format(processid, result.value))

            # resultQ.put(result)
            # resultQ.get() if resultQ.qsize() > 2 else time.sleep(0.001)
        else:
            result.value = False


def run_multi_camera():
    user_name = "admin"
    user_pwd_list = [
        "admin",
        "admin"
    ]
    camera_ip_list = [
        "172.15.88.117:80",
        "172.15.88.147:80"
    ]
    channel_list = [
        1,
        1
                    ]

    processes = []
    mp.set_start_method(method='spawn')
    imageQs = [mp.Queue(maxsize=4) for _ in camera_ip_list]

    # 单独开进程获取车辆稳定信息
    # requestsQ = mp.Queue(maxsize=8)
    weight = mp.Manager().Value(ctypes.c_int, 0)
    processes.append(mp.Process(target=weight_get, args=(weight,)))

    #单独开进程执行信息传送，记录log
    result1 = mp.Manager().Value(ctypes.c_bool, False)
    result2 = mp.Manager().Value(ctypes.c_bool, False)
    results = [result1, result2]
    processes.append(mp.Process(target=sendResult, args=(result1, result2)))

    #单独开进程记录log
    # processes.append(mp.Process(target=log, args=(result1, result2)))

    for i in range(len(imageQs)):
        imageQ, user_pwd, camera_ip, channel, result = imageQs[i], user_pwd_list[i], camera_ip_list[i], channel_list[i], results[i]
        processes.append(mp.Process(target=image_put, args=(weight, imageQ, user_name, user_pwd, camera_ip, i, channel)))
        processes.append(mp.Process(target=predict, args=(weight, imageQ, result, i)))
    for process in processes:
        process.daemon = True
        process.start()
    for process in processes:
        process.join()


if __name__ == "__main__":
    run_multi_camera()