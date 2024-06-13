import torch
import numpy as np
import os, cv2, time, torch, random, pytorchvideo, warnings, argparse, math, json

warnings.filterwarnings("ignore", category=UserWarning)

from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image, )
from torchvision.transforms._functional_video import normalize
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slowfast_r50_detection
from deep_sort.deep_sort import DeepSort
from Read_json import *


deepsort_results = []
yolo_results = []
deepsort_results_json = []




class MyVideoCapture:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.idx = -1
        self.end = False
        self.stack = []


    def read(self):
        self.idx += 1
        ret, img = self.cap.read()
        if ret:
            self.stack.append(img)
        else:
            self.end = True
        return ret, img

    def to_tensor(self, img):
        img = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return img.unsqueeze(0)

    def get_video_clip(self):
        assert len(self.stack) > 0, "clip length must large than 0 !"
        self.stack = [self.to_tensor(img) for img in self.stack]
        clip = torch.cat(self.stack).permute(-1, 0, 1, 2)
        del self.stack
        self.stack = []
        return clip

    def release(self):
        self.cap.release()

class Yolo_Main:
    def __init__(self, count,name):
        self.count = count
        self.name = name
        self.no_intention_list = []
        self.intention_noCommuniction_list = []
        if(self.count != 0):
            with open("E:\\Computing_Power_Measurement\\Intention\\yolo_slowfast-master\\next_states_list.json", 'r') as f6:
                next_states = json.load(f6)
            self.next = next_states
    def tensor_to_numpy(self,tensor):
        img = tensor.cpu().numpy().transpose((1, 2, 0))
        return img

    def ava_inference_transform(self,
            clip,
            boxes,
            num_frames=32,
            crop_size=640,
            data_mean=[0.45, 0.45, 0.45],
            data_std=[0.225, 0.225, 0.225],
            slow_fast_alpha=4,
    ):
        boxes = np.array(boxes)
        roi_boxes = boxes.copy()
        clip = uniform_temporal_subsample(clip, num_frames)
        clip = clip.float()
        clip = clip / 255.0
        height, width = clip.shape[2], clip.shape[3]
        boxes = clip_boxes_to_image(boxes, height, width)
        clip, boxes = short_side_scale_with_boxes(clip, size=crop_size, boxes=boxes, )
        clip = normalize(clip,
                         np.array(data_mean, dtype=np.float32),
                         np.array(data_std, dtype=np.float32), )
        boxes = clip_boxes_to_image(boxes, clip.shape[2], clip.shape[3])
        if slow_fast_alpha is not None:
            fast_pathway = clip
            slow_pathway = torch.index_select(clip, 1,
                                              torch.linspace(0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha).long())
            clip = [slow_pathway, fast_pathway]

        return clip, torch.from_numpy(boxes), roi_boxes


    def plot_one_box(self,x, img, color=[100, 100, 100], text_info="None",
                     velocity=None, thickness=1, fontsize=0.5, fontthickness=1):
        #print("x",x)
        if (self.count == 0):
            c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
            cv2.rectangle(img, c1, c2, color, thickness, lineType=cv2.LINE_AA)

            t_size = cv2.getTextSize(text_info, cv2.FONT_HERSHEY_TRIPLEX, fontsize, fontthickness + 2)[0]
            cv2.rectangle(img, c1, (c1[0] + int(t_size[0]), c1[1] + int(t_size[1] * 1.45)), color, -1)
            cv2.putText(img, text_info, (c1[0], c1[1] + t_size[1] + 2),
                        cv2.FONT_HERSHEY_TRIPLEX, fontsize, [255, 255, 255], fontthickness)

            if velocity is not None:
                x_center = (c1[0] + c2[0]) // 2
                y_center = (c1[1] + c2[1]) // 2

                dx, dy = velocity
                scale = 10
                color1 = (0, 0, 255)
                color2 = (255, 0, 255)
                cv2.arrowedLine(img, (x_center, y_center), (int(x_center + dx * scale), int(y_center + dy * scale)),
                                color1, 2)

                direction = ""
                if c2[0] > c1[0] + 30:
                    direction = "Right"
                elif c2[0] < c1[0] - 30:
                    direction = "Left"
                elif c2[1] > c1[1] + 30:
                    direction += "Down"
                else:
                    direction += "Up"

                color1 = [255, 0, 0]
                cv2.putText(img, direction, (c1[0], c1[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, fontsize, color1, fontthickness)
        if (self.count == 1):
            c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
            cv2.rectangle(img, c1, c2, color, thickness, lineType=cv2.LINE_AA)

            t_size = cv2.getTextSize(text_info, cv2.FONT_HERSHEY_TRIPLEX, fontsize, fontthickness + 2)[0]
            cv2.rectangle(img, c1, (c1[0] + int(t_size[0]), c1[1] + int(t_size[1] * 1.45)), color, -1)
            cv2.putText(img, text_info, (c1[0], c1[1] + t_size[1] + 2),
                        cv2.FONT_HERSHEY_TRIPLEX, fontsize, [255, 255, 255], fontthickness)

            if velocity is not None:
                x_center = (c1[0] + c2[0]) // 2
                y_center = (c1[1] + c2[1]) // 2

                dx, dy = velocity
                scale = 10
                color1 = (0, 0, 255)
                color2 = (255, 0, 255)
                cv2.arrowedLine(img, (x_center, y_center), (int(x_center + dx * scale), int(y_center + dy * scale)),
                                color1, 2)

                direction = ""
                if c2[0] > c1[0] + 30:
                    direction = "Right"
                elif c2[0] < c1[0] - 30:
                    direction = "Left"
                elif c2[1] > c1[1] + 30:
                    direction += "Down"
                else:
                    direction += "Up"

                color1 = [255, 0, 0]
                cv2.putText(img, direction, (c1[0], c1[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, fontsize, color1, fontthickness)
        if (self.count == 2):
            only_yolo = yolo_results[read_num]
            a1, a2 = (int(only_yolo[0]), int(only_yolo[1])), (int(only_yolo[2]), int(only_yolo[3]))
            c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
            cv2.rectangle(img, c1, c2, color, thickness, lineType=cv2.LINE_AA)

            t_size = cv2.getTextSize(text_info, cv2.FONT_HERSHEY_TRIPLEX, fontsize, fontthickness + 2)[0]
            cv2.rectangle(img, c1, (c1[0] + int(t_size[0]), c1[1] + int(t_size[1] * 1.45)), color, -1)
            cv2.putText(img, text_info, (c1[0], c1[1] + t_size[1] + 2),
                        cv2.FONT_HERSHEY_TRIPLEX, fontsize, [255, 255, 255], fontthickness)
            no_intention = calculate_accuracy(x,only_yolo,0.5)
            if (no_intention != 0):
                self.no_intention_list.append(no_intention)
            #print("无意图",no_intention)
            if velocity is not None:
                x_center = (c1[0] + c2[0]) // 2
                y_center = (c1[1] + c2[1]) // 2
                # print("x_center,y_center",x_center,y_center)
                a_x_center = (a1[0] + a2[0]) // 2
                a_y_center = (a1[1] + a2[1]) // 2
                # print("a_x_center,a_y_center", a_x_center,a_y_center)
                dx, dy = velocity
                scale = 10
                color1 = (0, 0, 255)
                color2 = (255, 0, 255)
                cv2.arrowedLine(img, (x_center, y_center), (int(x_center + dx * scale), int(y_center + dy * scale)),
                                color1, 2)
                # TODO:用来展示是否添加了主被动意图
                cv2.arrowedLine(img, (x_center,y_center),
                                (int(a_x_center + dx * scale), int(a_y_center + dy * scale)), color2, 2)

                direction = ""
                if c2[0] > c1[0] + 30:
                    direction = "Right"
                elif c2[0] < c1[0] - 30:
                    direction = "Left"
                elif c2[1] > c1[1] + 30:
                    direction += "Down"
                else:
                    direction += "Up"

                color1 = [255, 0, 0]
                cv2.putText(img, direction, (c1[0], c1[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, fontsize, color1, fontthickness)
        if (self.count == 3):

            only_zhudong = self.next[read_num]
            only_yolo = yolo_results[read_num]
            a1, a2 = (int(only_yolo[0]), int(only_yolo[1])), (int(only_yolo[2]), int(only_yolo[3]))
            b1, b2 = (int(only_zhudong[0]), int(only_zhudong[1])), (int(only_zhudong[2]), int(only_zhudong[3]))
            c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
            cv2.rectangle(img, c1, c2, color, thickness, lineType=cv2.LINE_AA)

            t_size = cv2.getTextSize(text_info, cv2.FONT_HERSHEY_TRIPLEX, fontsize, fontthickness + 2)[0]
            cv2.rectangle(img, c1, (c1[0] + int(t_size[0]), c1[1] + int(t_size[1] * 1.45)), color, -1)
            cv2.putText(img, text_info, (c1[0], c1[1] + t_size[1] + 2),
                        cv2.FONT_HERSHEY_TRIPLEX, fontsize, [255, 255, 255], fontthickness)
            intention_noCommuniction = calculate_accuracy(x, only_zhudong, 0.5)
            if (intention_noCommuniction != 0):
                self.intention_noCommuniction_list.append(intention_noCommuniction)
            #print("有意图无通信", intention_noCommuniction)
            if velocity is not None:
                x_center = (c1[0] + c2[0]) // 2
                y_center = (c1[1] + c2[1]) // 2
                #print("x_center,y_center", x_center, y_center)
                a_x_center = (a1[0] + a2[0]) // 2
                a_y_center = (a1[1] + a2[1]) // 2
                #print("a_x_center,a_y_center", a_x_center, a_y_center)

                b_x_center = (b1[0] + b2[0]) // 2
                b_y_center = (b1[1] + b2[1]) // 2
                #print("b_x_center,b_y_center", b_x_center, b_y_center)
                dx, dy = velocity
                scale = 10
                color1 = (0, 0, 255)
                color2 = (255, 0, 255)
                color3 = (255, 20, 100)
                cv2.arrowedLine(img, (x_center, y_center), (int(x_center + dx * scale), int(y_center + dy * scale)),
                                color1, 2)
                # TODO:用来展示是否添加了主被动意图
                #cv2.arrowedLine(img, (a_x_center, a_y_center),(int(a_x_center + dx * scale), int(a_y_center + dy * scale)), color2, 2)
                cv2.arrowedLine(img, (x_center, y_center),
                                (int(b_x_center + dx * scale), int(b_y_center + dy * scale)), color3, 2)
                direction = ""
                if c2[0] > c1[0] + 30:
                    direction = "Right"
                elif c2[0] < c1[0] - 30:
                    direction = "Left"
                elif c2[1] > c1[1] + 30:
                    direction += "Down"
                else:
                    direction += "Up"

                color1 = [255, 0, 0]
                cv2.putText(img, direction, (c1[0], c1[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, fontsize, color1, fontthickness)


        return img


    def deepsort_update(self,Tracker, pred, xywh, np_img):
        outputs = Tracker.update(xywh, pred[:, 4:5], pred[:, 5].tolist(), cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
        return outputs

    def save_yolopreds_tovideo(self, yolo_preds, id_to_ava_labels, color_map, output_video, vis=False):
        for i, (im, pred) in enumerate(zip(yolo_preds.ims, yolo_preds.pred)):
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            if pred.shape[0]:
                for j, (*box, cls, trackid, vx, vy) in enumerate(pred):
                    if int(cls) != 0:
                        ava_label = ''
                    elif trackid in id_to_ava_labels.keys():
                        ava_label = id_to_ava_labels[trackid].split(' ')[0]
                    else:
                        ava_label = 'Unknow'
                    if (self.count == 1):
                        with open("E:\Computing_Power_Measurement\Intention\yolo_slowfast-master\DQN_accuracy.json",
                                  'r') as f:
                            data = json.load(f)
                        accuracy = data[j]
                        text = '{} {} {:.2f}%'.format(yolo_preds.names[int(cls)], ava_label, accuracy)  # 显示准确率
                        color = color_map[int(cls)]
                        im = self.plot_one_box(box, im, color, text, (vx, vy))
                    if (self.count == 2):
                        only_yolo = yolo_results[read_num]
                        accuracy = calculate_accuracy(box, only_yolo, 0.5)
                        text = '{} {} {:.2f}%'.format(yolo_preds.names[int(cls)], ava_label, accuracy)  # 显示准确率
                        color = color_map[int(cls)]
                        im = self.plot_one_box(box, im, color, text, (vx, vy))
                    if (self.count == 3):
                        only_zhudong = self.next[read_num]
                        accuracy = calculate_accuracy(box, only_zhudong, 0.5)
                        text = '{} {} {:.2f}%'.format(yolo_preds.names[int(cls)], ava_label, accuracy)  # 显示准确率
                        color = color_map[int(cls)]
                        im = self.plot_one_box(box, im, color, text, (vx, vy))
            im = im.astype(np.uint8)
            output_video.write(im)
            if vis:
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                cv2.imshow("demo", im)

    def main(self,config):
        # 初始化保存 deepsort 和 slowfast 输出的字典
        global read_num
        read_num = 0
        device = config.device
        imsize = config.imsize

        model = torch.hub.load('ultralytics/yolov5', 'yolov5l6').to(device)
        model.conf = config.conf
        model.iou = config.iou
        # 同时跟踪预测的数量
        # model.max_det = 100
        model.max_det = 5
        if config.classes:
            model.classes = config.classes

        video_model = slowfast_r50_detection(True).eval().to(device)

        deepsort_tracker = DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")
        ava_labelnames, _ = AvaLabeledVideoFramePaths.read_label_map("selfutils/temp.pbtxt")
        coco_color_map = [[random.randint(0, 255) for _ in range(3)] for _ in range(80)]

        vide_save_path = config.output
        video = cv2.VideoCapture(config.input)
        width, height = int(video.get(3)), int(video.get(4))
        video.release()
        outputvideo = cv2.VideoWriter(vide_save_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))
        print("processing...")

        cap = MyVideoCapture(config.input)
        id_to_ava_labels = {}
        a = time.time()
        while not cap.end:
            read_num = read_num + 1
            ret, img = cap.read()
            if not ret:
                continue
            yolo_preds = model([img], size=imsize)

            yolo_preds.files = ["img.jpg"]
            for detection in yolo_preds.pred:
                for box in detection:
                    x1, y1, x2, y2, conf, cls = box[:6]
                    detection_info = [x1.item(), y1.item(), x2.item(), y2.item(), cls.item(), conf.item()]
                    yolo_results.append(detection_info)
            deepsort_outputs = []
            for j in range(len(yolo_preds.pred)):
                temp = self.deepsort_update(deepsort_tracker, yolo_preds.pred[j].cpu(), yolo_preds.xywh[j][:, 0:4].cpu(),
                                       yolo_preds.ims[j])
                if len(temp) == 0:
                    temp = np.ones((0, 8))

                deepsort_outputs.append(temp.astype(np.float32))
            deepsort_results.append(deepsort_outputs)
            yolo_preds.pred = deepsort_outputs

            if len(cap.stack) == 25:
                #print(f"processing {cap.idx // 25}th second clips")
                clip = cap.get_video_clip()
                if yolo_preds.pred[0].shape[0]:
                    inputs, inp_boxes, _ = self.ava_inference_transform(clip, yolo_preds.pred[0][:, 0:4], crop_size=imsize)
                    inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0], 1), inp_boxes], dim=1)
                    if isinstance(inputs, list):
                        inputs = [inp.unsqueeze(0).to(device) for inp in inputs]
                    else:
                        inputs = inputs.unsqueeze(0).to(device)
                    with torch.no_grad():
                        slowfaster_preds = video_model(inputs, inp_boxes.to(device))
                        slowfaster_preds = slowfaster_preds.cpu()

                    for tid, avalabel in zip(yolo_preds.pred[0][:, 5].tolist(),
                                             np.argmax(slowfaster_preds, axis=1).tolist()):
                        id_to_ava_labels[tid] = ava_labelnames[avalabel + 1]

            self.save_yolopreds_tovideo(yolo_preds, id_to_ava_labels, coco_color_map, outputvideo, config.show)
        read_num = 0
        deepsort_results_json = [[item.tolist() for item in result] for result in deepsort_results]
        with open("deepsort_output.json", "w") as f1:
            json.dump(deepsort_results_json, f1)

        with open("yolo_results.json", "w") as f2:
            json.dump(yolo_results, f2)

        yolo_Jsondata = ReadJson()

        file1_data = yolo_Jsondata.read_json('E:\Computing_Power_Measurement\Intention\yolo_slowfast-master\deepsort_output.json')
        file2_data = yolo_Jsondata.read_json('E:\Computing_Power_Measurement\Intention\yolo_slowfast-master\yolo_results.json')

        deepsort_data = yolo_Jsondata.read2_and_fill2(file1_data)
        yolo_data = yolo_Jsondata.read_and_fill(file2_data)

        with open("deepsort2data.json", "w") as f3:
            json.dump(deepsort_data, f3)

        # 保存 slowfast 输出到 json 文件
        with open("yolo2data.json", "w") as f4:
            json.dump(yolo_data, f4)
        with open("no_intention.json","w") as fno:
            json.dump(self.no_intention_list,fno)
        with open("intention_noCommuniction.json","w") as fno2:
            json.dump(self.intention_noCommuniction_list,fno2)
        if (len(self.no_intention_list) == 0):
            no_Number = 1
        else:
            no_Number = len(self.no_intention_list)
        no_intention_list_mean = np.sum(self.no_intention_list) / no_Number
        if (len(self.intention_noCommuniction_list) == 0):
            communiction_Number = 1
        else:
            communiction_Number = len(self.intention_noCommuniction_list)
        intention_noCommuniction_list_mean = np.sum(self.intention_noCommuniction_list) / communiction_Number
        #print("no_intention_list",self.no_intention_list)
        #print("intention_noCommuniction_list",self.intention_noCommuniction_list)

        #print("no_intention_list_mean", no_intention_list_mean)
        #print("intention_noCommuniction_list_mean", intention_noCommuniction_list_mean)
        filled_data.clear()
        deepsort_results_json.clear()
        yolo_results.clear()
        deepsort_results.clear()

        # deepsort_data.clear()
        # yolo_data.clear()

        # with open("E:\Computing_Power_Measurement\Intention\yolo_slowfast-master\deepsort2data.json", 'r') as f5:
        #     deepsort2data = json.load(f5)
        # with open("E:\Computing_Power_Measurement\Intention\yolo_slowfast-master\yolo2data.json", 'r') as f6:
        #     yolo2data = json.load(f6)
        # print("deepsort_data")
        # print(deepsort2data)

        # print("yolo_data:")
        # print(yolo2data)
        # 计算并打印YOLO和DeepSORT的准确度
        # yolo_accuracy, deepsort_accuracy = calculate_accuracy(yolo2data, deepsort2data, 0.5)
        # print("YOLO Accuracy:", yolo_accuracy)
        # print("DeepSORT Accuracy:", deepsort_accuracy)

        print("total cost: {:.3f} s, video length: {} s".format(time.time() - a, cap.idx / 25))

        # print("yolo_results长度：", len(yolo_results))
        # print("deepsort_results_json长度：", len(deepsort_results_json))

        cap.release()
        outputvideo.release()
        print('saved video to:', vide_save_path)


    def yolo_deepsort_main(self):
        self.no_intention_list.clear()
        self.intention_noCommuniction_list.clear()
        deepsort_results.clear()
        yolo_results.clear()
        parser = argparse.ArgumentParser()
        parser.add_argument('--input', type=str,
                            default="E:\\Computing_Power_Measurement\\Intention\\yolo_slowfast-master\\Data\\night.mp4",
                            help='test imgs folder or video or camera')
        parser.add_argument('--output', type=str, default=self.name,
                            help='folder to save result imgs, can not use input folder')
        parser.add_argument('--imsize', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf', type=float, default=0.4, help='object confidence threshold')
        parser.add_argument('--iou', type=float, default=0.4, help='IOU threshold for NMS')
        parser.add_argument('--device', default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--show', action='store_true', help='show img')
        config = parser.parse_args()

        if config.input.isdigit():
            print("using local camera.")
            config.input = int(config.input)

        print(config)
        self.main(config)


def calculate_iou(box1, box2):

    # 获取交集的坐标
    x1_intersection = max(box1[0], box2[0])
    y1_intersection = max(box1[1], box2[1])
    x2_intersection = min(box1[2], box2[2])
    y2_intersection = min(box1[3], box2[3])

    # 计算交集区域的面积
    intersection_area = max(0, x2_intersection - x1_intersection + 1) * \
                        max(0, y2_intersection - y1_intersection + 1)

    # 计算两个边界框的面积
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # 计算并集的面积
    union_area = box1_area + box2_area - intersection_area

    # 计算交并比
    iou = intersection_area / union_area if union_area > 0 else 0

    #print("iou",iou)

    return iou
def calculate_accuracy(yolo_predictions, qiyu, threshold):
    # 计算YOLO预测轨迹的准确度
    #print("开始计算准确度")
    #自己方法
    qiyu_accuracy = calculate_iou(yolo_predictions, qiyu)
    #不进行意图
    # 计算DeepSORT跟踪轨迹的准确度
    #被动意图但不通信
    return qiyu_accuracy


if __name__ == "__main__":
    # ym = Yolo_Main(0,"output.mp4")
    # ym.yolo_deepsort_main()

    # ym = Yolo_Main(1, "1output.mp4")
    # ym.yolo_deepsort_main()

    # ym = Yolo_Main(2, "2output.mp4")
    # ym.yolo_deepsort_main()

    ym = Yolo_Main(3, "3output.mp4")
    ym.yolo_deepsort_main()