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

def tensor_to_numpy(tensor):
    img = tensor.cpu().numpy().transpose((1, 2, 0))
    return img

def ava_inference_transform(
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

def plot_one_box(x, img, color=[100, 100, 100], text_info="None",
                 velocity=None, thickness=1, fontsize=0.5, fontthickness=1):
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
        cv2.arrowedLine(img, (x_center, y_center), (int(x_center + dx * scale), int(y_center + dy * scale)),
                        (0, 0, 255), 2)

        direction = ""
        if c2[0] > c1[0]:
            direction = "Right"
        elif  c2[0] < c1[0]:
            direction = "Left"
        elif c2[1] > c1[1] + 10:
            direction += "Down"
        else:
            direction += "Up"

        color1 = [255, 0, 0]
        cv2.putText(img, direction, (c1[0], c1[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, fontsize, color1, fontthickness)

    return img

def deepsort_update(Tracker, pred, xywh, np_img):
    outputs = Tracker.update(xywh, pred[:, 4:5], pred[:, 5].tolist(), cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
    return outputs

def save_yolopreds_tovideo(yolo_preds, id_to_ava_labels, color_map, output_video, vis=False):
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
                text = '{} {} {}'.format(int(trackid), yolo_preds.names[int(cls)], ava_label)
                color = color_map[int(cls)]
                im = plot_one_box(box, im, color, text, (vx, vy))
        im = im.astype(np.uint8)
        output_video.write(im)
        if vis:
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            cv2.imshow("demo", im)

def main(config):
    # 初始化保存 deepsort 和 slowfast 输出的字典
    global deepsort_results,yolo_results
    deepsort_results = {}
    yolo_results = {}

    device = config.device
    imsize = config.imsize

    model = torch.hub.load('ultralytics/yolov5', 'yolov5l6').to(device)
    model.conf = config.conf
    model.iou = config.iou
    model.max_det = 1
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
        ret, img = cap.read()
        if not ret:
            continue
        yolo_preds = model([img], size=imsize)

        yolo_preds.files = ["img.jpg"]
        # 遍历 yolo_preds.pred 获取每个物体的检测结果
        for detection in yolo_preds.pred:
            # 提取边界框坐标、类别标签、置信度等信息
            for box in detection:
                #x1, y1, x2, y2, conf, cls = box[:6].cpu().detach().numpy()  # 将张量转换为标量
                x1, y1, x2, y2, conf, cls = box[:6]
                # 创建一个包含当前检测结果的列表，并添加到结果数组中
                detection_info = [x1.item(), y1.item(), x2.item(), y2.item(), cls.item(), conf.item()]
                yolo_results.append(detection_info)

                print("YOLO------->", detection_info)
        deepsort_outputs = []
        for j in range(len(yolo_preds.pred)):
            temp = deepsort_update(deepsort_tracker, yolo_preds.pred[j].cpu(), yolo_preds.xywh[j][:, 0:4].cpu(),
                                   yolo_preds.ims[j])
            if len(temp) == 0:
                temp = np.ones((0, 8))
            deepsort_outputs.append(temp.astype(np.float32))
        """
        在这个代码段中，`outputs.append(np.array([x1,y1,x2,y2,label,track_id,Vx,Vy], dtype=np.int32))` 
        将一些跟踪对象的信息添加到 `outputs` 列表中。具体来说：
        - `x1``y1`、`x2`、`y2` 是目标边界框的左上角和右下角的坐标。
        - `label` 是目标的标签。
        - `track_id` 是跟踪对象的唯一标识符。
        - `Vx`、`Vy` 是跟踪对象在水平和垂直方向上的速度。
        这些信息合在一起，构成了对目标的基本描述，可以用于在视觉任务中进一步处理或显示跟踪结果。
        """
        print("deepsort------->", deepsort_outputs)
        deepsort_results.append(deepsort_outputs)
        
        yolo_preds.pred = deepsort_outputs

        if len(cap.stack) == 25:
            print(f"processing {cap.idx // 25}th second clips")
            clip = cap.get_video_clip()
            if yolo_preds.pred[0].shape[0]:
                inputs, inp_boxes, _ = ava_inference_transform(clip, yolo_preds.pred[0][:, 0:4], crop_size=imsize)
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

        save_yolopreds_tovideo(yolo_preds, id_to_ava_labels, coco_color_map, outputvideo, config.show)
    print("total cost: {:.3f} s, video length: {} s".format(time.time() - a, cap.idx / 25))

    # 保存 deepsort 输出到 json 文件
    with open("deepsort_output.json", "w") as f:
        json.dump(deepsort_results, f)

    # 保存 slowfast 输出到 json 文件
    with open("yolo_results.json", "w") as f:
        json.dump(yolo_results, f)

    cap.release()
    outputvideo.release()
    print('saved video to:', vide_save_path)


if __name__ == "__main__":
    deepsort_results.clear
    yolo_results.clear
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        default="E:\\Computing_Power_Measurement\\Intention\\yolo_slowfast-master\\Data\\night.mp4",
                        help='test imgs folder or video or camera')
    parser.add_argument('--output', type=str, default="output.mp4",
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
    main(config)
