import json

from DQNetwork import *
from yolo_deepsort import *

yolo_accuracy_max_list = []
deepsort_accuracy_max_list = []

class CustomEnvironment:
    def __init__(self):
    #def __init__(self,yolo_correct,deepsort_correct):
        self.reward = 0
        # self.yolo_correct = yolo_correct
        # self.deepsort_correct = deepsort_correct
        self.yolo_correct = 0
        self.deepsort_correct = 0
        self.yolo_accuracy = 0
        self.deepsort_accuracy = 0
        self.DQN_accuracy = 0

    def calculate_iou(self,box1, box2):

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
    def calculate_accuracy(self,total_length,ver_readnum,yolo_predictions, deepsort_tracks, threshold):
        # 计算YOLO预测轨迹的准确度
        #print("开始计算准确度")
        self.yolo_accuracy = self.calculate_iou(yolo_predictions, deepsort_tracks)
        if len(yolo_accuracy_max_list) == 0:
            yolo_accuracy_max_list.append(0)
        if self.yolo_accuracy > 0:
            yolo_accuracy_max_list.append(self.yolo_accuracy)

        # 计算DeepSORT跟踪轨迹的准确度

        self.deepsort_accuracy = self.calculate_iou(deepsort_tracks, yolo_predictions)
        if len(deepsort_accuracy_max_list) == 0:
            deepsort_accuracy_max_list.append(0)
        if self.deepsort_accuracy > 0:
            deepsort_accuracy_max_list.append(self.deepsort_accuracy)
        # if self.yolo_accuracy >= 0:
        #     if self.yolo_accuracy == 0:
        #         yolo_accuracy_max_list.append(0)
        #     if self.yolo_accuracy > 0:
        #         yolo_accuracy_max_list.append(self.yolo_accuracy)
        #
        # # 计算DeepSORT跟踪轨迹的准确度
        #
        # self.deepsort_accuracy = self.calculate_iou(deepsort_tracks, yolo_predictions)
        # if self.deepsort_accuracy >= 0:
        #     if self.deepsort_accuracy == 0:
        #         deepsort_accuracy_max_list.append(0)
        #     if self.deepsort_accuracy > 0:
        #         deepsort_accuracy_max_list.append(self.deepsort_accuracy)


        return self.yolo_accuracy, self.deepsort_accuracy

    # def calculate_iou(self,box1, box2,var_action):
    #
    #     # 获取交集的坐标
    #     direction = self.Testen_direction(box1,box2)
    #
    #     if var_action == direction:
    #         iou = 1
    #     else:
    #         iou = 0
    #     return iou

    # def calculate_accuracy(self,total_length,ver_readnum,yolo_predictions, deepsort_tracks, threshold):
    #     # 计算YOLO预测轨迹的准确度
    #     print("开始计算准确度")
    #     if self.calculate_iou(yolo_predictions, deepsort_tracks) > threshold:
    #         self.yolo_correct += 1
    #
    #     # 计算DeepSORT跟踪轨迹的准确度
    #
    #     if self.calculate_iou(deepsort_tracks, yolo_predictions) > threshold:
    #         self.deepsort_correct += 1
    #     if ver_readnum + 1 >= total_length:
    #         self.yolo_accuracy = self.yolo_correct / total_length
    #         self.deepsort_accuracy = self.deepsort_correct / total_length
    #     # print("self.yolo_accuracy",self.yolo_accuracy)
    #     # print("self.deepsort_accuracy", self.deepsort_accuracy)
    #     return self.yolo_accuracy, self.deepsort_accuracy
    def Testen_direction(self,yolo_predictions, deepsort_tracks):
        ###print("yolo_predictions[0]",yolo_predictions[0])
        ###print("deepsort_tracks[0]", deepsort_tracks[0])
        yolo_x_center = (yolo_predictions[0] + yolo_predictions[2]) / 2
        yolo_y_center = (yolo_predictions[1] + yolo_predictions[3]) / 2

        deepsort_x_center = (deepsort_tracks[0] + deepsort_tracks[2]) / 2
        deepsort_y_center = (deepsort_tracks[1] + deepsort_tracks[3]) / 2

        direction = 3
        if deepsort_x_center > yolo_x_center:
            #"Right"
            direction = 0
        elif deepsort_x_center < yolo_x_center:
            # "Left"
            direction = 1
        if deepsort_y_center < yolo_y_center:
            # "Down"
            direction = 3
        else:
            #"Up"
            direction = 2
        return direction


 # TODO:
    def custom_environment_step(self,ver_readnum,action):
        DQN_accuracy = 0
        with open("E:\Computing_Power_Measurement\Intention\yolo_slowfast-master\deepsort2data.json", 'r') as f5:
            deepsort2data = json.load(f5)
        with open("E:\Computing_Power_Measurement\Intention\yolo_slowfast-master\yolo2data.json", 'r') as f6:
            yolo2data = json.load(f6)
        # print("deepsort_data")
        # print(deepsort2data)

        # print("yolo_data:")
        # print(yolo2data)
        # 计算并打印YOLO和DeepSORT的准确度
        #print("准备计算准确度")
        yolo_accuracy, deepsort_accuracy = self.calculate_accuracy(len(yolo2data),ver_readnum,yolo2data[ver_readnum], deepsort2data[ver_readnum],0.5)
        ###print("YOLO Accuracy:", yolo_accuracy)
        ###print("DeepSORT Accuracy:", deepsort_accuracy)
        next_state = yolo2data[ver_readnum]
        direction_list = self.Testen_direction(yolo2data[ver_readnum],deepsort2data[ver_readnum])
        reward = self.Testen_reward_function(action,direction_list,yolo_accuracy)
        if ver_readnum + 1 >= len(yolo2data):
            done = True
            if done == True:
                DQN_accuracy = self.DQN_accuracy / len(yolo2data)

            #
            #     if os.path.exists("deepsort_output.json"):
            #         print("找到文件")
            #         os.remove("deepsort_output.json")
            #     if os.path.exists("yolo_results.json"):
            #         os.remove("yolo_results.json")
            #     if os.path.exists("deepsort2data.json"):
            #         os.remove("deepsort2data.json")
            #     if os.path.exists("yolo2data.json"):
            #         os.remove("yolo2data.json")

        else:
            done = False
        return next_state,reward, done,yolo_accuracy, deepsort_accuracy,DQN_accuracy

    def Testen_reward_function(self,action_list,direction_list,yolo_accuracy):
        if action_list != direction_list:
            self.reward = self.reward - 1
        if action_list == direction_list:
            self.DQN_accuracy = self.DQN_accuracy + 1
            self.reward = self.reward + 20 * yolo_accuracy
        return self.reward


# 初始化全局变量

