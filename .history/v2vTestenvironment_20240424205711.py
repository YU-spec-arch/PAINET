import json
import time

from DQNetwork import *
from yolo_deepsort import *

yolo_accuracy_max_list = []
deepsort_accuracy_max_list = []

class CustomEnvironment:
    def __init__(self):
        self.v2v_communication_delay = 0
        self.reward = 0
        self.DQN_accuracy = 0

    def calculate_iou(self, box1, box2):
        x1_intersection = max(box1[0], box2[0])
        y1_intersection = max(box1[1], box2[1])
        x2_intersection = min(box1[2], box2[2])
        y2_intersection = min(box1[3], box2[3])
        intersection_area = max(0, x2_intersection - x1_intersection + 1) * \
                            max(0, y2_intersection - y1_intersection + 1)
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
        union_area = box1_area + box2_area - intersection_area
        iou = intersection_area / union_area if union_area > 0 else 0
        return iou

    def calculate_accuracy(self, total_length, ver_readnum, yolo_predictions, deepsort_tracks, threshold):
        self.yolo_accuracy = self.calculate_iou(yolo_predictions, deepsort_tracks)
        if len(yolo_accuracy_max_list) == 0:
            yolo_accuracy_max_list.append(0)
        if self.yolo_accuracy > 0:
            yolo_accuracy_max_list.append(self.yolo_accuracy)

        self.deepsort_accuracy = self.calculate_iou(deepsort_tracks, yolo_predictions)
        if len(deepsort_accuracy_max_list) == 0:
            deepsort_accuracy_max_list.append(0)
        if self.deepsort_accuracy > 0:
            deepsort_accuracy_max_list.append(self.deepsort_accuracy)

        return self.yolo_accuracy, self.deepsort_accuracy

    def Testen_direction(self, yolo_predictions, deepsort_tracks):
        yolo_x_center = (yolo_predictions[0] + yolo_predictions[2]) / 2
        yolo_y_center = (yolo_predictions[1] + yolo_predictions[3]) / 2
        deepsort_x_center = (deepsort_tracks[0] + deepsort_tracks[2]) / 2
        deepsort_y_center = (deepsort_tracks[1] + deepsort_tracks[3]) / 2
        direction = 3
        if deepsort_x_center > yolo_x_center:
            direction = 0  # Right
        elif deepsort_x_center < yolo_x_center:
            direction = 1  # Left
        if deepsort_y_center < yolo_y_center:
            direction = 3  # Down
        else:
            direction = 2  # Up
        return direction

    def v2v_communication(self, my_vehicle_state, surrounding_vehicles_states):
        start_time = time.time()
        nearby_vehicle_states = {}
        for idx, state in enumerate(surrounding_vehicles_states):
            nearby_vehicle_states[idx] = state
        end_time = time.time()
        self.v2v_communication_delay = end_time - start_time
        print("self.v2v_communication_delay", self.v2v_communication_delay)
        return nearby_vehicle_states


    def custom_environment_step(self, ver_readnum, action):
        self.reward = 0  # 初始化奖励
        self.DQN_accuracy = 0  # 初始化DQN准确度

        with open(r"E:\Computing_Power_Measurement\Intention\yolo_slowfast-master\deepsort2data.json", 'r') as f5:
            deepsort2data = json.load(f5)
        with open(r"E:\Computing_Power_Measurement\Intention\yolo_slowfast-master\yolo2data.json", 'r') as f6:
            yolo2data = json.load(f6)
        my_vehicle_state = yolo2data[ver_readnum]
        surrounding_vehicles_states = yolo2data[ver_readnum + 1]
        print("my_vehicle_state", my_vehicle_state)
        nearby_vehicle_states = self.v2v_communication(my_vehicle_state, surrounding_vehicles_states)
        yolo_accuracy, deepsort_accuracy = self.calculate_accuracy(len(yolo2data), ver_readnum,
                                                                    yolo2data[ver_readnum], deepsort2data[ver_readnum], 0.5)
        next_state = yolo2data[ver_readnum]
        direction_list = self.Testen_direction(yolo2data[ver_readnum], deepsort2data[ver_readnum])
        reward = self.Testen_reward_function(action, direction_list, yolo_accuracy)
        if ver_readnum + 1 >= len(yolo2data):
            done = True
        else:
            done = False
        return next_state, reward, done, yolo_accuracy, deepsort_accuracy, self.v2v_communication_delay

    def Testen_reward_function(self, action_list, direction_list, yolo_accuracy):
        if action_list != direction_list:
            self.reward = self.reward - 1
        if action_list == direction_list:
            self.DQN_accuracy = self.DQN_accuracy + 1
            self.reward = self.reward + 20 * yolo_accuracy
        return self.reward


# 初始化全局变量
