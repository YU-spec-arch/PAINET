import json
import time


from yolo_deepsort import *

yolo_accuracy_max_list = []
deepsort_accuracy_max_list = []

class CustomEnvironment:
    def __init__(self):
        self.v2v_communication_delay = 0
        self.reward = 0
        self.DQN_accuracy = 0
        self.DQN_number = 0

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

    def calculate_accuracy(self, total_length, ver_readnum, yolo_predictions, deepsort_tracks, threshold,nearby_vehicle_states):
        #print("yolo_predictions",yolo_predictions)
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

    def Testen_direction(self, deepsort_tracks0, deepsort_tracks,nearby_vehicle_states):
        yolo_x_center = (deepsort_tracks0[0] + deepsort_tracks0[2]) / 2
        yolo_y_center = (deepsort_tracks0[1] + deepsort_tracks0[3]) / 2
        deepsort_x_center = (deepsort_tracks[0] + deepsort_tracks[2]) / 2
        deepsort_y_center = (deepsort_tracks[1] + deepsort_tracks[3]) / 2

        qita_x_center = (nearby_vehicle_states[0] + nearby_vehicle_states[2]) / 2
        qita_y_center = (nearby_vehicle_states[0] + nearby_vehicle_states[2]) / 2

        state_direction0 = deepsort_tracks[0] - deepsort_tracks0[0]
        state_direction1 = deepsort_tracks[1] - deepsort_tracks0[1]
        state_direction2 = deepsort_tracks[2] - deepsort_tracks0[2]
        state_direction3 = deepsort_tracks[3] - deepsort_tracks0[3]
        state_direction = [state_direction0,state_direction1,state_direction2,state_direction3]
        if qita_y_center > yolo_y_center :
            direction = 3
        if deepsort_x_center - yolo_x_center > 0:
            direction = 0  # Right
        elif deepsort_x_center - yolo_x_center < 0:
            direction = 1  # Left
        if deepsort_y_center - yolo_y_center  < 0:
            direction = 3  # Down
        else:
            direction = 2  # Up
        return direction,state_direction

    def v2v_communication(self, my_vehicle_state, surrounding_vehicles_states):
        start_time = time.time()
        nearby_vehicle_states = {}
        for idx, state in enumerate(surrounding_vehicles_states):
            nearby_vehicle_states[idx] = state
        end_time = time.time()
        self.v2v_communication_delay = end_time - start_time
        # print("self.v2v_communication_delay", self.v2v_communication_delay)

        # 将字典的值转换成列表
        nearby_vehicle_states_list = list(nearby_vehicle_states.values())

        return nearby_vehicle_states_list


    def custom_environment_step(self, ver_readnum, action):
        DQN_accuracy = 0
        ver_readnum1 = ver_readnum + 1

        with open(r"E:\Computing_Power_Measurement\Intention\yolo_slowfast-master\deepsort2data.json", 'r') as f5:
            deepsort2data = json.load(f5)
        with open(r"E:\Computing_Power_Measurement\Intention\yolo_slowfast-master\yolo2data.json", 'r') as f6:
            yolo2data = json.load(f6)
        my_vehicle_state = yolo2data[ver_readnum]
        if ver_readnum1  >= len(yolo2data):
            ver_readnum1 =  ver_readnum - 1
        surrounding_vehicles_states = yolo2data[ver_readnum1]
        #print("my_vehicle_state", my_vehicle_state)
        nearby_vehicle_states = self.v2v_communication(my_vehicle_state, surrounding_vehicles_states)
        #print("nearby_vehicle_states",nearby_vehicle_states)
        yolo_accuracy, deepsort_accuracy = self.calculate_accuracy(len(yolo2data), ver_readnum,
                                                                    yolo2data[ver_readnum], deepsort2data[ver_readnum], 0.5,nearby_vehicle_states)

        #next_state = deepsort2data[ver_readnum]
        direction_list,state_direction = self.Testen_direction(deepsort2data[ver_readnum], deepsort2data[ver_readnum1],nearby_vehicle_states)
        next_state = state_direction
        reward = self.Testen_reward_function(len(yolo2data),action, direction_list)
        # if ver_readnum + 1 >= len(yolo2data):
        #     ver_readnum = 0
        if self.DQN_accuracy > 0.96:
            done = True
        else:
            done = False
        return next_state, reward, done, yolo_accuracy, deepsort_accuracy,self.DQN_accuracy

    def Testen_reward_function(self, length,action_list, direction_list):
        #print("action_list",action_list)
        #print("direction_list",direction_list)
        if action_list != direction_list:
            self.reward = self.reward - 1
        if action_list == direction_list:
            self.DQN_number = self.DQN_number + 1
            self.DQN_accuracy = self.DQN_number / length
            #print(" self.DQN_accuracy", self.DQN_accuracy)
            self.reward = self.reward + 20 * self.DQN_accuracy
        return self.reward


# 初始化全局变量
