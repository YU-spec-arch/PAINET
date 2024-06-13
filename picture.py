import json

import numpy as np
from matplotlib import pyplot as plt

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    deepsort2data = read_json("E:\Computing_Power_Measurement\Intention\yolo_slowfast-master\deepsort2data.json")
    print("deepsort2data",len(deepsort2data))
    # yolo_accuracy_list = read_json("E:\\Computing_Power_Measurement\\Intention\\yolo_slowfast-master\\yolo_accuracy.json")
    # deepsort_accuracy_list = read_json("E:\\Computing_Power_Measurement\\Intention\\yolo_slowfast-master\\deepsort_accuracy.json")
    yolo_accuracy_list = read_json(
        "E:\Computing_Power_Measurement\Intention\yolo_slowfast-master\yolo_accuracy_list.json")
    deepsort_accuracy_list = read_json(
        "E:\Computing_Power_Measurement\Intention\yolo_slowfast-master\deepsort_accuracy_list.json")
    return_list = read_json("E:\\Computing_Power_Measurement\\Intention\\yolo_slowfast-master\\return_list.json")

    mean_DQNaccuracy = read_json("E:\Computing_Power_Measurement\Intention\yolo_slowfast-master\DQN_accuracy.json")
    # YOLO精度
    mean_accuracy = np.mean(mean_DQNaccuracy)
    print("mean_accuracy",mean_accuracy)
    episodes_list = list(range(len(yolo_accuracy_list)))
    plt.plot(episodes_list, yolo_accuracy_list)
    plt.xlabel('Episodes')
    plt.ylabel('YOLO Accuracy')
    plt.show()
    plt.savefig("YOLO Accuracy.png")

    # Deepsort精度
    # episodes_list = list(range(len(deepsort_accuracy_list)))
    # plt.plot(episodes_list, deepsort_accuracy_list)
    # plt.xlabel('Episodes')
    # plt.ylabel('Deepsort Accuracy')
    # plt.show()
    # plt.savefig("Deepsort Accuracy.png")

    # # 奖励
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('DQN on {reward}')
    plt.show()
    plt.savefig("reward.png")
    #
    # # 移动平均奖励
    # mv_return = rl_utils.moving_average(return_list, 10)  # 9
    # episodes_list = list(range(len(mv_return)))
    #
    # plt.plot(episodes_list, mv_return)
    # plt.xlabel('Episodes')
    # plt.ylabel('Moving Average Reward')
    # plt.title('DQN on {mean reward}')
    # plt.show()