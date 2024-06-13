import gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import rl_utils
from v2vTestenvironment import *
from yolo_deepsort import *

lr = 2e-3
num_episodes = 10
hidden_dim = 128
gamma = 0.95
epsilon = 0.03
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name1 = 'yolo_deepsort_Reward'
env_name2 = 'yolo_accuracy'
env_name3 = 'deepsort_accuracy'

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
replay_buffer = ReplayBuffer(buffer_size)
state_dim = 4
action_dim = 4
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
            target_update, device)

yolo_accuracy_list = []
deepsort_accuracy_list = []
return_list = []
DQN_accuracy_list = []

for i in range(10):  # 10
    ym = Yolo_Main(0, "output.mp4")
    ym.yolo_deepsort_main()
    with open("E:\Computing_Power_Measurement\Intention\yolo_slowfast-master\yolo2data.json", 'r') as f6:
        yolo2data = json.load(f6)
    length = len(yolo2data)
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            state = yolo2data[0]
            done = False
            ver_CustomEnvironment = CustomEnvironment()
            while not done:
                if ver_readnum > length:
                    ver_readnum = 0
                if ver_readnum == length:
                    ver_readnum = length - 1
                ver_readnum = ver_readnum + 1

                # 获取周围车辆的状态信息
                surrounding_vehicles_states = {}  # 这里需要根据实际情况获取周围车辆的状态信息

                action = agent.take_action(state)
                next_state, reward, done, yolo_accuracy, deepsort_accuracy, DQN_accuracy, v2v_communication_delay = \
                    ver_CustomEnvironment.custom_environment_step(ver_readnum, action, surrounding_vehicles_states)

                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward

                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    agent.update(transition_dict)
            ver_readnum = 0
            return_list.append(episode_return)
            DQN_accuracy_list.append(DQN_accuracy)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return': '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)

    yolo_accuracy_list.append(max(yolo_accuracy_max_list))
    deepsort_accuracy_list.append(max(deepsort_accuracy_max_list))
    yolo_accuracy_max_list.clear()
    deepsort_accuracy_max_list.clear()

# 保存模型
torch.save(agent.q_net.state_dict(), 'dqn_model.pth')

# 加载模型
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)
agent.q_net.load_state_dict(torch.load('dqn_model.pth'))
agent.q_net.eval()

with open("DQN_accuracy.json", "w") as f0:
    json.dump(DQN_accuracy_list, f0)

print("return_list", return_list)
print("开始画图")
with open("yolo_accuracy.json", "w") as fa:
    json.dump(yolo_accuracy_list, fa)

with open("deepsort_accuracy.json", "w") as fb:
    json.dump(deepsort_accuracy_list, fb)

print("yolo_accuracy_list", yolo_accuracy_list)
print("deepsort_accuracy_list", deepsort_accuracy_list)

with open("return_list.json", "w") as fc:
    json.dump(return_list, fc)
