import torch
import json
from Testenvironment import *
# 假设您的agent定义和CustomEnvironment在这里
# from your_dqn_definition_file import DQN
# from your_environment_definition_file import CustomEnvironment

# 加载模型
state_dim = 4
hidden_dim = 128
action_dim = 4
lr = 2e-3
gamma = 0.98
epsilon = 0.01
target_update = 10
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)
agent.q_net.load_state_dict(torch.load('dqn_model.pth'))
agent.q_net.eval()

# 实例化环境
env = CustomEnvironment()


def run_model(agent, env):
    state = env.reset()  # 假设您的环境有一个reset方法可以获取初始状态
    done = False
    total_reward = 0
    while not done:
        action = agent.take_action(state)  # 使用模型选择动作
        next_state, reward, done, _ = env.step(action)  # 假设您的环境有一个step方法来执行动作
        state = next_state
        total_reward += reward

    print(f"Total reward from the episode: {total_reward}")


# 运行模型
run_model(agent, env)