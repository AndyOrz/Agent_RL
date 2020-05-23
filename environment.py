import numpy as np
import math

# gym.Env是gym的环境基类,自定义的环境就是根据自己的需要重写其中的方法；
# 要重写的方法有:
# __init__()：构造函数
# reset()：初始化环境
# step()：环境动作,即环境对agent的反馈
# render()：如果要进行可视化则实现


class Environment():
    def __init__(self, file_path="./data/"):
        # 参数
        self.time_count = 24
        self.x_count = 35  # 经度
        self.y_count = 25  # 纬度
        self.action_space = 9
        self.time_space = 0.1  # 每一步增加的时间(小时)

        # 加载数据
        self.supply_list = []
        self.speed_list = []
        self.demand_list = []
        for i in range(0, self.time_count):
            self.supply_list.append(
                np.loadtxt(open(
                    file_path + "Su22/22Su-" + str(i) + ".txt", "rb"),
                           delimiter=",",
                           skiprows=0).T)
            self.speed_list.append(
                np.loadtxt(open(file_path + "V22/22V-" + str(i) + ".txt",
                                "rb"),
                           delimiter=",",
                           skiprows=0).T)
            self.demand_list.append(
                np.loadtxt(open(file_path + "D22/22D-" + str(i) + ".txt",
                                "rb"),
                           delimiter=",",
                           skiprows=0).T)
        self.dest_x = 0
        self.dest_y = 0

        # 初始状态
        self.obs = {"time": 0, "position": [0, 0]}

        # 动作空间，8个方向(0-8)，顺时针表示北，东北，东，东南，南，西南，西，西北，原地

        # 状态空间，时间(小时)×空间(经度，纬度)

    def reset(self):
        obs = self.set_obs(np.random.randint(self.time_count),
                           np.random.randint(self.x_count),
                           np.random.randint(self.y_count))
        self.available = True
        return obs

    def step(self, action):
        obs = self._get_observation(action)
        self.obs = obs
        reward, done, method = self._get_reward(action)
        info = {"method": method}  # 用于记录训练过程中的环境信息,便于观察训练状态

        return obs, reward, done, info

    def set_obs(self, time, x, y):  # 设置当前状态
        obs = {"time": time, "position": [x, y]}
        self.obs = obs
        return obs

    def get_obs(self):  # 获得当前状态
        return self.obs.copy()

    def set_des(self, x, y):  # 设置终点
        self.dest_x = x
        self.dest_y = y

    # 根据需要设计相关辅助函数
    def _get_observation(self, action):
        obs = self.obs
        obs["time"] += self.time_space
        x = obs["position"][0]
        y = obs["position"][1]
        # 北，东北，东，东南，南，西南，西，西北，原地
        if (action == 0):
            y -= 1
        elif (action == 1):
            x += 1
            y -= 1
        elif (action == 2):
            x += 1
        elif (action == 3):
            x += 1
            y += 1
        elif (action == 4):
            y += 1
        elif (action == 5):
            x -= 1
            y += 1
        elif (action == 6):
            x -= 1
        elif (action == 7):
            x -= 1
            y -= 1

        if (x < 0 or y < 0 or x >= self.x_count or y >= self.y_count):  # 越界处理
            pass
        else:
            obs["position"][0] = x
            obs["position"][1] = y

        return obs

    def _get_reward(self, action):
        obs = self.obs

        t = int(obs["time"]) % 24
        x = obs["position"][0]
        y = obs["position"][1]
        Su = self.supply_list[t][x][y]
        V = self.speed_list[t][x][y]
        D = self.demand_list[t][x][y]
        done = False
        method = ""

        if V == 0:
            taking_rate = 0
        else:
            taking_rate = 2 / V
        P = 1 - math.exp(-taking_rate * D / (1 + taking_rate * Su))
        random = np.random.rand()
        if (P >= random):
            method = "Pick"
            done = True
            return 10, done, method
        else:
            R1 = 0

        detx = self.dest_x - x
        dety = self.dest_y - y

        if (detx == 0 and dety == 0):
            method = "Destination"
            done = True
            R2 = 2  # 到达终点奖励2
        else:
            if (dety < 0):
                a = math.pi + math.atan(detx / dety)
            elif dety == 0:
                a = math.pi * (1 - 0.5 * abs(detx) / detx)
            else:
                if detx >= 0:
                    a = math.atan(detx / dety)
                else:
                    a = 2 * math.pi + math.atan(detx / dety)

            deta = abs((math.pi / 4) * action - a)
            dis = math.sqrt(detx * detx + dety * dety)

            # 修正R2
            R2 = -0.05 * deta / dis

        # 修正R3

        R3 = -math.exp(-0.01 * V)

        return R1 + R2 + R3, done, method
