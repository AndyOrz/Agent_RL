import numpy as np
import pandas as pd


class Agent:
    def __init__(self,
                 env,
                 data_path=None,
                 gamma=0.9,
                 learning_rate=0.1,
                 epsilon=.1):  # 初始化参数奖励衰减系数0.9，学习率0.1

        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.time_count = env.time_count
        self.action_n = env.action_space
        self.x_count = env.x_count
        self.y_count = env.y_count

        if data_path is None:
            self.qshape = (self.x_count, self.y_count, self.time_count,
                           self.x_count, self.y_count, self.action_n)
            self.q = np.zeros(self.qshape)
            # 边界初始化
            for des_x in range(self.x_count):
                for des_y in range(self.y_count):
                    for t in range(self.time_count):
                        for x in range(self.x_count):
                            self.q[des_x, des_y, t, x, 0,
                                   0] = np.finfo(np.float32).min
                            self.q[des_x, des_y, t, x, 0,
                                   1] = np.finfo(np.float32).min
                            self.q[des_x, des_y, t, x, 0,
                                   7] = np.finfo(np.float32).min
                            self.q[des_x, des_y, t, x, self.y_count - 1,
                                   3] = np.finfo(np.float32).min
                            self.q[des_x, des_y, t, x, self.y_count - 1,
                                   4] = np.finfo(np.float32).min
                            self.q[des_x, des_y, t, x, self.y_count - 1,
                                   5] = np.finfo(np.float32).min

                        for y in range(self.y_count):
                            self.q[des_x, des_y, t, 0, y,
                                   5] = np.finfo(np.float32).min
                            self.q[des_x, des_y, t, 0, y,
                                   6] = np.finfo(np.float32).min
                            self.q[des_x, des_y, t, 0, y,
                                   7] = np.finfo(np.float32).min
                            self.q[des_x, des_y, t, self.x_count - 1, y,
                                   1] = np.finfo(np.float32).min
                            self.q[des_x, des_y, t, self.x_count - 1, y,
                                   2] = np.finfo(np.float32).min
                            self.q[des_x, des_y, t, self.x_count - 1, y,
                                   3] = np.finfo(np.float32).min
        else:
            self.q = np.load(data_path)

    def _obs2txy(self, obs):
        t = int(obs['time']) % 24
        x = obs['position'][0]
        y = obs['position'][1]
        return t, x, y

    def decide(self, obs, des_x, des_y):  # 智能体决策，epsilon贪心决策，输入参数状态
        t, x, y = self._obs2txy(obs)
        rand = np.random.uniform(0, 1)
        if rand > self.epsilon:
            # 需要洗牌，防止选idxmax时一直为最大值中最小的索引
            state_action = pd.Series(data=np.array(self.q[des_x, des_y, t, x,
                                                          y, :]))
            state_action = state_action.reindex(
                np.random.permutation(state_action.index))
            action = state_action.idxmax()
        else:
            action = np.random.randint(self.action_n)
        return action

    def learnQ(self, obs, action, reward, next_obs, done, des_x,
               des_y):  # 智能体学习，Q-learning
        t, x, y = self._obs2txy(obs)
        nt, nx, ny = self._obs2txy(next_obs)
        # print("此轮更新行动"+str(action))
        u = reward + self.gamma * self.q[des_x, des_y, nt, nx,
                                         ny].max() * (1. - done)
        td_error = u - self.q[des_x, des_y, t, x, y, action]
        self.q[des_x, des_y, t, x, y,
               action] += self.learning_rate * td_error  # 更新Q表


'''
    def learnS(self, obs, action, reward, next_obs,
               done):  # 智能体学习，SARSA-learning
        state = self._obs2state(obs)
        next_state = self._obs2state(next_obs)

        v = (self.q[des_x, des_y, next_state].sum() * self.epsilon +
             self.q[des_x, des_y, next_state].max() * (1. - self.epsilon)
             )  # 计算（next_state,next_action)Q值期望
        u = reward + self.gamma * v * (1. - done)
        td_error = u - self.q[des_x, des_y, state, action]
        self.q[des_x, des_y, state,
               action] += self.learning_rate * td_error  # 更新Q表
'''
