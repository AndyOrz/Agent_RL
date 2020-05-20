import sys
import numpy as np
# from tqdm import *
import agent
import environment


# 智能体与环境交互
def Interact(env, agent, des_x, des_y, train=False):
    episode_reward = 0
    round_count = 0
    observation = env.reset()
    env.set_des(des_x, des_y)
    while True:
        round_count = round_count + 1
        obs = env.get_obs()
        action = agent.decide(observation, des_x, des_y)
        next_observation, reward, done, info = env.step(action)
        # print(episode_reward)
        episode_reward += reward
        if train:
            agent.learnQ(obs, action, reward, next_observation, done, des_x,
                         des_y)
        if done:
            break
        observation = next_observation
    return episode_reward, round_count  # 返回回合奖励和本回合的步数


# 模型训练
if __name__ == "__main__":

    env = environment.Environment()

    # 指定回合数训练
    episodes = int(sys.argv[1])
    episode_rewards = []
    if (len(sys.argv) == 2):
        Agent = agent.Agent(env)
    elif (len(sys.argv) == 3):
        Agent = agent.Agent(env, sys.argv[2])
    else:
        print("usage:python brain.py [训练回合数] [(可选)已有的q表文件]")
    print("初始化完成：\n", "训练回合数=", episodes)
    sys.stdout.flush()
    # with tqdm(total=env.x_count*env.y_count*episodes) as pbar:
    for episode in range(episodes):
        for i in range(env.x_count):
            for j in range(env.y_count):
                episode_reward, round_count = Interact(env,
                                                       Agent,
                                                       i,
                                                       j,
                                                       train=True)
                # 记录本轮reward、步数、每步平均reward
                episode_rewards.append([
                    episode_reward, round_count, episode_reward / round_count
                ])
                # pbar.update(1)
        print("已完成", episode, "轮")
        sys.stdout.flush()
        if episode % 100 == 0:
            print("保存")
            sys.stdout.flush()
            if (len(sys.argv) == 2):
                np.save("result-0.npy", Agent.q)
                np.save("rewards-0.npy", np.array(episode_rewards))
            elif (len(sys.argv) == 3):
                np.save("result-1.npy", Agent.q)
                np.save("rewards-1.npy", np.array(episode_rewards))

    # 保存训练结果
    print("训练完成，开始存储结果")
    sys.stdout.flush()
    np.save("result.npy", Agent.q)

    # '''
    # #指定收敛条件训练
    # def stop(x,y,epi):
    #     if 2*abs(x-y)/abs(x+y)<epi:
    #         return True
    #     else:
    #         return False
    # for i in range(env.x_count):
    #     for j in range(env.y_count):
    #         while True:
    #             Len=len(episode_rewards)
    #             episode_reward = Interact(env, Agent, i,j,train=True)
    #             episode_rewards.append(episode_reward)
    #             if stop(episode_rewards[Len-1],episode_rewards[Len]):
    #                 break

    # '''
    # '''
    #     #随机设置起点，起点20km以内
    #     x_s=obs["position"][0]
    #     y_s=obs["position"][1]
    #     while True:
    #         x=np.random.randint(-10,10)+x_s
    #         y=np.random.randint(-10,10)+y_s
    #         if not(x<0 or y<0 or x>=self.x_count or y>=self.y_count):
    #             #越界处理
    #             break
    # '''
