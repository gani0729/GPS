#!/usr/bin/env python
# coding: utf-8
import copy

# In[ ]:


import gym
import pandas as pd
from gym import spaces
import numpy as np
from datetime import datetime, date, time, timedelta
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import time
from calendar import monthrange
from gym.utils import seeding # random seed control 위해 import

import os
from stable_baselines.common.vec_env import VecEnv, sync_envs_normalization, DummyVecEnv
from typing import Union, List, Optional, Tuple

from stable_baselines import DQN, PPO2
from stable_baselines.common import callbacks
from stable_baselines.common.callbacks import EvalCallback, BaseCallback

def evaluate_policy_Display(
        model,
        env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        return_episode_rewards: bool = False,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:

    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    epi_rewards_discounted, epi_success_order, epi_abandon_amount, epi_excess_order, epi_times = [], [], [], [], []
    for i in range(n_eval_episodes):
        if not isinstance(env, VecEnv) or i == 0:
            obs = env.reset()
        done, state = False, None
        epi_reward_discounted = 0.0
        epi_success = 0
        epi_abandon = 0
        epi_excess = 0
        epi_time = 0
        num_steps = 0
        while not done:
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, _info = env.step(action)
            epi_reward_discounted += np.power(model.gamma, num_steps) * reward
            num_steps += 1


            if _info[0].get('abandon_amount'):
                epi_abandon += _info[0].get('abandon_amount')
            if _info[0].get('success'):
                epi_success += 1
            if _info[0].get('excess'):
                epi_excess += 1

            epi_time = float(_info[0]['usetime'] / 24)

            if render:
                env.render()
        epi_rewards_discounted.append(epi_reward_discounted)
        epi_success_order.append(epi_success)
        epi_abandon_amount.append(epi_abandon)
        epi_times.append(epi_time)
    mean_discounted_reward = np.mean(epi_rewards_discounted)
    std_discounted_reward = np.std(epi_rewards_discounted)
    if return_episode_rewards:
        return epi_rewards_discounted, epi_success_order, epi_abandon_amount, epi_excess_order, epi_times
    return mean_discounted_reward, std_discounted_reward

class EvalCallback_Display(EvalCallback):
    def __init__(self, eval_env: Union[gym.Env, VecEnv],
                 callback_on_new_best: Optional[BaseCallback] = None,
                 n_eval_episodes: int = 5,
                 eval_freq: int = 10000,
                 log_path: str = None,
                 best_model_save_path: str = None,
                 deterministic: bool = True,
                 render: bool = False,
                 verbose: int = 1):
        super(EvalCallback_Display, self).__init__(eval_env,
                 callback_on_new_best,
                 n_eval_episodes,
                 eval_freq,
                 log_path,
                 best_model_save_path,
                 deterministic,
                 render,
                 verbose)
        self.results_discounted = []
        self.results_success_order = []
        self.results_abandon = []
        self.results_excess = []


    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            epi_rewards_discounted, epi_success_order, epi_abandon_amount, epi_excess_order, epi_times \
                = evaluate_policy_Display(self.model, self.eval_env,
                                            n_eval_episodes=self.n_eval_episodes,
                                            render=self.render,
                                            deterministic=self.deterministic,
                                            return_episode_rewards=True)


            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.results_discounted.append(epi_rewards_discounted)
                self.results_success_order.append(epi_success_order)
                self.results_abandon.append(epi_abandon_amount)
                self.results_excess.append(epi_excess_order)
                self.evaluations_length.append(epi_times)
                np.savez(self.log_path, timesteps=self.evaluations_timesteps,
                         results_discounted=self.results_discounted,
                         results_success_order=self.results_success_order,
                         results_abandon=self.results_abandon,
                         results_excess=self.results_excess,
                         ep_lengths=self.evaluations_length)

            mean_reward_discounted, std_reward_discounted = np.mean(epi_rewards_discounted), np.std(epi_rewards_discounted)
            mean_success, std_success = np.mean(epi_success_order), np.std(epi_success_order   )
            mean_abandon, std_abandon = np.mean(epi_abandon_amount), np.std(epi_abandon_amount)
            mean_excess, std_excess = np.mean(epi_excess_order), np.std(epi_excess_order)
            mean_ep_length, std_ep_length = np.mean(epi_times), np.std(epi_times)
            # Keep track of the last evaluation, useful for classes that derive from this callback
            self.last_mean_reward = mean_reward_discounted

            if self.verbose > 0:
                print("Eval num_timesteps={}, "
                      "episode_discounted_reward={:.2f} +/- {:.2f}".format(self.num_timesteps, mean_reward_discounted, std_reward_discounted),
                      "episode_success={:.2f} +/- {:.2f}".format(mean_success, std_success),
                      "episode_abandon={:.2f} +/- {:.2f}".format(mean_abandon, std_abandon),
                      "episode_excess={:.2f} +/- {:.2f}".format(mean_excess, std_excess),)
                print("Episode day: {:.2f} +/- {:.2f}".format(mean_ep_length, std_ep_length))

            # if mean_success < 1.0e-4 and self.n_calls % (self.eval_freq*5):
            #     self.model.setup_model()

            if mean_reward_discounted > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, 'best_model'))
                self.model.save(os.path.join(self.best_model_save_path, 'model' + str(self.num_timesteps)))
                self.best_mean_reward = mean_reward_discounted
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True

class Display(gym.Env):
    def __init__(self, order_data, cut_data, setup_time, device_max_1, device_max_2):


        ## 주문 정보 받아오기
        self.order = order_data
        ## 셋업에 소요되는 시간 받아오기
        self.setup_time = setup_time

        self.device_max_1 = device_max_1
        self.device_max_2 = device_max_2

        # self.processing_date = datetime(date.today().year, 4, 1, 0, 0, 0)

        ## 우리가 처리해야 할 주문의 갯수
        self.count = (self.order.drop('time', axis=1) != 0).sum().sum()

        ## 각 mol에 대한 생산시간
        self.p_m1_time = 40
        self.p_m2_time = 70
        self.p_m3_time = 50
        self.p_m4_time = 40


        ## 성형 공정에 1시간당 투입 가능한 MOL 수
        self.put_mol = 6



        ## mol 간의 변경 시간
        #같은 소자를 쓰는 mol끼리의 변경 시간
        self.change_mol = (6, 13)


        #성형 양품률
        self.molding_rate = 0.975
        #자르기 양품률
        self.cut_rate = 0.9

       ## observation_space 나타내기 위한 max 함수들
        self.p_m_time = max(self.p_m1_time, self.p_m2_time, self.p_m3_time, self.p_m4_time)
        self.device_max = max(device_max_1, device_max_2)

        ## 소자 종류 -> 0 : 소자1, 1 : 소자 2 존재
        self.device_type = (0, 1)


        self.max_data = (self.order.drop(['time'], axis=1)).max().max()
        # self.max_time = ((self.max_data / self.molding_rate) / 506) / self.cut_rate * 70

        self.action_space = spaces.Discrete(3) # SETUP1, SETUP2, STAY
        self.observation_space = spaces.MultiDiscrete([203, 4, self.device_max, 2, 3000])


        # 하나의 mol 성형이 끝났을 때
        self.reward_per_success = 1

        # 하나의 mol을 완전히 성형하진 못했지만, 투입은 해놓은 상태라면
        self.reward_per_success2 = 30

        # # 세팅한 소자를 모두 사용했을 때
        # self.reward_operation_max = 20.0
        #
        # # 낮은 가동률에 대한 패널티
        # self.reward_operation_rate1 = 10
        # self.reward_operation_rate2 = 5

        # 잘못된 셋업에 대한 패널티
        self.reward_per_miss = 300


        self.update_order = []

        for row in order_data.iterrows():

            month_index = 0
            if (row[1][0][5:7] == "04"):
                month_index = 0
            elif (row[1][0][5:7] == "05"):
                month_index = 1
            elif (row[1][0][5:7] == "06"):
                month_index = 2

            row[1][1] *= (100 / float(cut_data.loc[month_index, "BLK_1"])) / 506
            row[1][2] *= (100 / float(cut_data.loc[month_index, "BLK_2"])) / 506
            row[1][3] *= (100 / float(cut_data.loc[month_index, "BLK_3"])) / 400
            row[1][4] *= (100 / float(cut_data.loc[month_index, "BLK_4"])) / 400

            new_row = [row[1][0], row[1][1], row[1][2], row[1][3], row[1][4]]
            self.update_order.append(new_row)

        # In[84]:

        self.demand = pd.DataFrame(self.update_order, columns=["time", "MOL_1", "MOL_2", "MOL_3", "MOL_4"])
        self.demand['time'] = pd.to_datetime(self.demand['time'])
        self.demand['time'] = pd.to_datetime(self.demand['time'].dt.date) + pd.to_timedelta('18:00:00')
        # demand['time'] = demand['time'].apply(lambda x: datetime.combine(x.date(), time(18, 0, 0)))
        # demand.set_index("time", inplace=True)
        self.base_time = datetime(2020, 4, 1, 18)
        self.demand['time'] = self.demand['time'].apply(lambda x: int((x - self.base_time).total_seconds() / 3600))
        self.p_demand = self.demand.copy()


        self.viewer = None

        self.reset()


    def excess_penalty(self, due_time, c_time): #현재 성형해야 하는 mol의 납기가 지났을 때, 패널티
        penalty1 = 0
        coef1 = 0.0001
        #coef1 = 0.0001
        # processing_date = datetime(year, month, day)
        # due_date = datetime(year, due_month, due_day)

        difference = (c_time - due_time)
       # difference = (c_time - due_time) / 24
        # difference = delta.days
        penalty1 += coef1 * (difference ** 2)

        return penalty1

    def excess_penalty2(self, device, c_time, demand):
        coef3 = 0.5
        coef4 = 10
        penalty_mol1 = 0
        penalty_mol2 = 0
        penalty_mol3 = 0
        penalty_mol4 = 0
        mol_1_zero = 0
        mol_2_zero = 0
        mol_3_zero = 0
        mol_4_zero = 0

        if device == 0:
            if (self.demand['MOL_3'] != 0).any() == True:
                for i in range(len(demand)):
                    if demand.loc[i, 'MOL_3'] != 0:
                        mol_3_zero = i
                        break
                due_time3 = demand.iloc[mol_3_zero, 0]
                if int(due_time3) - int(c_time) >= 0:
                    penalty_mol3 = coef4*(1/(due_time3 - c_time + 1))
                elif int(due_time3) - int(c_time) < 0:
                    penalty_mol3 = coef3 * (abs(c_time - due_time3))**2

            if  (self.demand['MOL_4'] != 0).any() == True:
                for j in range(len(demand)):
                    if demand.loc[j, 'MOL_4'] != 0:
                        mol_4_zero = j
                        break
                due_time4 = demand.iloc[mol_4_zero, 0]
                if int(due_time4) - int(c_time) >= 0:
                    penalty_mol4 = coef4 * (1/(due_time4 - c_time + 1))
                elif int(due_time4) - int(c_time) < 0:
                    penalty_mol4 = coef3 * (abs(c_time - due_time4))**2

            penalty3 = penalty_mol3 + penalty_mol4

            return penalty3

        elif device == 1:
            if (self.demand['MOL_1'] != 0).any() == True:
                for k in range(len(demand)):
                    if demand.loc[k, 'MOL_1'] != 0:
                        mol_1_zero = k
                        break
                due_time1 = demand.iloc[mol_1_zero, 0]
                if int(due_time1) - int(c_time) >= 0:
                    penalty_mol1 = coef4*(1/(due_time1 - c_time + 1))
                elif int(due_time1) - int(c_time) < 0:
                    penalty_mol1 = coef3 * (abs(c_time - due_time1))**2

            if  (self.demand['MOL_2'] != 0).any() == True:
                for l in range(len(demand)):
                    if demand.loc[l, 'MOL_2'] != 0:
                        mol_2_zero = l
                        break
                due_time2 = demand.iloc[mol_2_zero, 0]
                if int(due_time2) - int(c_time) >= 0:
                    penalty_mol2 = coef4 * (1/(due_time2 - c_time + 1))
                elif int(due_time2) - int(c_time) < 0:
                    penalty_mol2 = coef3 * (abs(c_time - due_time2))**2

            penalty3 = penalty_mol1 + penalty_mol2

            return penalty3

    def abandon_penalty(self, p_amount_set_up): #세팅한 소자를 다 쓰지 않고 버렸을 때 패널티
        penalty2 = 0
        coef2 = 0.5
        y = p_amount_set_up
        penalty2 += coef2 * (y**2)

        return penalty2

    def calculate_utilization(self, running_times, setup_times, model_change_times): #낮은 가동률에 대한 패널티
        total_running_time = sum(running_times)
        total_setup_time = sum(setup_times)
        total_model_change_time = sum(model_change_times)
        total_times = total_model_change_time + total_setup_time + total_running_time
        if int(total_times) == 0:
            return 0
        else:
            utilization = float(total_running_time) / float(total_times)
            return utilization


    def load_model(self, p_device, device, p_demand, demand, current_time, p_mol_name):
        if p_device != device:  # 이전 소자와 다른 소자 셋업
            if device == 0:  # 소자 0사용
                for i in range(len(demand)):
                    if demand.loc[i, 'MOL_1'] != 0:
                        first_nonzero_1 = i
                        break
                for j in range(len(demand)):
                    if demand.loc[j, 'MOL_2'] != 0:
                        first_nonzero_2 = j
                        break
                # MOL_1과 MOL_1의 주문량이 모두 존재할 때
                if (demand['MOL_1'] != 0).any() == True and (demand['MOL_2'] != 0).any() == True:

                    # else: #amount=None이라면 즉, 모든 열의 값이 0이라면 => 두 모델 모두 필요한 주문양 생산완료
                    #    mol_name = 'MOL_1'
                    #    time = 0
                    #    amount = 0
                    #    info = pd.DataFrame({'time': [time],
                    #                         'mol_name': [mol_name],
                    #                         'amount': [amount]})
                    if first_nonzero_1 < first_nonzero_2:  # 납기일 이른 거 먼저 선택
                        amount = demand.iloc[first_nonzero_1, 1]
                        mol_name = 'MOL_1'
                        time = demand.loc[demand[(demand[mol_name] == amount)].index[0], 'time']
                        info = pd.DataFrame({'time': [time],
                                             'mol_name': [mol_name],
                                             'amount': [amount]})
                    elif first_nonzero_1 == first_nonzero_2:  # 납기일 동일 시
                        if demand.iloc[first_nonzero_1, 1] < demand.iloc[first_nonzero_2, 2]:  # 더 작은 값을 가지는 열 선택
                            amount = demand.iloc[first_nonzero_1, 1]
                            mol_name = 'MOL_1'
                            time = demand.loc[demand[(demand[mol_name] == amount)].index[0], 'time']
                            info = pd.DataFrame({'time': [time],
                                                 'mol_name': [mol_name],
                                                 'amount': [amount]})
                        else:
                            amount = demand.iloc[first_nonzero_2, 2]
                            mol_name = 'MOL_2'
                            time = demand.loc[demand[(demand[mol_name] == amount)].index[0], 'time']
                            info = pd.DataFrame({'time': [time],
                                                 'mol_name': [mol_name],
                                                 'amount': [amount]})
                    else:
                        amount = demand.iloc[first_nonzero_2, 2]
                        mol_name = demand.columns[(demand == amount).any()][0]
                        time = demand.loc[demand[(demand[mol_name] == amount)].index[0], 'time']
                        info = pd.DataFrame({'time': [time],
                                             'mol_name': [mol_name],
                                             'amount': [amount]})
                # MOL_1의 주문량만 존재할 때
                elif (demand['MOL_1'] != 0).any() == True and (demand['MOL_2'] == 0).all() == True:
                    for i in range(len(demand)):
                        if demand.loc[i, 'MOL_1'] != 0:
                            first_nonzero_1 = i
                            break
                    amount = demand.iloc[first_nonzero_1, 1]  # 동일 모델 연속 생산
                    mol_name = 'MOL_1'
                    time = demand.loc[first_nonzero_1, 'time']
                    info = pd.DataFrame({'time': [time],
                                         'mol_name': [mol_name],
                                         'amount': [amount]})

                # MOL_2에만 주문량이 존재하는 경우
                elif (demand['MOL_1'] == 0).all() == True and (demand['MOL_2'] != 0).any() == True:
                    for j in range(len(demand)):
                        if demand.loc[j, 'MOL_2'] != 0:
                            first_nonzero_2 = j
                            break
                    amount = demand.iloc[first_nonzero_2, 2]  # 동일 모델 연속 생산
                    mol_name = 'MOL_2'
                    time = demand.loc[first_nonzero_2, 'time']
                    info = pd.DataFrame({'time': [time],
                                         'mol_name': [mol_name],
                                         'amount': [amount]})

                # MOL_1과 MOL_2 모두 주문량이 존재하지 않을 때
                elif (demand['MOL_1'] == 0).all() == True and (demand['MOL_2'] == 0).all() == True:
                    time = 0
                    amount = 0
                    mol_name = 'MOL_1'
                    info = pd.DataFrame({'time': [time],
                                         'mol_name': [mol_name],
                                         'amount': [amount]})

            else:  # 소자 1 사용
                for i in range(len(demand)):
                    if demand.loc[i, 'MOL_3'] != 0:
                        first_nonzero_3 = i
                        break
                for j in range(len(demand)):
                    if demand.loc[j, 'MOL_4'] != 0:
                        first_nonzero_4 = j
                        break

                # MOL_3와 MOL_4 모두 주문량이 존재할 때
                if (demand['MOL_3'] != 0).any() == True and (demand['MOL_4'] != 0).any() == True:
                    if first_nonzero_3 < first_nonzero_4:  # 납기일 이른 거 먼저 선택
                        amount = demand.iloc[first_nonzero_3, 3]
                        mol_name = 'MOL_3'
                        time = demand.loc[demand[(demand[mol_name] == amount)].index[0], 'time']
                        info = pd.DataFrame({'time': [time],
                                             'mol_name': [mol_name],
                                             'amount': [amount]})
                    elif first_nonzero_3 == first_nonzero_4:  # 납기일 동일 시
                        if demand.iloc[first_nonzero_3, 3] < demand.iloc[first_nonzero_4, 4]:  # 더 작은 값을 가지는 열 선택
                            amount = demand.iloc[first_nonzero_3, 3]
                            mol_name = 'MOL_3'
                            time = demand.loc[demand[(demand[mol_name] == amount)].index[0], 'time']
                            info = pd.DataFrame({'time': [time],
                                                 'mol_name': [mol_name],
                                                 'amount': [amount]})
                        else:
                            amount = demand.iloc[first_nonzero_4, 4]
                            mol_name = 'MOL_4'
                            time = demand.loc[demand[(demand[mol_name] == amount)].index[0], 'time']
                            info = pd.DataFrame({'time': [time],
                                                 'mol_name': [mol_name],
                                                 'amount': [amount]})
                    else:
                        amount = demand.iloc[first_nonzero_4, 4]
                        mol_name = 'MOL_4'
                        time = demand.loc[demand[(demand[mol_name] == amount)].index[0], 'time']
                        info = pd.DataFrame({'time': [time],
                                             'mol_name': [mol_name],
                                             'amount': [amount]})
                    # else:
                    #    mol_name = 'MOL_3'
                    #    time = 0
                    #    amount = 0
                    #    info = pd.DataFrame({'time': [time],
                    #                         'mol_name': [mol_name],
                    #                         'amount': [amount]})

                # MOL_3에만 주문량이 존재하는 경우
                elif (demand['MOL_3'] != 0).any() == True and (demand['MOL_4'] == 0).all() == True:
                    for i in range(len(demand)):
                        if demand.loc[i, 'MOL_3'] != 0:
                            first_nonzero_3 = i
                            break
                    amount = demand.iloc[first_nonzero_3, 3]  # 동일 모델 연속 생산
                    mol_name = 'MOL_3'
                    time = demand.loc[first_nonzero_3, 'time']
                    info = pd.DataFrame({'time': [time],
                                         'mol_name': [mol_name],
                                         'amount': [amount]})

                # MOL_4에만 주문량이 존재하는 경우
                elif (demand['MOL_3'] == 0).all() == True and (demand['MOL_4'] != 0).any() == True:
                    for j in range(len(demand)):
                        if demand.loc[j, 'MOL_4'] != 0:
                            first_nonzero_4 = j
                            break
                    amount = demand.iloc[first_nonzero_4, 4]  # 동일 모델 연속 생산
                    mol_name = 'MOL_4'
                    time = demand.loc[first_nonzero_4, 'time']
                    info = pd.DataFrame({'time': [time],
                                         'mol_name': [mol_name],
                                         'amount': [amount]})

                # MOL_3과 MOL_4 모두 주문량이 존재하지 않을 때
                elif (demand['MOL_3'] == 0).all() == True and (demand['MOL_4'] == 0).all() == True:
                    time = 0
                    amount = 0
                    mol_name = 'MOL_3'
                    info = pd.DataFrame({'time': [time],
                                         'mol_name': [mol_name],
                                         'amount': [amount]})

        # p_demand와 demand를 만들어서 첫번째로 0이 아닌 값의 행을 비교 행이 같으면 아직 덜 만들었다는 뜻 그대로 생산 행이 다르면 두 행 사이에 0의 개수가 3개 이하라면 ~ 판단, 3개 초과라면 ~ ㅏ른 모델 생산
        else:  # 같은 종류의 소자 셋업
            if device == 0:  # 소자 0 셋업

                # MOL_1과 MOL_2 모두 주문 존재하는 경우
                if (demand['MOL_1'] != 0).any() == True and (demand['MOL_2'] != 0).any() == True:

                    # demand와 p_demand에서 MOL_1과 MOL_2의 0이 아닌 첫번째 행 찾기
                    for i in range(len(demand)):
                        if demand.loc[i, 'MOL_1'] != 0:
                            first_nonzero_1 = i
                            break
                    for j in range(len(demand)):
                        if demand.loc[j, 'MOL_2'] != 0:
                            first_nonzero_2 = j
                            break
                    for k in range(len(p_demand)):
                        if p_demand.loc[k, 'MOL_1'] != 0:
                            first_nonzero_1_p = k
                            break
                    for l in range(len(p_demand)):
                        if p_demand.loc[l, 'MOL_2'] != 0:
                            first_nonzero_2_p = l
                            break

                    if p_mol_name == 1:  # 이전 모델이 MOL_1이었을 때
                        if (demand.loc[
                                first_nonzero_2, 'time'] - current_time) <= 225:  # 현재시간과 MOL_2의 납기일의 차이가 225시간 이하라면 (납기일을 이미 지난 경우도 포함)
                            amount = demand.iloc[first_nonzero_2, 2]  # MOL_2 선택
                            mol_name = 'MOL_2'
                            time = demand.loc[first_nonzero_2, 'time']
                            info = pd.DataFrame({'time': [time],
                                                 'mol_name': [mol_name],
                                                 'amount': [amount]})

                        else:  # MOL_2의 납기일이 여유가 있다면
                            num_zeros_1 = (demand.iloc[first_nonzero_1_p + 1:first_nonzero_1 + 1,
                                           1] == 0).sum()  # MOL_1의 이전 생산 행과 첫 번째 0이 아닌 값을 가지는 행 사이의 0의 개수 세기
                            # 연속되는 0의 개수가 3 이하라면 (demand와 p_demand의 0이 아닌 행이 같을 경우(이전 모델의 생산량을 모두 충족시키지 못하고 남기고 끝난 경우)도 자동 포함)
                            if num_zeros_1 <= 3:
                                amount = demand.iloc[first_nonzero_1, 1]  # 동일 모델 연속 생산
                                mol_name = 'MOL_1'
                                time = demand.loc[first_nonzero_1, 'time']
                                info = pd.DataFrame({'time': [time],
                                                     'mol_name': [mol_name],
                                                     'amount': [amount]})

                            else:  # 연속되는 0의 개수가 3 초과라면
                                amount = demand.iloc[first_nonzero_2, 2]  # 동일 소자 다른 모델 선택
                                mol_name = 'MOL_2'
                                time = demand.loc[first_nonzero_2, 'time']
                                info = pd.DataFrame({'time': [time],
                                                     'mol_name': [mol_name],
                                                     'amount': [amount]})

                    else:  # 이전 모델이 MOL_2였을 때
                        if (demand.loc[
                                first_nonzero_1, 'time'] - current_time) <= 225:  # 현재 시간과 MOL_1의 납기일의 차이가 225시간 이하라면 (납기일을 이미 지난 경우도 포함)
                            amount = demand.iloc[first_nonzero_1, 1]  # MOL_1 선택
                            mol_name = 'MOL_1'
                            time = demand.loc[first_nonzero_1, 'time']
                            info = pd.DataFrame({'time': [time],
                                                 'mol_name': [mol_name],
                                                 'amount': [amount]})

                        else:  # MOL_1의 납기일이 여유가 있다면
                            num_zeros_2 = (demand.iloc[first_nonzero_2_p + 1:first_nonzero_2 + 1,
                                           2] == 0).sum()  # MOL_2의 이전 생산 행과 첫 번째 0이 아닌 값을 가지는 행 사이의 0의 개수 세기
                            # 연속되는 0의 개수가 3 이하라면 (demand와 p_demand의 0이 아닌 행이 같을 경우(이전 모델의 생산량을 모두 충족시키지 못하고 남기고 끝난 경우)도 자동 포함)
                            if num_zeros_2 <= 3:
                                amount = demand.iloc[first_nonzero_2, 2]  # 동일 모델 연속 생산
                                mol_name = 'MOL_2'
                                time = demand.loc[first_nonzero_2, 'time']
                                info = pd.DataFrame({'time': [time],
                                                     'mol_name': [mol_name],
                                                     'amount': [amount]})
                            else:  # 연속되는 0의 개수가 3 초과라면
                                amount = demand.iloc[first_nonzero_1, 1]  # 동일 소자 다른 모델 선택
                                mol_name = 'MOL_1'
                                time = demand.loc[first_nonzero_1, 'time']
                                info = pd.DataFrame({'time': [time],
                                                     'mol_name': [mol_name],
                                                     'amount': [amount]})

                # MOL_1에만 주문량이 존재하는 경우
                elif (demand['MOL_1'] != 0).any() == True and (demand['MOL_2'] == 0).all() == True:
                    for i in range(len(demand)):
                        if demand.loc[i, 'MOL_1'] != 0:
                            first_nonzero_1 = i
                            break
                    amount = demand.iloc[first_nonzero_1, 1]  # 동일 모델 연속 생산
                    mol_name = 'MOL_1'
                    time = demand.loc[first_nonzero_1, 'time']
                    info = pd.DataFrame({'time': [time],
                                         'mol_name': [mol_name],
                                         'amount': [amount]})

                # MOL_2에만 주문량이 존재하는 경우
                elif (demand['MOL_1'] == 0).all() == True and (demand['MOL_2'] != 0).any() == True:
                    for j in range(len(demand)):
                        if demand.loc[j, 'MOL_2'] != 0:
                            first_nonzero_2 = j
                            break
                    amount = demand.iloc[first_nonzero_2, 2]  # 동일 모델 연속 생산
                    mol_name = 'MOL_2'
                    time = demand.loc[first_nonzero_2, 'time']
                    info = pd.DataFrame({'time': [time],
                                         'mol_name': [mol_name],
                                         'amount': [amount]})

                # MOL_1과 MOL_2 모두 주문량이 존재하지 않을 때
                elif (demand['MOL_1'] == 0).all() == True and (demand['MOL_2'] == 0).all() == True:
                    time = 0
                    amount = 0
                    mol_name = 'MOL_1'
                    info = pd.DataFrame({'time': [time],
                                         'mol_name': [mol_name],
                                         'amount': [amount]})
            # 소자 1 셋업
            else:
                # MOL_3과 MOL_4 모두 주문 존재하는 경우
                if (demand['MOL_3'] != 0).any() == True and (demand['MOL_4'] != 0).any() == True:

                    # demand와 p_demand에서 MOL_1과 MOL_2의 0이 아닌 첫번째 행 찾기
                    for i in range(len(demand)):
                        if demand.loc[i, 'MOL_3'] != 0:
                            first_nonzero_3 = i
                            break
                    for j in range(len(demand)):
                        if demand.loc[j, 'MOL_4'] != 0:
                            first_nonzero_4 = j
                            break
                    for k in range(len(p_demand)):
                        if p_demand.loc[k, 'MOL_3'] != 0:
                            first_nonzero_3_p = k
                            break
                    for l in range(len(p_demand)):
                        if p_demand.loc[l, 'MOL_4'] != 0:
                            first_nonzero_4_p = l
                            break

                    if p_mol_name == 3:  # 이전 모델이 MOL_3였을 때
                        if (demand.loc[
                                first_nonzero_4, 'time'] - current_time) <= 225:  # 현재시간과 MOL_4의 납기일의 차이가 225시간 이하라면 (납기일을 이미 지난 경우도 포함)
                            amount = demand.iloc[first_nonzero_4, 4]  # MOL_2 선택
                            mol_name = 'MOL_4'
                            time = demand.loc[first_nonzero_4, 'time']
                            info = pd.DataFrame({'time': [time],
                                                 'mol_name': [mol_name],
                                                 'amount': [amount]})

                        else:  # MOL_4의 납기일이 여유가 있다면
                            num_zeros_3 = (demand.iloc[first_nonzero_3_p + 1:first_nonzero_3 + 1,
                                           3] == 0).sum()  # MOL_3의 이전 생산 행과 첫 번째 0이 아닌 값을 가지는 행 사이의 0의 개수 세기
                            # 연속되는 0의 개수가 3 이하라면 (demand와 p_demand의 0이 아닌 행이 같을 경우(이전 모델의 생산량을 모두 충족시키지 못하고 남기고 끝난 경우)도 자동 포함)
                            if num_zeros_3 <= 3:
                                amount = demand.iloc[first_nonzero_3, 3]  # 동일 모델 연속 생산
                                mol_name = 'MOL_3'
                                time = demand.loc[first_nonzero_3, 'time']
                                info = pd.DataFrame({'time': [time],
                                                     'mol_name': [mol_name],
                                                     'amount': [amount]})

                            else:  # 연속되는 0의 개수가 3 초과라면
                                amount = demand.iloc[first_nonzero_4, 4]  # 동일 소자 다른 모델 선택
                                mol_name = 'MOL_4'
                                time = demand.loc[first_nonzero_4, 'time']
                                info = pd.DataFrame({'time': [time],
                                                     'mol_name': [mol_name],
                                                     'amount': [amount]})

                    else:  # 이전 모델이 MOL_4였을 때
                        if (demand.loc[
                                first_nonzero_3, 'time'] - current_time) <= 225:  # 현재 시간과 MOL_3의 납기일의 차이가 225시간 이하라면 (납기일을 이미 지난 경우도 포함)
                            amount = demand.iloc[first_nonzero_3, 3]  # MOL_3 선택
                            mol_name = 'MOL_3'
                            time = demand.loc[first_nonzero_3, 'time']
                            info = pd.DataFrame({'time': [time],
                                                 'mol_name': [mol_name],
                                                 'amount': [amount]})

                        else:  # MOL_3의 납기일이 여유가 있다면
                            num_zeros_4 = (demand.iloc[first_nonzero_4_p + 1:first_nonzero_4 + 1,
                                           4] == 0).sum()  # MOL_4의 이전 생산 행과 첫 번째 0이 아닌 값을 가지는 행 사이의 0의 개수 세기
                            # 연속되는 0의 개수가 3 이하라면 (demand와 p_demand의 0이 아닌 행이 같을 경우(이전 모델의 생산량을 모두 충족시키지 못하고 남기고 끝난 경우)도 자동 포함)
                            if num_zeros_4 <= 3:
                                amount = demand.iloc[first_nonzero_4, 4]  # 동일 모델 연속 생산
                                mol_name = 'MOL_4'
                                time = demand.loc[first_nonzero_4, 'time']
                                info = pd.DataFrame({'time': [time],
                                                     'mol_name': [mol_name],
                                                     'amount': [amount]})
                            else:  # 연속되는 0의 개수가 3 초과라면
                                amount = demand.iloc[first_nonzero_3, 3]  # 동일 소자 다른 모델 선택
                                mol_name = 'MOL_3'
                                time = demand.loc[first_nonzero_3, 'time']
                                info = pd.DataFrame({'time': [time],
                                                     'mol_name': [mol_name],
                                                     'amount': [amount]})

                # MOL_3에만 주문량이 존재하는 경우
                elif (demand['MOL_3'] != 0).any() == True and (demand['MOL_4'] == 0).all() == True:
                    for i in range(len(demand)):
                        if demand.loc[i, 'MOL_3'] != 0:
                            first_nonzero_3 = i
                            break
                    amount = demand.iloc[first_nonzero_3, 3]  # 동일 모델 연속 생산
                    mol_name = 'MOL_3'
                    time = demand.loc[first_nonzero_3, 'time']
                    info = pd.DataFrame({'time': [time],
                                         'mol_name': [mol_name],
                                         'amount': [amount]})

                # MOL_4에만 주문량이 존재하는 경우
                elif (demand['MOL_3'] == 0).all() == True and (demand['MOL_4'] != 0).any() == True:
                    for j in range(len(demand)):
                        if demand.loc[j, 'MOL_4'] != 0:
                            first_nonzero_4 = j
                            break
                    amount = demand.iloc[first_nonzero_4, 4]  # 동일 모델 연속 생산
                    mol_name = 'MOL_4'
                    time = demand.loc[first_nonzero_4, 'time']
                    info = pd.DataFrame({'time': [time],
                                         'mol_name': [mol_name],
                                         'amount': [amount]})

                # MOL_3과 MOL_4 모두 주문량이 존재하지 않을 때
                elif (demand['MOL_3'] == 0).all() == True and (demand['MOL_4'] == 0).all() == True:
                    time = 0
                    amount = 0
                    mol_name = 'MOL_3'
                    info = pd.DataFrame({'time': [time],
                                         'mol_name': [mol_name],
                                         'amount': [amount]})

        return info

        # def update_model(self, demand, result, column_name, index_name):
    #     demand.loc[index_name, column_name] -= result
    #     return demand

    def update_model(self, demand, order_info, required_mol):
        demand.loc[demand['time'] == order_info.loc[0, 'time'],order_info['mol_name']] = required_mol
        return demand



    def step(self, action):
        # 현재 생산해야 하는 mol의 성형에 소요되는 시간 / 현재 셋업된 소자 잔량 / 현재 세팅된 소자 종류 / 현재 날짜
        required_mol, mol_name, amount_set_up, device, c_time = self.state

        # 셋업하기 이전의 소자 잔량 저장
        p_amount_set_up = amount_set_up
        p_device = device
        p_mol_name = mol_name
        #p_amount_set_up = int(p_amount_set_up)
        #amount_set_up = int(amount_set_up)

        reward = 0




        done = False
        self.steps += 1
        info = {}



        #set-up action
        if action == 0: #소자1을 셋업
            amount_set_up = device_max_1
            device = 0
            c_time += self.setup_time
            #info['abandon_amount'] = p_amount_set_up
            self.setup_times.append(self.setup_time)
            info['set-up'] = True
        elif action == 1: #소자2를 셋업
            amount_set_up = device_max_2
            device = 1
            c_time += self.setup_time
            #info['abandon_amount'] = p_amount_set_up
            self.setup_times.append(self.setup_time)
            info['set-up'] = True
        elif action == 2: #아무런 셋업하지 않음 -> stay
            amount_set_up += 0
            device = p_device
        else:
            raise Exception('bad action {}'.format(action))


        order_info = self.load_model(p_device, device, self.p_demand, self.demand, c_time, p_mol_name)
        self.p_demand = self.demand
        required_mol = order_info['amount'].values[0]
        if (order_info['mol_name'] == 'MOL_1').any():
            mol_name = 1
        elif (order_info['mol_name'] == 'MOL_2').any():
            mol_name = 2
        elif (order_info['mol_name'] == 'MOL_3').any():
            mol_name = 3
        elif (order_info['mol_name'] == 'MOL_4').any():
            mol_name = 4


        #현재 셋업된 소자량으로 생산가능한 mol
        possible_mol = amount_set_up * 6


        # 모델 체인지 시간 반영. 만약 STAY라면 setup과 별개로 모델 체인지 발생
        if action == 2:
            if p_mol_name == mol_name:
                c_time += 0
                self.model_change_times.append(0)
            elif p_mol_name != mol_name:
                if p_device == device:
                    c_time += self.change_mol[0]
                    self.model_change_times.append(self.change_mol[0])
                else:
                    c_time += self.change_mol[1]
                    self.model_change_times.append(self.change_mol[1])

        # 만약 새로운 셋업 액션을 했더라면,모델 체인지 시간은 setup 내에서 이루어지기 때문에 미포함..
        else:
            self.model_change_times.append(0)
            c_time += 0


        if p_amount_set_up > 0 and action != 2:
            reward -= self.abandon_penalty(p_amount_set_up)
            info['abandon_amount'] = p_amount_set_up
        elif p_amount_set_up == 0 and action != 2:
            reward += self.reward_per_miss

            # if p_device == 0:
            #     if p_amount_set_up > 0.2 * device_max_1:
            #         reward = self.abandon_penalty2(p_amount_set_up)
            #         info['abandon_amount'] = p_amount_set_up
            #     else:
            #         reward -= self.abandon_penalty(p_amount_set_up)
            #         info['abandon_amount'] = p_amount_set_up
            # else:
            #     if p_amount_set_up > 0.2 * device_max_2:
            #         reward = self.abandon_penalty2(p_amount_set_up)
            #         info['abandon_amount'] = p_amount_set_up
            #     else:
            #         reward -= self.abandon_penalty(p_amount_set_up)
            #         info['abandon_amount'] = p_amount_set_up





            # reward -= self.abandon_penalty(p_amount_set_up)
            # info['abandon_amount'] = p_amount_set_up

        elif p_amount_set_up == 0 and required_mol != 0 and action == 2:
           reward -= self.reward_per_miss


        # 주문양이 남아있는 모델의 소자 선택하도록 유도
        if (self.demand['MOL_1'] == 0).all() == True and (self.demand['MOL_2'] == 0).all() == True:
            if (self.demand['MOL_3'] != 0).any() == True or (self.demand['MOL_4'] != 0).any() == True:
                if device == 0:
                    reward -= self.reward_per_miss
                elif action == 2:
                    reward -= self.reward_per_miss
        elif (self.demand['MOL_3'] == 0).all() == True and (self.demand['MOL_4'] == 0).all() == True:
            if (self.demand['MOL_1'] != 0).any() == True or (self.demand['MOL_2'] != 0).any() == True:
                if device == 1:
                    reward -= self.reward_per_miss
                elif action == 2:
                    reward -= self.reward_per_miss

        #소자 태우는 상태 전이 표현
        if required_mol < possible_mol and required_mol > 0:
            self.running_times.append(required_mol / 6)
            reward += (self.reward_per_success * required_mol)
            if (required_mol % 6).any() == 0:
                c_time += required_mol // 6
            else:
                c_time += (required_mol // 6) + 1
            possible_mol -= required_mol
            required_mol = 0
            info['success'] = True
            amount_set_up = possible_mol / 6
            #reward += self.reward_per_success


        elif required_mol == possible_mol and required_mol > 0:
            self.running_times.append(amount_set_up)
            reward += (self.reward_per_success * required_mol)
            if (required_mol % 6).any() == 0:
                c_time += required_mol // 6
            else:
                c_time += (required_mol // 6) + 1
            required_mol = 0
            info['success'] = True
            possible_mol = 0
            amount_set_up = possible_mol / 6
            # reward += self.reward_operation_max
            #reward += self.reward_per_success



        elif required_mol > possible_mol and required_mol > 0 and possible_mol > 0:
            self.running_times.append(amount_set_up)
            reward += (self.reward_per_success * possible_mol)
            if possible_mol % 6 == 0:
                c_time += possible_mol // 6
            else:
                c_time += (possible_mol // 6) + 1
            required_mol -= possible_mol
            possible_mol = 0
            amount_set_up = possible_mol / 6
            # reward += self.reward_operation_max
            #reward += self.reward_per_success2



        due_time = order_info['time']
        due_time_list = []
        required_mol_list = []
        mol_name_list = []

        if int(due_time) != 0:
            if (due_time < c_time).any():
                if due_time not in due_time_list:
                    due_time_list.append(due_time)
                    required_mol_list.append(order_info['amount'].values[0])
                    mol_name_list.append(order_info['mol_name'])
                    reward -= self.excess_penalty(due_time, c_time)
                    info['excess'] = True
                else:
                    if order_info['amount'].values[0] in required_mol_list and order_info['mol_name'] in mol_name_list:
                        if due_time_list.index(due_time) == required_mol_list.index(order_info['amount'].values[0]) == mol_name_list.index(order_info['mol_name']):
                            required_mol_list.append(order_info['amount'].values[0])
                        else:
                            due_time_list.append(due_time)
                            required_mol_list.append(order_info['amount'].values[0])
                            mol_name_list.append(order_info['mol_name'])
                            reward -= self.excess_penalty(due_time, c_time)
                            info['excess'] = True
                    elif order_info['amount'].values[0] in required_mol_list and order_info['mol_name'] not in mol_name_list:
                        due_time_list.append(due_time)
                        required_mol_list.append(order_info['amount'].values[0])
                        mol_name_list.append(order_info['mol_name'])
                        reward -= self.excess_penalty(due_time, c_time)
                        info['excess'] = True
                    elif order_info['amount'].values[0] not in required_mol_list and order_info['mol_name'] in mol_name_list:
                        due_time_list.append(due_time)
                        required_mol_list.append(order_info['amount'].values[0])
                        mol_name_list.append(order_info['mol_name'])
                        reward -= self.excess_penalty(due_time, c_time)
                        info['excess'] = True


                    #reward -= self.excess_penalty(due_time, c_time)

        info['usetime'] = c_time
        # required_mol = order_info['result']
        self.demand = self.update_model(self.demand, order_info, required_mol)

        if (self.demand.astype(int)[['MOL_1', 'MOL_2', 'MOL_3', 'MOL_4']] == 0).all().all():
            done = True
            self.reset()
        if self.steps == 1000:
            done = True
            self.reset()


        self.state = (required_mol, mol_name, amount_set_up, device, c_time)
        return np.array(self.state, dtype=object), reward, done, info

    def reset(self):
        self.state = (0, 2, 0, 0,) + (0,)

        self.steps = 0
        self.running_times = []
        self.setup_times = []
        self.model_change_times = []

        self.demand = pd.DataFrame(self.update_order, columns=["time", "MOL_1", "MOL_2", "MOL_3", "MOL_4"])
        self.demand['time'] = pd.to_datetime(self.demand['time'])
        self.demand['time'] = pd.to_datetime(self.demand['time'].dt.date) + pd.to_timedelta('18:00:00')
        self.base_time = datetime(2020, 4, 1, 18)
        self.demand['time'] = self.demand['time'].apply(lambda x: int((x - self.base_time).total_seconds() / 3600))
        self.p_demand = self.demand.copy()

        return np.array(self.state)

    def render(self):
        pass

    def close(self):
        pass
#
# order_data = pd.read_csv('C:/SDOLab-server/황인근/디스플레이 강화학습/order.csv')
# cut_data = pd.read_csv('C:/SDOLab-server/황인근/디스플레이 강화학습/cut_yield.csv')
#
# order_data_ori = order_data.copy()
# cut_data_ori = cut_data.copy()

if __name__ == '__main__':
    action_name = {0: "Device 1 set-up", 1: "Device 2 set-up", 2: "Not set-up"}

    order_data_orig = pd.read_csv('C:/SDOLab-server/황인근/디스플레이 강화학습/order.csv')
    order_data = copy.deepcopy(order_data_orig)
    cut_data = pd.read_csv('C:/SDOLab-server/황인근/디스플레이 강화학습/cut_yield.csv')
    setup_time = 28
    device_max_1 = 150
    device_max_2 = 130
    env = Display(order_data=order_data, cut_data=cut_data, setup_time=setup_time, device_max_1=device_max_1, device_max_2=device_max_2)
    # eval_env = Displapy(order_data=order_data, cut_data=cut_data, setup_time=setup_time, device_max_1=device_max_1, device_max_2=device_max_2)
    #
    # cb = EvalCallback_Display(eval_env=eval_env, n_eval_episodes=10, eval_freq=1000,
    #                           log_path="./model",
    #                           best_model_save_path="./best_model"
    #                           )
    model = PPO2('MlpPolicy', env, verbose=1)
    total_timesteps = int(1.0e4)
    model.learn(total_timesteps=total_timesteps)

    # plt.plot(range(100, total_timesteps + 1, 100), callback.mean_rewards)
    # plt.xlabel('Number of steps')
    # plt.ylabel('Mean reward')
    # plt.title('Learning Curve')
    # plt.show()

    # # TEST
    # count_abandon = 0
    # count_success = 0
    # count_excess = 0
    # cumul_reward = 0
    # env.reset()
    # for iter in range(1000):
    #     env.render()
    #     # print("state: ", "{:.1f}".format(np.array(env.state, dtype=float)))
    #     print("state: ", [round(x, 1) for x in np.array(env.state, dtype=float)])
    #     action = env.action_space.sample()
    #     observation, reward, done, info = env.step(action)
    #     cumul_reward += reward
    #     if info.get('abandon_amount'):
    #         count_abandon += 1
    #     if info.get('success'):
    #         count_success += 1
    #     if info.get('excess'):
    #         count_excess += 1
    #     print("action: ", action_name[action])
    #     print("reward this step: ", "{:.1f}".format(float(reward)))
    #     print("total reward: ", "{:.1f}".format(float(cumul_reward)))
    #     print("=" * 50)
    #     # time.sleep(1)
    #     if done:
    #         break
    # print("Total successful move: ", count_success)
    # print("Total abandon a mount", count_abandon)
    # print("Total excess count", count_excess)

    # order_data = pd.read_csv('C:/SDOLab-server/황인근/디스플레이 강화학습/order.csv')
    # env = Displapy(order_data=order_data, cut_data=cut_data, setup_time=setup_time, device_max_1=device_max_1,
    #                device_max_2=device_max_2)
    # #
    # Enjoy trained agent
    obs = env.reset()
    count_abandon = 0
    count_success = 0
    count_excess = 0
    count_set_up = 0
    count_failed = 0
    cumul_reward = 0
    production_ready = pd.DataFrame(columns=['require_mol', 'mol_name', 'amount_setup', 'device', 'time', 'Action']) # 학습 결과를 출력하기 위해 틀 준비
    for i in range(1000):
        # print("state: ", np.array(env.state))
        print("state: ", [round(x, 1) for x in np.array(env.state, dtype=float)])
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        # env.render()

        cumul_reward += rewards
        if info.get('abandon_amount'):
            count_abandon += info.get('abandon_amount')
        if info.get('success'):
            count_success += 1
        if info.get('excess'):
            count_excess += 1
        if info.get('failed_use'):
            count_failed += 1
        if info.get('set-up'):
            count_set_up += 1
        total_time = info['usetime']
        print("action: ", action_name[action])
        # print("reward this step: ", rewards)
        # print("total reward: ", cumul_reward)
        print("reward this step: ", "{:.1f}".format(float(rewards)))
        print("total reward: ", "{:.1f}".format(float(cumul_reward)))
        print("=" * 50)

        production_ready = production_ready.append({
            'require_mol': [round(x, 1) for x in np.array([env.state[0]], dtype=float)],
            'mol_name': [round(x, 1) for x in np.array([env.state[1]], dtype=float)],
            'amount_setup': [round(x, 1) for x in np.array([env.state[2]], dtype=float)],
            'device': [round(x, 1) for x in np.array([env.state[3]], dtype=float)],
            'time': [round(x, 1) for x in np.array([env.state[4]], dtype=float)],
            'Action': action_name[action],
        }, ignore_index=True)

        if dones:
            break

    print("Number of successful processing: ", count_success)
    print("Total abandon amount", count_abandon)
    print("Total excess count", count_excess)
    print('Less than 80% usage', count_failed)
    print('Number of set-up', count_set_up)
    print('Total time required: ', "{:.1f}".format(float(total_time/24)), 'days')
    total_time_u = math.ceil(total_time)
    production_ready = production_ready.applymap(lambda x: str(x).lstrip('[').rstrip(']'))
    rein_result = production_ready[~production_ready.duplicated(keep='first')]

    exclude_columns = ['Action']
    for column in rein_result.columns:
        if column not in exclude_columns:
            # rein_result[column] = rein_result[column].astype(float)
            rein_result.loc[:, column] = rein_result[column].astype(float)

    exclude_columns2 = ['require_mol', 'amount_setup' 'Action']
    for column in rein_result.columns:
        if column not in exclude_columns:
            # rein_result[column] = rein_result[column].astype(int)
            rein_result.loc[:, column] = rein_result[column].astype(int)


    production_plan = pd.DataFrame(columns=['Time', 'model_1', 'model_2', 'model_3', 'model_4'])

    date = order_data.loc[0, 'time']
    date_time = datetime.strptime(date, "%Y-%m-%d")
    form = date_time.strftime("%Y-%m-%d %H:%M")
    base_time = datetime.strptime(form, "%Y-%m-%d %H:%M")

    production_plan['Time'] = [base_time + timedelta(hours=i) for i in range(total_time_u)]
    production_plan[['model_1', 'model_2', 'model_3', 'model_4']] = 0

    use_time = 0
    for i in range(len(rein_result)):
        if rein_result.iloc[i, 5] == 'Device 1 set-up':  # 소자 1 셋업
            use_time = 150 - rein_result.iloc[i, 2]

        elif rein_result.iloc[i, 5] == 'Device 2 set-up':  # 소자 2 셋업
            use_time = 130 - rein_result.iloc[i, 2]

        elif rein_result.iloc[i, 5] == 'Not set-up':  # 소자 셋업 x
            if i == 0:
                use_time = 0
            elif i != 0:
                use_time = rein_result.iloc[i - 1, 2] - rein_result.iloc[i, 2]

        if rein_result.iloc[i, 5] == 'Device 1 set-up' or rein_result.iloc[i, 5] == 'Device 2 set-up':
            production_plan.iloc[rein_result.iloc[i, 4] - 28 - use_time: rein_result.iloc[i, 4] - use_time,
            rein_result.iloc[i, 1]] = 's'  # 셋업 28시간
            if use_time > 0:
                production_plan.iloc[rein_result.iloc[i, 4] - use_time: rein_result.iloc[i, 4], rein_result.iloc[i, 1]] = 6

        elif rein_result.iloc[i, 5] == 'Not set-up':
            if use_time > 0:
                if rein_result.iloc[i, 1] != rein_result.iloc[i - 1, 1]:
                    production_plan.iloc[rein_result.iloc[i, 4] - use_time - 6: rein_result.iloc[i, 4] - use_time, rein_result.iloc[i, 1]] = 'c'
                    production_plan.iloc[rein_result.iloc[i, 4] - use_time: rein_result.iloc[i, 4], rein_result.iloc[i, 1]] = 6
                elif rein_result.iloc[i, 1] == rein_result.iloc[i - 1, 1]:
                    production_plan.iloc[rein_result.iloc[i, 4] - use_time: rein_result.iloc[i, 4], rein_result.iloc[i, 1]] = 6

    #production_plan.to_excel('C:/SDOLab-server/황인근/디스플레이 강화학습/production_plan_5.xlsx', index=False)