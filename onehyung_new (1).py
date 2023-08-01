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
from gym.utils import seeding  # random seed control 위해 import
# 납기 코드 수정(주문양만큼 비례)
# 버려지는 소자의 TERM 바꾸기 MAX(버려지는 소자 -20, 0)

import os
from stable_baselines.common.vec_env import VecEnv, sync_envs_normalization, DummyVecEnv
from typing import Union, List, Optional, Tuple

from stable_baselines import DQN, PPO2
from stable_baselines.common import callbacks
from stable_baselines.common.callbacks import EvalCallback, BaseCallback
from load_data import load_data

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

    epi_rewards_discounted, epi_success_order, epi_abandon_amount, epi_excess_order, epi_setup, epi_times = [], [], [], [], [], []
    for i in range(n_eval_episodes):
        if not isinstance(env, VecEnv) or i == 0:
            obs = env.reset()
        done, state = False, None
        epi_reward_discounted = 0.0
        epi_success = 0
        epi_abandon = 0
        epi_excess = 0
        epi_set = 0
        epi_time = 0
        num_steps = 0
        while not done:
            action, state = model.predict(obs, state=state)
            obs, reward, done, _info = env.step(action)
            epi_reward_discounted += np.power(model.gamma, num_steps) * reward
            num_steps += 1

            if _info[0].get('abandon_amount'):
                epi_abandon += _info[0].get('abandon_amount')
            if _info[0].get('success'):
                epi_success += 1
            if _info[0].get('excess'):
                epi_excess += 1
            if _info[0].get('set_up'):
                epi_set += 1

            epi_time = float(_info[0]['usetime'] / 24)

            if render:
                env.render()
        epi_rewards_discounted.append(epi_reward_discounted)
        epi_success_order.append(epi_success)
        epi_abandon_amount.append(epi_abandon)
        epi_excess_order.append(epi_excess)
        epi_setup.append(epi_set)
        epi_times.append(epi_time)
    mean_discounted_reward = np.mean(epi_rewards_discounted)
    std_discounted_reward = np.std(epi_rewards_discounted)
    if return_episode_rewards:
        return epi_rewards_discounted, epi_success_order, epi_abandon_amount, epi_excess_order, epi_setup, epi_times
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
        self.results_setup = []
        self.best_mean_abandon = 1000000

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            epi_rewards_discounted, epi_success_order, epi_abandon_amount, epi_excess_order, epi_setup, epi_times \
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
                self.results_setup.append(epi_setup)
                self.evaluations_length.append(epi_times)
                np.savez(self.log_path, timesteps=self.evaluations_timesteps,
                         results_discounted=self.results_discounted,
                         results_success_order=self.results_success_order,
                         results_abandon=self.results_abandon,
                         results_excess=self.results_excess,
                         results_setup=self.results_setup,
                         ep_lengths=self.evaluations_length)

            mean_reward_discounted, std_reward_discounted = np.mean(epi_rewards_discounted), np.std(epi_rewards_discounted)
            mean_success, std_success = np.mean(epi_success_order), np.std(epi_success_order)
            mean_abandon, std_abandon = np.mean(epi_abandon_amount), np.std(epi_abandon_amount)
            mean_excess, std_excess = np.mean(epi_excess_order), np.std(epi_excess_order)
            mean_setup, std_setup = np.mean(epi_setup), np.std(epi_setup)
            mean_ep_length, std_ep_length = np.mean(epi_times), np.std(epi_times)
            # Keep track of the last evaluation, useful for classes that derive from this callback
            self.last_mean_reward = mean_reward_discounted

            if self.verbose > 0:
                print("Eval num_timesteps={}, "
                      "episode_discounted_reward={:.2f} +/- {:.2f}".format(self.num_timesteps, mean_reward_discounted, std_reward_discounted),
                      "episode_success={:.2f} +/- {:.2f}".format(mean_success, std_success),
                      "episode_abandon={:.2f} +/- {:.2f}".format(mean_abandon, std_abandon),
                      "episode_excess={:.2f} +/- {:.2f}".format(mean_excess, std_excess),
                      "episode_setup={:.2f} +/- {:.2f}".format(mean_setup, std_setup))
                print("Episode day: {:.2f} +/- {:.2f}".format(mean_ep_length, std_ep_length))

            # if mean_success < 1.0e-4 and self.n_calls % (self.eval_freq*5):
            #     self.model.setup_model()

            # if mean_reward_discounted > self.best_mean_reward:
            if mean_abandon < self.best_mean_abandon:
                if mean_success > 108:
                    if self.verbose > 0:
                        # print("New best mean reward!")
                        print("New best mean abandon!")
                    if self.best_model_save_path is not None:
                        self.model.save(os.path.join(self.best_model_save_path, 'best_model3'))
                    self.model.save(os.path.join(self.best_model_save_path, 'model' + str(self.num_timesteps)))
                    self.best_mean_abandon = mean_abandon
                    # self.best_mean_reward = mean_reward_discounted
                    # Trigger callback if needed
                    if self.callback is not None:
                        return self._on_event()

        return True


class Display(gym.Env):
    def __init__(self, data):

        #load_data 통해 필요한 변수들 준비해놓기
        self.load = load_data(data)

        # order 파일
        self.demand = self.load.create_demand('성형1공장')
        # 주문 총개수
        self.demand_num = self.demand[self.demand != 0].count().sum()
        # 소자 종류 정보
        self.device = self.load.device_load('성형1공장', 1)
        # 셋업 소요 시간 가져오기
        self.pm = self.load.make_array_PM('성형1공장', 1)
        # 모델 체인지 시간 가져오기
        self.mc = self.load.make_array_MC('성형1공장', 1)
        # 원형 조달 시간
        self.tat = self.load.make_dict_TAT('성형1공장', 1)
        # 성형 & 원형 제작 개당 소요 시간
        self.tt = self.load.make_dict_AS('성형1공장', 1)
        # 현재 주문을 충족시키기 위해 필요한 PRT 종류 개수
        self.PRT_num = len(self.load.create_prt('성형1공장'))
        # 주문 충족을 위해 필요한 PRT 종류 불러오기
        self.prt = self.load.create_prt('성형1공장')
        # 소자 최대 가동시간 정보
        self.load_time = self.load.max_load('성형1공장')



        # 성형 양품률
        self.molding_rate = 0.975
        # 자르기 양품률
        self.cut_rate = 0.9

        self.action_space = spaces.Discrete(len(self.device) + 1)  # 소자 종류 수 만큼 액션 + stay 액션
        self.observation_space = spaces.MultiDiscrete([self.demand.max().max()+1, len(self.demand.columns)+1, max(self.load_time)+1, len(self.device) + 1, 5000])


        # 하나의 mol 성형이 끝났을 때
        self.reward_per_success = 1

        # # 세팅한 소자를 모두 사용했을 때
        self.reward_operation_max = 60

        # 낮은 가동률에 대한 패널티
        self.reward_operation_rate1 = 10
        self.reward_operation_rate2 = 5

        # 잘못된 셋업에 대한 패널티
        self.reward_per_miss = 500

        # 원형에서 생산해야하는 PRT 수 count
        self.required_PRT = []


        self.viewer = None

        self.reset()



    def excess_penalty(self, due_time, c_time, amount):  # 현재 성형해야 하는 mol의 납기가 지났을 때, 패널티
        penalty1 = 0
        coef1 = 0.00001
        # processing_date = datetime(year, month, day)
        # due_date = datetime(year, due_month, due_day)

        difference = c_time - due_time
        # difference = delta.days
        penalty1 += coef1 * amount * (difference ** 2)

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
                    penalty_mol3 = coef4 * (1 / (due_time3 - c_time + 1))
                elif int(due_time3) - int(c_time) < 0:
                    penalty_mol3 = coef3 * (abs(c_time - due_time3)) ** 2

            if (self.demand['MOL_4'] != 0).any() == True:
                for j in range(len(demand)):
                    if demand.loc[j, 'MOL_4'] != 0:
                        mol_4_zero = j
                        break
                due_time4 = demand.iloc[mol_4_zero, 0]
                if int(due_time4) - int(c_time) >= 0:
                    penalty_mol4 = coef4 * (1 / (due_time4 - c_time + 1))
                elif int(due_time4) - int(c_time) < 0:
                    penalty_mol4 = coef3 * (abs(c_time - due_time4)) ** 2

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
                    penalty_mol1 = coef4 * (1 / (due_time1 - c_time + 1))
                elif int(due_time1) - int(c_time) < 0:
                    penalty_mol1 = coef3 * (abs(c_time - due_time1)) ** 2

            if (self.demand['MOL_2'] != 0).any() == True:
                for l in range(len(demand)):
                    if demand.loc[l, 'MOL_2'] != 0:
                        mol_2_zero = l
                        break
                due_time2 = demand.iloc[mol_2_zero, 0]
                if int(due_time2) - int(c_time) >= 0:
                    penalty_mol2 = coef4 * (1 / (due_time2 - c_time + 1))
                elif int(due_time2) - int(c_time) < 0:
                    penalty_mol2 = coef3 * (abs(c_time - due_time2)) ** 2

            penalty3 = penalty_mol1 + penalty_mol2

            return penalty3

    def abandon_penalty(self, p_amount_set_up, p_device):  # 세팅한 소자를 다 쓰지 않고 버렸을 때 패널티
        penalty2 = 0
        coef2 = 1.8
        y = 0

        y = max(p_amount_set_up - self.device[p_device] * 0.2, 0)
        penalty2 += coef2 * (y ** 2)

        return penalty2

    def calculate_utilization(self, running_times, setup_times, model_change_times):  # 낮은 가동률에 대한 패널티
        total_running_time = sum(running_times)
        total_setup_time = sum(setup_times)
        total_model_change_time = sum(model_change_times)
        total_times = total_model_change_time + total_setup_time + total_running_time
        if int(total_times) == 0:
            return 0
        else:
            utilization = float(total_running_time) / float(total_times)
            return utilization


    def check_feasibility(self, line, amount, time-tat, PRT_num, PRT_stock):
        # PRT_demand와 현재 시간(날짜)를 가지고 원형 공정에서 납기가 가능한지 체크
        feasibility = True
        start_PRT = time-tat
        capa = self.load.check_linecapa(line, time-tat)

        # 현재 다루고 있는 목표 모델 line의 원형공정 모델 생산 리스트 불러오기
        line_PRT = [lst for lst in self.required_PRT if lst[0] == line]


        # 현재 다루고 있는 목표 모델의 time-tat보다 작은 모델 생산 리스트 불러오기
        constraint1 = [lst for lst in line_PRT if lst[2] < time-tat]

        # constraint1에서 현재 사용하고 있는 capa 합
        constraint2 = [sum(lst[1] for lst2 in constraint1)]

        # required_capa가 amount보다 크거나 같아야 True
        required_capa = capa - constraint2

        # 불가능 시 False
        if required_capa < amount:
            feasibility = False


        return feasibility

    def load_model(self, p_device, selected_device, p_demand, demand, current_time, p_mol_name):

        while True:
            #이전과 같은 소자를 셋업한 경우
            if p_device == device:
                # 투입 모델 선정
                name = self.device[selected_device]
                mol_cols = [col for col in demand.columns if name in col]
                new_df = demand[mol_cols]  # 셋업된 소자를 사용하는 mol들로 구성된 새로운 df 생성
                # 연속생산
                for i in range(len(new_df)):
                    if (new_df.loc[new_df.index[i], p_mol_name] != 0).any():
                        time = new_df.index[i]
                        amounts = new_df.loc[time, p_mol_name]
                        info = pd.DataFrame({'time': [time],
                                             'mol_name': p_mol_name,
                                             'amount': amounts})

                    else:
                        # 납기일 빠른 거 먼저 생산
                        for i in range(len(new_df)):
                            if (new_df.iloc[i] != 0).any():
                                time = new_df.index[i]
                                mol_names = new_df.iloc[i][new_df.iloc[i] != 0].index.tolist()
                                amounts = new_df.iloc[i][new_df.iloc[i] != 0].tolist()
                                break
                            # 주문이 없을 때
                            else:
                                time = 0
                                mol_names = 0
                                amounts = 0

                        # 납기일 동일한 게 존재한다면 생산량이 더 큰 걸 선택
                        if len(mol_names) > 1:
                            amount = max(amounts)
                            mol_name = mol_names[amounts.index(amount)]
                            info = pd.DataFrame({'time': [time],
                                                 'mol_name': mol_name,
                                                 'amount': amount})
                        else:
                            info = pd.DataFrame({'time': [time],
                                                 'mol_name': mol_names,
                                                 'amount': amounts})

            else:
                # 투입 모델 선정
                name = self.device[selected_device]
                mol_cols = [col for col in demand.columns if name in col]
                new_df = demand[mol_cols]
                # 납기일 빠른 거 먼저 생산
                for i in range(len(new_df)):
                    if (new_df.iloc[i] != 0).any():
                        time = new_df.index[i]
                        mol_names = new_df.iloc[i][new_df.iloc[i] != 0].index.tolist()
                        amounts = new_df.iloc[i][new_df.iloc[i] != 0].tolist()
                        break
                    else:
                        time = 0
                        mol_names = 0
                        amounts = 0

                # 납기일 동일한 게 존재한다면 생산량이 더 큰 걸 선택
                if len(mol_names) > 1:
                    amount = max(amounts)
                    mol_name = mol_names[amounts.index(amount)]
                    info = pd.DataFrame({'time': [time],
                                         'mol_name': mol_name,
                                         'amount': amount})
                else:
                    info = pd.DataFrame({'time': [time],
                                         'mol_name': mol_names,
                                         'amount': amounts})


            selectedPRT = demand.columns.get_loc(info['mol_name'])
            PRT_name = self.prt[selectedPRT]
            line_num = self.load.check_line(PRT_name)
            ratio = self.load.check_ratio(info['mol_name'])
            tat = self.tat[PRT_name]
            demand_delay = []

            # 첫번째 load_model 우선순위 상 가능한지 확인
            #for문으로 demand_delay에 있는 게 먼저 생산 가능한지 확인
            for i in range(len(demand_delay)):
                selectedPRT2 = demand.columns.get_loc(demand_delay[i]['mol_name'])
                PRT_name = self.prt[selectedPRT2]
                line_num = self.load.check_line(PRT_name)
                ratio2 = self.load.check_ratio(demand_delay[i]['mol_name'])
                tat = self.tat[PRT_name]
                if self.check_feasibility(line_num, demand_delay[i]['amount'] * ratio2, time - tat):
                    # PRT 필요량 업데이트
                    self.required_PRT.append([line_num, demand_delay[i]['amount'] * ratio2, time - tat])
                    del(demand_delay[i])
                    # self.required_PRT[selectdPRT] += amount * ratio  # selectedPRT = 선택된 모델이 필요로하는 PRT 번호, amount = 필요한 prt 야
                    break


            if self.check_feasibility(line_num, info['amount'] * ratio, time - tat):
                # PRT 필요량 업데이트
                self.required_PRT.append([line_num, info['amount'] * ratio, time - tat])
                # self.required_PRT[selectdPRT] += amount * ratio  # selectedPRT = 선택된 모델이 필요로하는 PRT 번호, amount = 필요한 prt 야
                break

            else:
                demand_delay.append(info)
                demand.loc[info['time'], info['mol_name']] = 0
                #0으로 넘겼던 주문들에 대해 따로 리스트를 만들어 나중에 다시 고려하게끔
                pass


            # 두번째 load_model을 명시하는 게 가능한가??, 제약 확인을 loadmodel 밖으로 빼느 건? 두 번째가 else문이 필요할까?  우선순위 상 가능한지 확인
            #else:
            #    selectedPRT2 = demand.columns.get_loc(info['mol_name'])
            #    PRT_name = self.prt[selectedPRT2]
            #    line_num = self.load.check_line(PRT_name)
            #    ratio2 = self.load.check_ratio(info['mol_name'])
            #    tat = self.tat[PRT_name]
            #    if self.check_feasibility(line_num, info['amount'] * ratio2, time - tat):
            #        self.required_PRT.append([line_num, info['amount'] * ratio2, time - tat])
            #        break
            #    else: # 방금 선택된 모델을 demand에서 제외
            #        pass


            return info

    def update_model(self, demand, order_info, required_mol):
        demand.loc[demand['time'] == order_info.loc[0, 'time'], order_info['mol_name']] = required_mol
        return demand


    def step(self, action):
        # 현재 생산해야 하는 mol의 성형에 소요되는 시간 / 현재 셋업된 소자 잔량 / 현재 세팅된 소자 종류 / 현재 날짜
        required_mol, mol_name, amount_set_up, device, c_time = self.state

        # 셋업하기 이전의 소자 잔량 저장
        p_amount_set_up = amount_set_up
        p_device = device
        p_mol_name = mol_name
        p_time = c_time
        # p_amount_set_up = int(p_amount_set_up)
        # amount_set_up = int(amount_set_up)

        reward = 0

        done = False
        self.steps += 1
        info = {}


        # set-up action
        if action == len(self.device):
            amount_set_up += 0
            device = action
            device = p_device
        elif action < len(self.device):
            amount_set_up = self.load_time[action]
            device = action
            c_time += self.pm[p_device][action]
            self.setup_times.append(self.pm[p_device][action])
            info['set_up'] = True
        else:
            raise Exception('bad action {}'.format(action))


        # 모델 투입 룰에 따라 모델 가져오기
        order_info = self.load_model(p_device, device, self.p_demand, self.demand, c_time, p_mol_name)
        self.p_demand = self.demand
        required_mol = order_info['amount'].values[0]

        # mol_name 정의해주기
        for i in range(len(demand.columns)):
            if order_info['mol_name'] == demand.columns[i]:
                mol_name = i

        # 현재 셋업된 소자량으로 생산가능한 mol
        time_per = self.tt[mol_name] / 3600
        possible_mol = amount_set_up / time_per

        # 모델 체인지 시간 반영. 만약 STAY라면 setup과 별개로 모델 체인지 발생
        if action == len(self.device):
            if p_mol_name == mol_name:
                c_time += 0
                self.model_change_times.append(0)
            elif p_mol_name != mol_name:
                if p_device == device:
                    c_time += self.mc[p_mol_name][mol_name]
                    self.model_change_times.append(self.mc[p_mol_name][mol_name])
                else:
                    c_time += self.change_mol[p_mol_name][mol_name]
                    self.model_change_times.append(self.mc[p_mol_name][mol_name])

        # 만약 새로운 셋업 액션을 했더라면,모델 체인지 시간은 setup 내에서 이루어지기 때문에 미포함..
        else:
            self.model_change_times.append(0)
            c_time += 0

        # 버려지는 소자량에 대한 패널티 발생
        if p_amount_set_up > 0 and action != len(self.device):
            reward -= self.abandon_penalty(p_amount_set_up, p_device)
            info['abandon_amount'] = p_amount_set_up
            # 패널티가 존재한다는 것은 80% 보다 못 쓴 것
            if self.abandon_penalty(p_amount_set_up, p_device) > 0:
                info['failed_use'] = True

        # 소자 태우는 상태 전이 표현
        if required_mol < possible_mol and required_mol > 0:
            # self.running_times.append(required_mol / 6)
            reward += (self.reward_per_success * required_mol)
            if required_mol % 6 == 0:
                c_time += required_mol // 6
            else:
                c_time += (required_mol // 6) + 1
            possible_mol -= required_mol
            self.PRT1, self.PRT2, self.PRT3, self.PRT4 = self.cal_prt(self.PRT1, self.PRT2, self.PRT3, self.PRT4, mol_name, required_mol)
            # self.cal_prt(self.PRT1, self.PRT2, self.PRT3, self.PRT4, mol_name, required_mol)
            required_mol = 0
            info['success'] = True
            amount_set_up = possible_mol / 6



        elif required_mol == possible_mol and required_mol > 0:
            self.running_times.append(amount_set_up)
            reward += (self.reward_per_success * required_mol) + self.reward_operation_max
            # reward += self.reward_per_success + self.reward_operation_max
            if required_mol % 6 == 0:
                c_time += required_mol // 6
            else:
                c_time += (required_mol // 6) + 1
            self.PRT1, self.PRT2, self.PRT3, self.PRT4 = self.cal_prt(self.PRT1, self.PRT2, self.PRT3, self.PRT4, mol_name, required_mol)
            # self.cal_prt(self.PRT1, self.PRT2, self.PRT3, self.PRT4, mol_name, required_mol)
            required_mol = 0
            info['success'] = True
            possible_mol = 0
            amount_set_up = possible_mol / 6
            # reward += self.reward_operation_max


        elif required_mol > possible_mol and required_mol > 0 and possible_mol > 0:
            self.running_times.append(amount_set_up)
            reward += (self.reward_per_success * possible_mol) + self.reward_operation_max
            # reward += self.reward_per_success + self.reward_operation_max
            if possible_mol % 6 == 0:
                c_time += possible_mol // 6
            else:
                c_time += (possible_mol // 6) + 1
            required_mol -= possible_mol
            self.PRT1, self.PRT2, self.PRT3, self.PRT4 = self.cal_prt(self.PRT1, self.PRT2, self.PRT3, self.PRT4, mol_name, possible_mol)
            # self.cal_prt(self.PRT1, self.PRT2, self.PRT3, self.PRT4, mol_name, possible_mol)
            possible_mol = 0
            amount_set_up = possible_mol / 6
            # reward += self.reward_operation_max


        # 다은, 정은 수정해야할 부분 601~642 까지
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
                    reward -= self.excess_penalty(due_time, c_time, order_info['amount'].values[0])
                    info['excess'] = True
                else:
                    if order_info['amount'].values[0] in required_mol_list and order_info['mol_name'] in mol_name_list:
                        if due_time_list.index(due_time) == required_mol_list.index(
                                order_info['amount'].values[0]) == mol_name_list.index(order_info['mol_name']):
                            required_mol_list.append(order_info['amount'].values[0])
                        else:
                            due_time_list.append(due_time)
                            required_mol_list.append(order_info['amount'].values[0])
                            mol_name_list.append(order_info['mol_name'])
                            reward -= self.excess_penalty(due_time, c_time, order_info['amount'].values[0])
                            info['excess'] = True
                    elif order_info['amount'].values[0] in required_mol_list and order_info[
                        'mol_name'] not in mol_name_list:
                        due_time_list.append(due_time)
                        required_mol_list.append(order_info['amount'].values[0])
                        mol_name_list.append(order_info['mol_name'])
                        reward -= self.excess_penalty(due_time, c_time, order_info['amount'].values[0])
                        info['excess'] = True
                    elif order_info['amount'].values[0] not in required_mol_list and order_info[
                        'mol_name'] in mol_name_list:
                        due_time_list.append(due_time)
                        required_mol_list.append(order_info['amount'].values[0])
                        mol_name_list.append(order_info['mol_name'])
                        reward -= self.excess_penalty(due_time, c_time, order_info['amount'].values[0])
                        info['excess'] = True
        else:
            due_time_list.append(due_time)
            required_mol_list.append(order_info['amount'].values[0])
            mol_name_list.append(order_info['mol_name'])


        info['usetime'] = c_time
        # required_mol = order_info['result']
        self.demand = self.update_model(self.demand, order_info, required_mol)

        # order에 주어진 모든 주문을 처리한 경우, 종료
        if (self.demand.astype(int)[['MOL_1', 'MOL_2', 'MOL_3', 'MOL_4']] == 0).all().all():
            done = True
        if self.steps == 1000:
            done = True

        self.state = (required_mol, mol_name, amount_set_up, device, c_time)
        return np.array(self.state, dtype=object), reward, done, info

    def reset(self):
        self.state = (0, 2, 0, 0,) + (0,)

        self.steps = 0
        self.running_times = []
        self.setup_times = []
        self.model_change_times = []

        self.PRT = [0] * self.PRT_num



        # 에피소드 초기화 됐을 때, 모든 주문이 0된거 말고, 새로운 주문 파일을 불러와야함.(아직 수정 X)
        self.demand = pd.DataFrame(self.update_order, columns=["time", "MOL_1", "MOL_2", "MOL_3", "MOL_4"])
        self.demand['time'] = pd.to_datetime(self.demand['time'])
        self.demand['time'] = pd.to_datetime(self.demand['time'].dt.date) + pd.to_timedelta('18:00:00')
        self.base_time = datetime(2020, 4, 1, 18)
        self.demand['time'] = self.demand['time'].apply(lambda x: int((x - self.base_time).total_seconds() / 3600))
        self.p_demand = self.demand.copy()

        # 필요한 PRT 수 reset
        self.required_PRT = np.zeros(self.PRT_num) # PRT_num = PRT 종류 수

        return np.array(self.state)

    def render(self):
        pass

    def close(self):
        pass


if __name__ == '__main__':
    action_name = {0: "Device 1 set-up", 1: "Device 2 set-up", 2: "Not set-up"}

    data = pd.read_csv('C:/SDOLab-server/황인근/디스플레이 강화학습/order.csv')

    env = Display(data=data)
    eval_env = Display(data=data)

    cb = EvalCallback_Display(eval_env=eval_env, n_eval_episodes=10, eval_freq=1000,
                              log_path="./model",
                              best_model_save_path="./best_model"
                              )
    model = PPO2('MlpPolicy', env, verbose=0)
    # total_timesteps = int(3.0e4)
    # model.learn(total_timesteps=total_timesteps, callback=cb)
    # #
    # eval_env = DummyVecEnv([lambda: eval_env])
    # epi_rewards_discounted, epi_success_order, epi_abandon_amount, epi_excess_order, epi_setup, epi_times \
    #     = evaluate_policy_Display(model, eval_env,
    #                               n_eval_episodes=10,
    #                               render=False,
    #                               deterministic=False,
    #                               return_episode_rewards=True)
    # mean_reward_discounted, std_reward_discounted = np.mean(epi_rewards_discounted), np.std(epi_rewards_discounted)
    # mean_success, std_success = np.mean(epi_success_order), np.std(epi_success_order)
    # mean_abandon, std_abandon = np.mean(epi_abandon_amount), np.std(epi_abandon_amount)
    # mean_excess, std_excess = np.mean(epi_excess_order), np.std(epi_excess_order)
    # mean_setup, std_setup = np.mean(epi_setup), np.std(epi_setup)
    # mean_ep_length, std_ep_length = np.mean(epi_times), np.std(epi_times)
    # #
    # print(
    #     "episode_discounted_reward={:.2f} +/- {:.2f}".format(mean_reward_discounted,
    #                                                          std_reward_discounted),
    #     "episode_success={:.2f} +/- {:.2f}".format(mean_success, std_success),
    #     "episode_abandon={:.2f} +/- {:.2f}".format(mean_abandon, std_abandon),
    #     "episode_excess={:.2f} +/- {:.2f}".format(mean_excess, std_excess),
    #     "episode_setup={:.2f} +/- {:.2f}".format(mean_setup, std_setup))
    # # "Episode day: {:.2f} +/- {:.2f}".format(mean_ep_length, std_ep_length))
    # print("Episode day: {:.2f} +/- {:.2f}".format(mean_ep_length, std_ep_length))

    #
    # Enjoy trained agent
    #model = PPO2.load("C:/SDOLab-server/황인근/디스플레이 강화학습/best_model/best_model3.zip")
    # model = PPO2.load("C:/SDOLab-server/황인근/디스플레이 강화학습/best_model/model17000.zip")
    # model = PPO2('MlpPolicy', env, verbose=1)
    # model.learn(total_timesteps=int(3.0e4))
    #
    # order_data = pd.read_csv('C:/SDOLab-server/황인근/디스플레이 강화학습/order.csv')
    # env = Display(order_data=order_data, cut_data=cut_data, setup_time=setup_time, device_max_1=device_max_1,
    #                device_max_2=device_max_2)

    obs = env.reset()
    count_abandon = 0
    count_success = 0
    count_excess = 0
    count_failed = 0
    count_set_up = 0
    cumul_reward = 0
    # production_ready = pd.DataFrame(columns=['require_mol', 'mol_name', 'amount_setup', 'device', 'time', 'Action']) # 학습 결과를 출력하기 위해 state와 action 들어갈 수 있는 틀 만들어놓기
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
        if info.get('set_up'):
            count_set_up += 1


        total_time = info['usetime']
        print("action: ", action_name[action])
        # print("reward this step: ", rewards)
        # print("total reward: ", cumul_reward)
        print("reward this step: ", "{:.1f}".format(float(rewards)))
        print("total reward: ", "{:.1f}".format(float(cumul_reward)))
        print("=" * 50)


        ## 위에서 만들어 놓은 틀에 실제 state랑 action 넣기
        # production_ready = production_ready.append({
        #     'require_mol': [round(x, 1) for x in np.array([env.state[0]], dtype=float)],
        #     'mol_name': [round(x, 1) for x in np.array([env.state[1]], dtype=float)],
        #     'amount_setup': [round(x, 1) for x in np.array([env.state[2]], dtype=float)],
        #     'device': [round(x, 1) for x in np.array([env.state[3]], dtype=float)],
        #     'time': [round(x, 1) for x in np.array([env.state[4]], dtype=float)],
        #     'Action': action_name[action],
        # }, ignore_index=True)

        if dones:
            break

    print("Number of successful processing: ", count_success)
    print("Total abandon amount", count_abandon)
    print("Total excess count", count_excess)
    print("Number of set up", count_set_up)
    print('Less than 80% usage', count_failed)
    print('Total time required: ', "{:.1f}".format(float(total_time/24)), 'days')

    # # 여기서부터 엑셀 뽑는 코드
    # total_time_u = math.ceil(total_time) #총 소요 시간 소수점 없애고 올림 해주기.
    # production_ready = production_ready.applymap(lambda x: str(x).lstrip('[').rstrip(']')) #state랑 action 저장해놓은거에서 필요없는 괄호 없애주기
    # rein_result = production_ready[~production_ready.duplicated(keep='first')] #prudoction_ready 파일에서 중복된 행은 없애주기
    #
    #
    # # rein_result 프레임에서 'Action' 열 제외하고 float으로 변환
    # exclude_columns = ['Action']
    # for column in rein_result.columns:
    #     if column not in exclude_columns:
    #         rein_result.loc[:, column] = rein_result[column].astype(float)
    #
    # # rein_result 프레임에서 'require_mol, amount_set_up, Action' 빼고 정수로 변환
    # exclude_columns2 = ['require_mol', 'amount_setup', 'Action']
    # for column in rein_result.columns:
    #     if column not in exclude_columns2:
    #         rein_result.loc[:, column] = rein_result[column].astype(int)
    #
    #
    #
    # # 생산계획 출력할 파일 틀 만들어놓기
    # production_plan = pd.DataFrame(columns=['Time', 'MOL_1', 'MOL_2', 'MOL_3', 'MOL_4'])
    #
    #
    # # 생산계획 파일에 시간을 추가하기 위해 형식 만들어놓기
    # date = order_data.loc[0, 'time']
    # date_time = datetime.strptime(date, "%Y-%m-%d")
    # form = date_time.strftime("%Y-%m-%d %H:%M")
    # base_time = datetime.strptime(form, "%Y-%m-%d %H:%M")
    #
    #
    # # 이제 생산계획 파일에 Time 열에 값 넣어주기, 위에서 보여줬던 total time_u 까지 1시간씩 오름차순으로 넣음
    # production_plan['Time'] = [base_time + timedelta(hours=i) for i in range(total_time_u)]
    # # 일단 모델 열들에는 0으로 넣어놓기
    # production_plan[['MOL_1', 'MOL_2', 'MOL_3', 'MOL_4']] = 0
    #
    #
    # # 사용 시간 초기화
    # use_time = 0
    #
    # # 이제 모델 열들에 있는 0 값들을 생산계획으로 채워넣어주기 위해 for 문을 돌린다.
    # for i in range(len(rein_result)):
    #     if i != 0:
    #         use_time = rein_result.iloc[i, 4] - rein_result.iloc[i - 1, 4]
    #     else:
    #         use_time = rein_result.iloc[i, 4]
    #
    #     if rein_result.iloc[i, 5] == 'Device 1 set-up' or rein_result.iloc[i, 5] == 'Device 2 set-up':
    #         if use_time > 0:
    #             production_plan.iloc[rein_result.iloc[i, 4] - use_time: rein_result.iloc[i, 4] - use_time + 28, rein_result.iloc[i, 1]] = 's'
    #             production_plan.iloc[rein_result.iloc[i, 4] - use_time + 28: rein_result.iloc[i, 4], rein_result.iloc[i, 1]] = 6
    #     elif rein_result.iloc[i, 5] == 'Not set-up':
    #         if rein_result.iloc[i, 1] != rein_result.iloc[i - 1, 1]:
    #             if use_time > 0:
    #                 production_plan.iloc[rein_result.iloc[i, 4] - use_time: rein_result.iloc[i, 4] - (use_time - 6), rein_result.iloc[i, 1]] = 'c'
    #                 production_plan.iloc[rein_result.iloc[i, 4] - (use_time - 6): rein_result.iloc[i, 4], rein_result.iloc[i, 1]] = 6
    #         else:
    #             if use_time > 0:
    #                 production_plan.iloc[rein_result.iloc[i, 4] - use_time: rein_result.iloc[i, 4], rein_result.iloc[i, 1]] = 6
    # production_plan.to_excel('production_plan.xlsx')  # 생산계획 출력
    # #production_plan.to_excel('C:/SDOLab-server/황인근/디스플레이 강화학습/production_plan_5.xlsx', index=False) # 생산계획 출력