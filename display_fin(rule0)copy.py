
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gym
import pandas as pd
from gym import spaces
import numpy as np
from datetime import datetime, date, time, timedelta
import time
from calendar import monthrange
from gym.utils import seeding  # random seed control 위해 import

import os
from stable_baselines.common.vec_env import VecEnv, sync_envs_normalization, DummyVecEnv
from typing import Union, List, Optional, Tuple

from stable_baselines import DQN, PPO2
from stable_baselines.common.callbacks import EvalCallback, BaseCallback


class Displapy(gym.Env):
    def __init__(self, order_data, cut_data, setup_time, device_max_1, device_max_2):

        ## 주문 정보 받아오기
        self.order = order_data
        ## 셋업에 소요되는 시간 받아오기
        self.setup_time = setup_time

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
        # 같은 소자를 쓰는 mol끼리의 변경 시간
        self.change_mol = (6, 13)

        # 성형 양품률
        self.molding_rate = 0.975
        # 자르기 양품률
        self.cut_rate = 0.9

        ## observation_space 나타내기 위한 max 함수들
        self.p_m_time = max(self.p_m1_time, self.p_m2_time, self.p_m3_time, self.p_m4_time)
        self.device_max = max(device_max_1, device_max_2)

        ## 소자 종류 -> 0 : 소자1, 1 : 소자 2 존재
        self.device_type = (0, 1)

        self.max_data = (self.order.drop(['time'], axis=1)).max().max()
        # self.max_time = ((self.max_data / self.molding_rate) / 506) / self.cut_rate * 70

        self.action_space = spaces.Discrete(5)  # SETUP1, SETUP2, SETUP1(50%), SETUP2(50%),  STAY
        self.observation_space = spaces.MultiDiscrete([10000, 4, self.device_max, 2, 10000])

        # 하나의 mol 성형이 끝났을 때
        self.reward_per_success = 0.5

        # 하나의 mol을 완전히 성형하진 못했지만, 투입은 해놓은 상태라면
        self.reward_per_success2 = 15

        # 세팅한 소자를 모두 사용했을 때
        self.reward_operation_max = 100

        # 낮은 가동률에 대한 패널티
        self.reward_operation_rate1 = 10
        self.reward_operation_rate2 = 5

        # 잘못된 셋업에 대한 패널티
        self.reward_per_miss = 100

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

    def excess_penalty(self, due_time, c_time):  # 현재 성형해야 하는 mol의 납기가 지났을 때, 패널티
        penalty1 = 0
        coef1 = 0.05
        # processing_date = datetime(year, month, day)
        # due_date = datetime(year, due_month, due_day)

        difference = c_time - due_time
        # difference = delta.days
        penalty1 += coef1 * (difference ** 2)

        return penalty1

    def excess_penalty2(self, device, c_time, demand):
        coef3 = 0.05
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

    def abandon_penalty(self, p_amount_set_up):  # 세팅한 소자를 다 쓰지 않고 버렸을 때 패널티
        penalty2 = 0
        coef2 = 2
        y = p_amount_set_up
        penalty2 += coef2 * y

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
                    result = []
                    for i in range(len(demand)):  # 모든 행 탐색
                        if demand.iloc[i, 1] > demand.iloc[i, 2]:  # 같은 행에서 더 큰 값을 가지는 열 먼저 result에 추가
                            result.append(demand.iloc[i, 1])
                            result.append(demand.iloc[i, 2])
                        else:
                            result.append(demand.iloc[i, 2])
                            result.append(demand.iloc[i, 1])
                    amount = next((x for x in result if x != 0),
                                  None)  # result에서 0이 아닌 첫번째 요소를 값으로 가짐 => 납기일 제일 빠르고 남기일 동일 시 주문양 더 많은 거에 해당
                    if amount is not None:
                        mol_name = demand.columns[(demand == amount).any()][0]
                        time = demand.loc[demand[(demand[mol_name] == amount)].index[0], 'time']
                        info = pd.DataFrame({'time': [time],
                                             'mol_name': [mol_name],
                                             'amount': [amount]})
                    # else: #amount=None이라면 즉, 모든 열의 값이 0이라면 => 두 모델 모두 필요한 주문양 생산완료
                    #    mol_name = 'MOL_1'
                    #    time = 0
                    #    amount = 0
                    #    info = pd.DataFrame({'time': [time],
                    #                         'mol_name': [mol_name],
                    #                         'amount': [amount]})
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
                if (demand['MOL_3'] != 0).any() == True and (demand['MOL_4'] != 0).any() == True:
                    result = []
                    for i in range(len(demand)):
                        if demand.iloc[i, 3] > demand.iloc[i, 4]:
                            result.append(demand.iloc[i, 3])
                            result.append(demand.iloc[i, 4])
                        else:
                            result.append(demand.iloc[i, 4])
                            result.append(demand.iloc[i, 3])
                    amount = next((x for x in result if x != 0), None)
                    if amount is not None:
                        mol_name = demand.columns[(demand == amount).any()][0]
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
        # load_model(1,0,demand,0,1)
        # demand.loc[:, (info['time'], info['mol_name'])]
        # demand.loc[demand['time']==info['time'],info['mol_name']]
        # demand.loc[demand['time'] == info.loc[0, 'time'],info['mol_name']]
        # load_model(1,0,p_demand,demand,0,1)
    def update_model(self, demand, order_info, required_mol):
        demand.loc[demand['time'] == order_info.loc[0, 'time'], order_info['mol_name']] = required_mol
        return demand

    def step(self, action):
        # 현재 생산해야 하는 mol의 성형에 소요되는 시간 / 현재 셋업된 소자 잔량 / 현재 세팅된 소자 종류 / 현재 날짜
        required_mol, mol_name, amount_set_up, device, c_time = self.state
        # q러닝, 다이나믹 프로그래밍으로 할 수 있는지
        # month, day , c_time을 하나로 합쳐볼 것
        # 모델 투입 룰을 다양하게 가져가 볼 것. 모든 case에서도 강화학습이 잘 학습이 되도록 강화학습 a vs 강화학습 b 각기 다른 룰
        # case 2에 대해서, 연속생산에 대해서 일 수는 ga로 찾아가는 것도 방법 (확장의 방안)
        # 기존연구, 관련 연구와의 차별성 (이미 공유된 두 논문부터 찾아볼 것)

        # 셋업하기 이전의 소자 잔량 저장
        p_amount_set_up = amount_set_up
        p_device = device
        p_mol_name = mol_name
        p_amount_set_up = int(p_amount_set_up)
        amount_set_up = int(amount_set_up)

        reward = 0

        # 의사결정 시점 5번마다 가동률을 체크한 후 패널티 발생
        if self.steps % 5 == 0:
            if self.calculate_utilization(self.running_times, self.setup_times, self.model_change_times) < 0.7:
                # reward -= self.reward_operation_rate1
                self.running_times.clear()
                self.setup_times.clear()
                self.model_change_times.clear()
            elif self.calculate_utilization(self.running_times, self.setup_times, self.model_change_times) < 0.75:
                # reward -= self.reward_operation_rate2
                self.running_times.clear()
                self.setup_times.clear()
                self.model_change_times.clear()

        done = False
        self.steps += 1
        # order에 주어진 모든 주문을 처리한 경우, 종료
        if (self.demand.astype(int)[['MOL_1', 'MOL_2', 'MOL_3', 'MOL_4']] == 0).all().all():
            done = True
        if self.steps == 1000:
            done = True
        info = {}

        # set-up action
        if action == 0:  # 소자1을 셋업
            amount_set_up = device_max_1
            device = 0
            c_time += self.setup_time
            self.setup_times.append(self.setup_time)
        elif action == 1:  # 소자2를 셋업
            amount_set_up = device_max_2
            device = 1
            c_time += self.setup_time
            self.setup_times.append(self.setup_time)
        elif action == 2: #소자 1을 50% 셋업
            amount_set_up = 0.5*device_max_1
            device = 0
            c_time += self.setup_time
        elif action == 3: # 소자 2를 50% 셋업
            amount_set_up = 0.5*device_max_2
            device = 1
            c_time += self.setup_time
        elif action == 4:  # 아무런 셋업하지 않음 -> stay
            amount_set_up += 0
            if p_device == 0:
                device = 0
            else:
                device = 1
        else:
            raise Exception('bad action {}'.format(action))

        # if required_mol > 0 and amount_set_up == 0:
        #     reward -= 100
        # elif required_mol > 0 and amount_set_up > 0:
        #     reward += 10
        # elif required_mol == 0 and amount_set_up <= 50:
        #     reward += 10
        #     0
        # month, day, c_time = self.plustime(month, day, c_time)

        # if self.steps == 1:
        #     order_info = self.load_model(p_device, device, self.demand, month, day, c_time)
        #     required_mol = order_info['result'].values[0]
        # else:
        order_info = self.load_model(p_device, device, self.p_demand, self.demand, c_time, p_mol_name)
        self.p_demand = self.demand
        required_mol = order_info['amount'].values[0]

        if required_mol > 0 and amount_set_up == 0:
            reward -= 100
        elif required_mol > 0 and amount_set_up>0:
            reward += 10

        if (order_info['mol_name'] == 'MOL_1').any():
            mol_name = 1
        elif (order_info['mol_name'] == 'MOL_2').any():
            mol_name = 2
        elif (order_info['mol_name'] == 'MOL_3').any():
            mol_name = 3
        elif (order_info['mol_name'] == 'MOL_4').any():
            mol_name = 4

        # 현재 셋업된 소자량으로 생산가능한 mol
        possible_mol = amount_set_up * 6

        # 모델 체인지 시간 반영. 만약 STAY라면 setup과 별개로 모델 체인지 발생
        if action == 4:
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

        # month, day, c_time = self.plustime(month, day, c_time)

        # 새로운 세팅으로 버려지는 소자만큼 패널티 발생
        #같은 소자 사용하며 남은 소자가 required_mol 이상인데 한번 더 셋업한 경우
        if p_amount_set_up >= required_mol and p_device == device and p_amount_set_up != amount_set_up:
            reward -= self.abandon_penalty(p_amount_set_up)
            info['abandon_amount'] = p_amount_set_up
        if p_amount_set_up > 0 and p_amount_set_up < required_mol:
            if p_device == device and p_amount_set_up != amount_set_up:
                reward -= self.abandon_penalty(p_amount_set_up)
                info['abandon_amount'] = p_amount_set_up
            elif p_device != device:
                reward -= self.abandon_penalty(p_amount_set_up)
                info['abandon_amount'] = p_amount_set_up
        # if p_amount_set_up > 0:
        #     if p_device == device and p_amount_set_up <= amount_set_up:
        #         reward -= self.abandon_penalty(p_amount_set_up)
        #         info['abandon_amount'] = p_amount_set_up
        #     elif p_device != device:
        #         reward -= self.abandon_penalty(p_amount_set_up)
        #         info['abandon_amount'] = p_amount_set_up

        # elif p_amount_set_up >= amount_set_up:
        #     reward += 20
        # elif p_amount_set_up == 0 and amount_set_up != 0:
        #     reward += 100
        # elif p_amount_set_up == 0 and amount_set_up == 0:
        #     reward -= 500
        # elif p_amount_set_up < amount_set_up and p_device == device:
        #     reward += 20


        # # 주문양이 남아있는 모델의 소자 선택하도록 유도
        if (self.demand['MOL_1'] == 0).all() == True and (self.demand['MOL_2'] == 0).all() == True:
            if (self.demand['MOL_3'] != 0).any() == True or (self.demand['MOL_4'] != 0).any() == True:
                if device == 0:
                    reward -= self.reward_per_miss
        elif (self.demand['MOL_3'] == 0).all() == True and (self.demand['MOL_4'] == 0).all() == True:
            if (self.demand['MOL_1'] != 0).any() == True or (self.demand['MOL_2'] != 0).any() == True:
                if device == 1:
                    reward -= self.reward_per_miss

        # if required_mol > 0 and amount_set_up == 0:
        #     reward -= 100
        # elif required_mol > 0 and amount_set_up > 0:
        #     reward += 10
        # elif required_mol == 0 and amount_set_up <= 50:
        #     reward += 100

        # 소자 태우는 상태 전이 표현
        if required_mol < possible_mol and required_mol > 0:
            self.running_times.append(required_mol / 6)
            reward += (self.reward_per_success*required_mol)
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
            reward += (self.reward_operation_max)
            reward += (self.reward_per_success * required_mol)
            if (required_mol % 6).any() == 0:
                c_time += required_mol // 6
            else:
                c_time += (required_mol // 6) + 1
            required_mol = 0
            info['success'] = True
            possible_mol = 0
            # amount_set_up = possible_mol / 6
            # reward += self.reward_operation_max
            # reward += self.reward_per_success

        #수정 필요할듯
        elif required_mol > possible_mol and required_mol > 0 and possible_mol > 0:
            self.running_times.append(amount_set_up)
            reward += self.reward_operation_max
            #reward += self.reward_per_success2
            if possible_mol % 6 == 0:
                c_time += possible_mol // 6
            else:
                c_time += (possible_mol // 6) + 1
            required_mol -= possible_mol
            possible_mol = 0
            amount_set_up = possible_mol / 6
            # reward += self.reward_operation_max
            # reward += self.reward_per_success2

        # amount_set_up = possible_mol / 6

        # month, day, c_time = self.plustime(month, day, c_time)

        # 납기를 지키지 못한 것에 대한 패널티
        due_time = order_info['time']
        # datetime_series = pd.to_datetime(due_)
        # due_year = datetime_series.dt.year
        # due_month = datetime_series.dt.month
        # due_day = datetime_series.dt.day
        # processing_date = datetime(year = due_year, month = month, day = day, hour = c_time)
        # due_date = pd.to_datetime(pd.Timestamp(year=due_year, month=due_month, day=due_day, hour=due_time))
        # due_date = datetime_series.apply(lambda x: pd.Timestamp(year=x.year, month=x.month, day=x.day, hour=x.hour))

        # month, day, c_time = self.plustime(month, day, c_time)
        if int(due_time) != 0:
            if (due_time < c_time).any():
                reward -= self.excess_penalty(due_time, c_time)
                info['excess'] = True

        #reward -= self.excess_penalty2(device, c_time, self.demand)

        info['usetime'] = c_time
        # required_mol = order_info['result']


        self.demand = self.update_model(self.demand, order_info, required_mol)

        self.state = (required_mol, mol_name, amount_set_up, device, c_time)
        return np.array(self.state, dtype=object), reward, done, info

    def reset(self):
        self.state = (0, 2, 0, 0,) + (0,)

        self.steps = 0
        self.running_times = []
        self.setup_times = []
        self.model_change_times = []

        return np.array(self.state)

    def render(self):
        pass

    def close(self):
        pass


if __name__ == '__main__':
    action_name = {0: "Device 1 set-up", 1: "Device 2 set-up", 2: "Device 1 50% set-up",3: "Device 2 50% set-up",4: "Not set-up"}

    order_data = pd.read_csv('C:/Users/user/Desktop/GPS/order.csv')
    cut_data = pd.read_csv('C:/Users/user/Desktop/GPS/cut_yield.csv')
    setup_time = 28
    device_max_1 = 150
    device_max_2 = 130
    env = Displapy(order_data=order_data, cut_data=cut_data, setup_time=setup_time, device_max_1=device_max_1,
                   device_max_2=device_max_2)

    model = PPO2('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=int(1.0e4))

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

    order_data = pd.read_csv('C:/Users/user/Desktop/GPS/order.csv')
    env = Displapy(order_data=order_data, cut_data=cut_data, setup_time=setup_time, device_max_1=device_max_1,
                   device_max_2=device_max_2)

    # Enjoy trained agent
    obs = env.reset()
    count_abandon = 0
    count_success = 0
    count_excess = 0
    cumul_reward = 0
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
        total_time = info['usetime']
        print("action: ", action_name[action])
        # print("reward this step: ", rewards)
        # print("total reward: ", cumul_reward)
        print("reward this step: ", "{:.1f}".format(float(rewards)))
        print("total reward: ", "{:.1f}".format(float(cumul_reward)))
        print("=" * 50)
        if dones:
            break
            break

    print("Number of successful processing: ", count_success)
    print("Total abandon amount", count_abandon)
    print("Total excess count", count_excess)
    print('Total time required: ', "{:.1f}".format(float(total_time / 24)), 'days')
