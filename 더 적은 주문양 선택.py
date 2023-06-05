import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta, time, date




def load_model(p_device, device, p_demand, demand, current_time, p_mol_name):
    if p_device != device:  # 이전 소자와 다른 소자 셋업
        if device == 0:  # 소자 0사용
            for i in range(len(demand)):
                if demand.loc[i, 'MOL_1'] != 0:
                    first_nonzero_1 = i
                    break
            for j in range(len(demand)):
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
                        amount = demand.iloc[first_nonzero_1, 1]  # 동일 모델 연속 생산
                        mol_name = 'MOL_1'
                        time = demand.loc[first_nonzero_1, 'time']
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
                        amount = demand.iloc[first_nonzero_2, 2]  # 동일 모델 연속 생산
                        mol_name = 'MOL_2'
                        time = demand.loc[first_nonzero_2, 'time']
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
                # for k in range(len(p_demand)):
                #    if p_demand.loc[k,'MOL_3'] != 0:
                #        first_nonzero_3_p = k
                #        break
                # for l in range(len(p_demand)):
                #    if p_demand.loc[l,'MOL_4'] != 0:
                #        first_nonzero_4_p = l
                #        break

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
                        amount = demand.iloc[first_nonzero_3, 3]  # 동일 모델 연속 생산
                        mol_name = 'MOL_3'
                        time = demand.loc[first_nonzero_3, 'time']
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
                        amount = demand.iloc[first_nonzero_4, 4]  # 동일 모델 연속 생산
                        mol_name = 'MOL_4'
                        time = demand.loc[first_nonzero_4, 'time']
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