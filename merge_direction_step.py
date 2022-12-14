from dataloader import TestCase
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import step_predictor
import direction_predictor
import math


# name 是回归分类器的名字
def merge_dir_step(test_case: TestCase, model_name='ExtraTree', distance_frac_step=5, clean_data=5, optimized_mode_ratio=0.1, butter_Wn=0.005) -> np.ndarray:
    '''
    :param test_case: 测试用例
    :param model_name: 回归分类器的名字, 默认 'ExtraTree' ('DecisionTree', 'Linear', 'SVR', 'RandomForest', 'AdaBoost', 'GradientBoosting', 'Bagging', 'ExtraTree')
    :param distance_frac_step: 用于计算步长的距离分数, 一般取 5 秒
    :param clean_data: 清洗数据的秒数, 默认为 5 秒
    :param optimized_mode_ratio: 方向预测中优化器模式的比例, 默认 0.1
    :param butter_Wn: 滤波器的截止频率, 默认 0.005

    :return: np.ndarray, shape=(n, 3), n 为步数, 3 列分别为 time, x 和 y
    '''

    # 最后输出
    time_list = []
    x_list = []
    y_list = []
    mean_step = 0
    # 去除扰动数据大小

    # 去除扰动数据前先保存
    for i in range(0, clean_data):
        time_list.append(test_case.time_location[i])
        x_list.append(test_case.x[i])
        y_list.append(test_case.y[i])

    test_case = test_case.slice(clean_data, 0)
    direction_pred = direction_predictor.predict_direction(test_case, optimized_mode_ratio=optimized_mode_ratio, butter_Wn=butter_Wn)
    if model_name != 'Mean':
        model = step_predictor.step_process_regression(
            test_case, model_name, write=False, distance_frac_step=distance_frac_step)
    else:
        mean_step = step_predictor.step_process_mean(test_case)
    filtered_a = step_predictor.filter(10, test_case.a_mag)
    num_peak_3 = signal.find_peaks(filtered_a, distance=20)
    mean_peak = sum(filtered_a[num_peak_3[0]])/len(num_peak_3[0])
    real_peak = num_peak_3[0][np.where(
        filtered_a[num_peak_3[0]] > mean_peak*0.8)]

    test_begin = len(test_case.x)-1
    step_test_begin = 0
    while test_case.time[real_peak[step_test_begin]] < test_case.time_location[test_begin]:
        step_test_begin += 1

    # 把前百分之十的时间位置先记录下来
    for i in range(0, len(test_case.x)):
        time_list.append(test_case.time_location[i])
        x_list.append(test_case.x[i])
        y_list.append(test_case.y[i])

    for i in range(step_test_begin, len(real_peak)):
        f = 1/(test_case.time[real_peak[i]] - test_case.time[real_peak[i-1]])
        sigma = np.var(filtered_a[real_peak[i-1]:real_peak[i]])
        if model_name != 'Mean':
            step_pred = model.predict([[f, sigma]])[0]
        else:
            step_pred = mean_step


        if i == len(real_peak)-1:
            mean_direction = np.mean(direction_pred[real_peak[i]:-1])
        else:
            mean_direction = np.mean(
                direction_pred[real_peak[i]:real_peak[i+1]])
        dx = step_pred*math.cos(mean_direction*math.pi/180)
        dy = step_pred*math.sin(mean_direction*math.pi/180)

        # dx = step_pred*math.cos(direction_pred[real_peak[i]]*math.pi/180)
        # dy = step_pred*math.sin(direction_pred[real_peak[i]]*math.pi/180)

        time_list.append(test_case.time[real_peak[i]])
        x_list.append(dx + x_list[-1])
        y_list.append(dy + y_list[-1])

    return np.array([time_list, x_list, y_list]).T, direction_pred


def unit_test():
    test_case = TestCase("test_case0")
    # test_case = TestCase("../Dataset-of-Pedestrian-Dead-Reckoning\Bag-Walk\Bag-Walk-08-001")
    # test_case = TestCase("../Dataset-of-Pedestrian-Dead-Reckoning\Bag-Walk\Bag-Walk-08-002")

    args = [
        ('DecisionTree', (3, 3, 1)),
        ('Linear', (3, 3, 2)),
        ('SVR', (3, 3, 3)),
        ('RandomForest', (3, 3, 4)),
        ('AdaBoost', (3, 3, 5)),
        ('GradientBoosting', (3, 3, 6)),
        ('Bagging', (3, 3, 7)),
        ('ExtraTree', (3, 3, 8)),
        ('Mean', (3, 3, 9))
    ]

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    for name, arg in args:

        t_x_y_list, _ = merge_dir_step(test_case, name)
        t_x_y_list = t_x_y_list.T

        plt.subplot(*arg)
        plt.plot(t_x_y_list[1], t_x_y_list[2], label="pred")
        # print(len(x_list))
        if test_case.have_location_valid:
            plt.plot(test_case.x_valid, test_case.y_valid, label="valid")

        plt.title(name)
        plt.legend()

    plt.show()


if __name__ == '__main__':
    unit_test()
