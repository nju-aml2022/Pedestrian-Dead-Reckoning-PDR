import os
import numpy as np
from geopy.distance import geodesic
from matplotlib import pyplot as plt
from dataloader import TestCase
from merge_direction_step import merge_dir_step
from direction_predictor import predict_direction


def linear_interpolation(time, time_data, data):
    '''
    使用线性插值获取新的 data_interp
    '''
    data_interp = []
    # 当前下标 i
    i = 0
    for t in time:
        while i < len(time_data) - 1 and t >= time_data[i + 1]:
            i += 1
        data_interp.append(data[i] + (data[i + 1] - data[i]) / (time_data[i + 1] - time_data[i]) * (t - time_data[i]))
    return np.array(data_interp)


def pdr(test_case: TestCase, model_name='ExtraTree', distance_frac_step=5, clean_data_ratio=0.01, optimized_mode_ratio=0.1, butter_Wn=0.005) -> None:
    '''
    :param test_case: 测试用例
    :param model_name: 回归分类器的名字, 默认 'ExtraTree' ('DecisionTree', 'Linear', 'SVR', 'RandomForest', 'AdaBoost', 'GradientBoosting', 'Bagging', 'ExtraTree')
    :param distance_frac_step: 用于计算步长的距离分数, 一般取 5 秒
    :param clean_data_ratio: 清洗数据的比例 (比如 0.01 表示清洗掉前 1% 的数据)
    :param optimized_mode_ratio: 方向预测中优化器模式的比例, 默认 0.1
    :param butter_Wn: 滤波器的截止频率, 默认 0.005

    最后将结果保存在 test_case 中
    '''
    # 获取合并的步伐
    steps = merge_dir_step(test_case, model_name, distance_frac_step, clean_data_ratio, optimized_mode_ratio, butter_Wn)
    time_step, x_step, y_step = steps[:, 0], steps[:, 1], steps[:, 2]
    # 我们需要需要插值到 time_location
    time_location = test_case.time_location
    # 插值
    x_interp = linear_interpolation(time_location, time_step, x_step)
    y_interp = linear_interpolation(time_location, time_step, y_step)
    # 以及方向
    direction_pred = predict_direction(test_case, optimized_mode_ratio=optimized_mode_ratio, butter_Wn=butter_Wn)
    # 每隔 50 取一次平均得到预测的前 10% 数据 (无初始值偏移量)
    direction_interp = []
    for i in range(0, len(direction_pred), 50):
        direction_interp.append(np.mean(direction_pred[i: i + 50]))
    direction_interp = np.array(direction_interp)
    # 将结果存储到 test_case 中
    test_case.set_location_output(x_interp, y_interp, direction_interp)


def eval_model(test_case: TestCase):
    if not test_case.have_location_output:
        print("No location output")
        return
    if not test_case.have_location_valid:
        print("No location valid")
        return
    dist_error = test_case.get_dist_error()
    dir_error = test_case.get_dir_error()
    dir_ratio = test_case.get_dir_ratio()
    print("Distances error: ", dist_error)
    print("Direction error: ", dir_error)
    print("Direction ratio: ", dir_ratio)


def unit_test():
    test_case = TestCase('test_case0')
    pdr(test_case)
    # 进行了 pdr 之后, 输出数据就会被保存在 test_case 中, 也会输出到 CSV 文件中
    test_case.draw_route()
    test_case.eval_model()


if __name__ == '__main__':
    unit_test()
