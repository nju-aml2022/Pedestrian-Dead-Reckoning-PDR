import scipy
import numpy as np
from geopy.distance import geodesic
from scipy import signal
from matplotlib import pyplot as plt
from dataloader import TestCase


def direction_diff(a, b):
    '''
    计算 a 和 b 之间的夹角
    '''
    a = a % 360
    b = b % 360
    abs1 = np.abs(a - b)
    abs2 = 360 - abs1
    return np.where(abs1 < abs2, abs1, abs2)


def direction_mix(a, b, ratio):
    '''
    计算 a 和 b 之间按比例 ratio 混合的方向
    '''
    a = a % 360
    b = b % 360
    return np.where(np.abs(a - b) < 180, a * ratio + b * (1 - ratio), ((a - 360) * ratio + b * (1 - ratio)) % 360)


def predict_direction(tc: TestCase, optimized_mode_ratio=0.4, butter_N=2, butter_Wn=0.005) -> np.ndarray:
    '''
    使用磁力计数据和前 10% 的方向数据预测后续的方向.

    有两种模式, 最优化模式和非最优化模式.
    
    最优化模式会通过优化平均误差来选择最优的 direction0, 缺点是容易过拟合.

    非最优化模式会直接使用第 10% 个数据作为 direction0, 缺点是容易被干扰.
    
    :param tc: 测试用例
    :param optimized_mode_ratio: 最优化模式的比例, 默认为 0.4
    :param butter_N: 滤波器阶数, 默认为 2
    :param butter_Wn: 滤波器截止频率, 默认为 0.005

    :return: 预测的方向 direction_pred, 其中 direction_pred.shape = (50 * n,)
    '''
    # 低通滤波
    b, a = signal.butter(butter_N, butter_Wn, 'lowpass')
    m_x = signal.filtfilt(b, a, tc.m_x)
    m_y = signal.filtfilt(b, a, tc.m_y)
    m_z = signal.filtfilt(b, a, tc.m_z)
    m = np.array([m_x, m_y, m_z])
    m = m.T
    # 对 a 低通滤波得到重力加速度
    g_x = signal.filtfilt(b, a, tc.g_x)
    g_y = signal.filtfilt(b, a, tc.g_y)
    g_z = signal.filtfilt(b, a, tc.g_z)
    g = np.array([g_x, g_y, g_z])
    g = g.T

    # 通过磁场方向和重力加速度叉乘得到东向量
    e = np.cross(m, g)

    # 第 10% 附近的 50 个东向量平均值作为初始东向量
    no_opt_size = len(tc.direction)
    no_opt_e0 = np.mean(e[50 * no_opt_size - 50: 50 * no_opt_size], axis=0)
    # 第 10% 的 direction 作为初始 direction
    no_opt_direction0 = np.mean(tc.direction[-1])
    # 求出所有东向量和初始东向量的角度
    no_opt_angles = np.arccos(np.dot(e, no_opt_e0) / (np.linalg.norm(e, axis=1) * np.linalg.norm(no_opt_e0))) * 180 / np.pi
    # e 和 e0 叉乘后与重力点乘得到符号
    no_opt_signs = - np.sign([np.dot(c, _g) for c, _g in zip(np.cross(e, no_opt_e0), g)])
    # 方向等于角度加上 direction0
    no_opt_direction_pred = no_opt_signs * no_opt_angles + no_opt_direction0
    # 取模 360
    no_opt_direction_pred %= 360

    # 取前 10% 个东向量平均值作为初始东向量
    opt_size = tc.len_input * 50
    opt_e0 = np.mean(e[:opt_size], axis=0)
    # 求出所有东向量和初始东向量的角度
    opt_angles = np.arccos(np.dot(e, opt_e0) / (np.linalg.norm(e, axis=1) * np.linalg.norm(opt_e0))) * 180 / np.pi
    # e 和 e0 叉乘后与重力点乘得到符号
    opt_signs = - np.sign([np.dot(c, _g) for c, _g in zip(np.cross(e, opt_e0), g)])
    # 输入的正确的前 10% 数据
    direction_valid = tc.direction
    # 每隔 50 取一次平均得到预测的前 10% 数据 (无初始值偏移量)
    direction_offset = []
    for i in range(0, opt_size, 50):
        direction_offset.append(np.mean((opt_signs * opt_angles)[i: i + 50]))
    direction_offset = np.array(direction_offset)
    # 平均误差
    error_fn = lambda x: np.mean(direction_diff(direction_valid, (direction_offset + x)))
    # 最小化误差获取最佳初始值
    opt_direction0 = scipy.optimize.minimize(error_fn, 0).x[0]
    # 方向等于角度加上 direction0
    opt_direction_pred = opt_signs * opt_angles + opt_direction0
    # 取模 360
    opt_direction_pred %= 360

    return direction_mix(opt_direction_pred, no_opt_direction_pred, optimized_mode_ratio)


def unit_test():
    direction1 = np.array([10, 20, 30, 40])
    direction2 = np.array([355, 350, 300, 50])
    print(f"{direction_diff(direction1, direction2) = }")
    print(f"{direction_mix(direction1, direction2, 0.5) = }")

    tc = TestCase('test_case0')
    direction_pred = predict_direction(tc, optimized_mode_ratio=0.9, butter_Wn=0.005)
    print(f"{direction_pred.shape = }")

    plt.plot(tc.time, direction_pred)
    if tc.have_location_valid:
        plt.plot(tc.time_location, tc.direction_valid)
    plt.show()


if __name__ == '__main__':
    unit_test()