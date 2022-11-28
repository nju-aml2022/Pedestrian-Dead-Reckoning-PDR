import os
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from matplotlib import pyplot as plt

class TestCase:

    def __init__(self, test_case_path) -> None:
        self.test_case_path = test_case_path
        self.load_data_from_csv()
        self.preprocess_data()
        

    def load_data_from_csv(self):
        '''
        从 CSV 文件中加载数据到 self.pd_xxx
        '''
        self.pd_accelerometer = pd.read_csv(os.path.join(self.test_case_path, "Accelerometer.csv"))
        if os.path.exists(os.path.join(self.test_case_path, "Linear Accelerometer.csv")):
            self.pd_linear_accelererometer = pd.read_csv(os.path.join(self.test_case_path, "Linear Accelerometer.csv"))
        else:
            self.pd_linear_accelererometer = pd.read_csv(os.path.join(self.test_case_path, "Linear Acceleration.csv"))
        # 忽略气压计数据
        # self.pd_barometer = pd.read_csv(os.path.join(self.test_case_path, "Barometer.csv"))
        self.pd_gyroscope = pd.read_csv(os.path.join(self.test_case_path, "Gyroscope.csv"))
        self.pd_magnetometer = pd.read_csv(os.path.join(self.test_case_path, "Magnetometer.csv"))
        # 加载 Location input
        if os.path.exists(os.path.join(self.test_case_path, "Location_input.csv")):
            self.pd_location_input = pd.read_csv(os.path.join(self.test_case_path, "Location_input.csv"))
        else:
            self.pd_location_input = pd.read_csv(os.path.join(self.test_case_path, "Location.csv"))

        # 存在 Location output 则加载, self.have_location_output 用来判断是否有 Location output
        self.have_location_output = os.path.exists(os.path.join(self.test_case_path, "Location_output.csv"))
        if self.have_location_output:
            self.pd_location_output = pd.read_csv(os.path.join(self.test_case_path, "Location_output.csv"))
        # 存在 Location 则加载, self.have_location_valid 用来判断是否有 Location
        self.have_location_valid = os.path.exists(os.path.join(self.test_case_path, "Location.csv"))
        if self.have_location_valid:
            self.pd_location = pd.read_csv(os.path.join(self.test_case_path, "Location.csv"))



    @staticmethod
    def nearest_neighbor_interpolation(time, time_data, data):
        '''
        使用最近邻插值获取新的 data_interp
        '''
        data_interp = []
        # 当前下标 i
        i = 0
        for t in time:
            while i < len(time_data) - 1 and t >= time_data[i + 1]:
                i += 1
            data_interp.append(data[i])
        return np.array(data_interp)

    
    def preprocess_data(self):
        '''
        对数据进行预处理
        '''
        # 0. Location_input 对应的 time_location
        self.time_location = np.array(self.pd_location_input[self.pd_location_input.columns[0]])

        # 1. 如果不存在 preprocessed.csv 文件，则进行预处理
        if not os.path.exists(os.path.join(self.test_case_path, "preprocessed.csv")):

            # 2. 通过 time_location 进行 1 : 50 的插值获取 time
            self.time = np.zeros((len(self.time_location) * 50, ))
            for i in range(len(self.time_location) - 1):
                self.time[i * 50: (i + 1) * 50] = np.linspace(self.time_location[i], self.time_location[i + 1] - 0.02, 50)
            i = len(self.time_location) - 1
            self.time[i * 50: (i + 1) * 50] = np.linspace(self.time_location[i], self.time_location[i] + 0.98, 50)

            # 3. 根据 time 使用最近邻插值获取 a, la, gs, m
            self._a = self.nearest_neighbor_interpolation(
                    self.time,
                    self.pd_accelerometer[self.pd_accelerometer.columns[0]],
                    np.array(self.pd_accelerometer[self.pd_accelerometer.columns[1:4]]))
            self._la = self.nearest_neighbor_interpolation(
                    self.time,
                    self.pd_linear_accelererometer[self.pd_linear_accelererometer.columns[0]],
                    np.array(self.pd_linear_accelererometer[self.pd_linear_accelererometer.columns[1:4]]))
            self._gs = self.nearest_neighbor_interpolation(
                    self.time,
                    self.pd_gyroscope[self.pd_gyroscope.columns[0]],
                    np.array(self.pd_gyroscope[self.pd_gyroscope.columns[1:4]]))
            self._m = self.nearest_neighbor_interpolation(
                    self.time,
                    self.pd_magnetometer[self.pd_magnetometer.columns[0]],
                    np.array(self.pd_magnetometer[self.pd_magnetometer.columns[1:4]]))

            # 4. 保存 preprocessed.csv, 每一列分别为 "t", "a", "la", "gs", "m"
            self.preprocessed_data = np.concatenate((self.time.reshape(-1, 1), self._a, self._la, self._gs, self._m), axis=1)
            pd.DataFrame(self.preprocessed_data).to_csv(os.path.join(self.test_case_path, "preprocessed.csv"), index=False, header=[
                "t", "a_x", "a_y", "a_z", "la_x", "la_y", "la_z", "gs_x", "gs_y", "gs_z", "m_x", "m_y", "m_z"])

        # 5. 如果存在 preprocessed.csv 文件，则直接读取
        else:
            self.preprocessed_data = np.array(pd.read_csv(os.path.join(self.test_case_path, "preprocessed.csv")))
            self.time = self.preprocessed_data[:, 0]
            self._a = self.preprocessed_data[:, 1:4]
            self._la = self.preprocessed_data[:, 4:7]
            self._gs = self.preprocessed_data[:, 7:10]
            self._m = self.preprocessed_data[:, 10:13]

        # 6. 为 a, la, gs, m 扩充成 a_x, a_y, a_z, la_x, la_y, la_z, gs_x, gs_y, gs_z, m_x, m_y, m_z
        self._a_x = self._a[:, 0]
        self._a_y = self._a[:, 1]
        self._a_z = self._a[:, 2]
        self._la_x = self._la[:, 0]
        self._la_y = self._la[:, 1]
        self._la_z = self._la[:, 2]
        self._gs_x = self._gs[:, 0]
        self._gs_y = self._gs[:, 1]
        self._gs_z = self._gs[:, 2]
        self._m_x = self._m[:, 0]
        self._m_y = self._m[:, 1]
        self._m_z = self._m[:, 2]
        
        # 7. 通过 a - la 算出它自带的 g
        self._g = self._a - self._la
        self._g_x = self._g[:, 0]
        self._g_y = self._g[:, 1]
        self._g_z = self._g[:, 2]

        # 8. 前 10% 的 Location_input 的数据
        self._len_input = int(len(self.time_location) * 0.1)
        self._location = np.array(
            self.pd_location_input[self.pd_location_input.columns[1:]][:self._len_input])
        self._latitude = self._location[:, 0]
        self._longitude = self._location[:, 1]
        self._height = self._location[:, 2]
        self._velocity = self._location[:, 3]
        self._direction = self._location[:, 4]
        self._horizontal_accuracy = self._location[:, 5]
        self._vertical_accuracy = self._location[:, 6]

        # 9. 对经纬度进行处理: 减去原点后乘以 K
        #    选取前 10% 中最后一个数据作为经纬度原点
        self._origin = (self._latitude[-1], self._longitude[-1])
        self._K = 1e5
        self._x = (self._latitude - self._origin[0]) * self._K
        self._y = (self._longitude - self._origin[1]) * self._K

        # 10. 对 Location 进行相同的处理
        if self.have_location_valid:
            self._location_valid = np.array(
                self.pd_location[self.pd_location.columns[1:]])
            self._latitude_valid = self._location_valid[:, 0]
            self._longitude_valid = self._location_valid[:, 1]
            self._height_valid = self._location_valid[:, 2]
            self._velocity_valid = self._location_valid[:, 3]
            self._direction_valid = self._location_valid[:, 4]
            self._horizontal_accuracy_valid = self._location_valid[:, 5]
            self._vertical_accuracy_valid = self._location_valid[:, 6]
            self._x_valid = (self._latitude_valid - self._origin[0]) * self._K
            self._y_valid = (self._longitude_valid - self._origin[1]) * self._K
        
        # 11. 对 Location_output 进行相同的处理
        if self.have_location_output:
            self._location_output = np.array(
                self.pd_location_output[self.pd_location_output.columns[1:]])
            self._latitude_output = self._location_output[:, 0]
            self._longitude_output = self._location_output[:, 1]
            self._height_output = self._location_output[:, 2]
            self._velocity_output = self._location_output[:, 3]
            self._direction_output = self._location_output[:, 4]
            self._horizontal_accuracy_output = self._location_output[:, 5]
            self._vertical_accuracy_output = self._location_output[:, 6]
            self._x_output = (self._latitude_output - self._origin[0]) * self._K
            self._y_output = (self._longitude_output - self._origin[1]) * self._K

    
    # 取绝对值
    def magnitude(self, x, y, z):
        return np.sqrt(x ** 2 + y ** 2 + z ** 2)


    # 画路线图
    def draw_route(self, number_of_arrows=25, draw_type="valid", x=None, y=None, direction=None):
        if x and y and direction:
            _x = x
            _y = y
            _direction = direction
        elif draw_type == "valid" and self.have_location_valid:
            _x = self._x_valid
            _y = self._y_valid
            _direction = self._direction_valid
        elif draw_type == "output" and self.have_location_output:
            _x = self._x_output
            _y = self._y_output
            _direction = self._direction_output
        else:
            _x = self._x
            _y = self._y
            _direction = self._direction
        plt.plot(_x, _y)

        # 每隔 n / number_of_arrows 个点画一个方向箭头
        n = len(_x)
        period = n // number_of_arrows
        for i in range(0, n - period, period):
            length = ((_x[i + period] - _x[i]) ** 2 + (_y[i + period] - _y[i]) ** 2) ** 0.5
            # deg2rag
            angle = _direction[i] * np.pi / 180
            dx, dy = length * np.cos(angle), length * np.sin(angle)
            plt.arrow(_x[i], _y[i], dx, dy, head_width=5, head_length=5, fc='r', ec='r')

        plt.xlabel('latitude')
        plt.ylabel('longitude')
        plt.show()

        
    def eval_model(self):
        if not self.have_location_output:
            print("No location output")
            return
        if not self.have_location_valid:
            print("No location valid")
            return
        dist_error = self.get_dist_error(self.pd_location, self.pd_location_output)
        dir_error = self.get_dir_error(self.pd_location, self.pd_location_output)
        print("Distances error: ", dist_error)
        print("Direction error: ", dir_error)


    @staticmethod
    def get_dir_error(gt, pred):
        dir_list = []
        for i in range(int(len(gt) * 0.1), len(gt)):
            dir = min(abs(gt[gt.columns[5]][i] - pred[pred.columns[5]][i]), 360 - abs(gt[gt.columns[5]][i] - pred[pred.columns[5]][i]))
            dir_list.append(dir)
        error = sum(dir_list) / len(dir_list)
        return error


    @staticmethod
    def get_dist_error(gt, pred):
        print("local_error")
        dist_list = []
        for i in range(int(len(gt) * 0.1), len(gt)):
            dist = geodesic((gt[gt.columns[1]][i], gt[gt.columns[2]][i]), (pred[pred.columns[1]][i], pred[pred.columns[2]][i])).meters
            dist_list.append(dist)
        error = sum(dist_list) / len(dist_list)
        return error



def unit_test():
    test_case = TestCase("test_case0")
    test_case.eval_model()
    test_case.draw_route()
    # test_case = TestCase("../Dataset-of-Pedestrian-Dead-Reckoning/Hand-Walk/Hand-Walk-09-001")


if __name__ == "__main__":
    unit_test()