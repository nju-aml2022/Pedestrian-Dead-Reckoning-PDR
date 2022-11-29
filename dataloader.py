import os
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from matplotlib import pyplot as plt

class TestCase:

    def __init__(self, test_case_path, is_slice=False) -> None:
        self.test_case_path = test_case_path
        if not is_slice:
            self.load_data_from_csv()
            self.preprocess_data()
            self.generate_data()
        

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

        self.slice_start = 0
        self.slice_end = len(self.time_location)

        # 1. 如果不存在 preprocessed.csv 文件，则进行预处理
        if not os.path.exists(os.path.join(self.test_case_path, "preprocessed.csv")):

            # 2. 通过 time_location 进行 1 : 50 的插值获取 time
            self.time = np.zeros((len(self.time_location) * 50, ))
            for i in range(len(self.time_location) - 1):
                self.time[i * 50: (i + 1) * 50] = np.linspace(self.time_location[i], self.time_location[i + 1] - 0.02, 50)
            i = len(self.time_location) - 1
            self.time[i * 50: (i + 1) * 50] = np.linspace(self.time_location[i], self.time_location[i] + 0.98, 50)

            # 3. 根据 time 使用最近邻插值获取 a, la, gs, m
            self.a = self.nearest_neighbor_interpolation(
                    self.time,
                    self.pd_accelerometer[self.pd_accelerometer.columns[0]],
                    np.array(self.pd_accelerometer[self.pd_accelerometer.columns[1:4]]))
            self.la = self.nearest_neighbor_interpolation(
                    self.time,
                    self.pd_linear_accelererometer[self.pd_linear_accelererometer.columns[0]],
                    np.array(self.pd_linear_accelererometer[self.pd_linear_accelererometer.columns[1:4]]))
            self.gs = self.nearest_neighbor_interpolation(
                    self.time,
                    self.pd_gyroscope[self.pd_gyroscope.columns[0]],
                    np.array(self.pd_gyroscope[self.pd_gyroscope.columns[1:4]]))
            self.m = self.nearest_neighbor_interpolation(
                    self.time,
                    self.pd_magnetometer[self.pd_magnetometer.columns[0]],
                    np.array(self.pd_magnetometer[self.pd_magnetometer.columns[1:4]]))

            # 4. 保存 preprocessed.csv, 每一列分别为 "t", "a", "la", "gs", "m"
            self.preprocessed_data = np.concatenate((self.time.reshape(-1, 1), self.a, self.la, self.gs, self.m), axis=1)
            pd.DataFrame(self.preprocessed_data).to_csv(os.path.join(self.test_case_path, "preprocessed.csv"), index=False, header=[
                "t", "a_x", "a_y", "a_z", "la_x", "la_y", "la_z", "gs_x", "gs_y", "gs_z", "m_x", "m_y", "m_z"])

        # 5. 如果存在 preprocessed.csv 文件，则直接读取
        else:
            self.preprocessed_data = np.array(pd.read_csv(os.path.join(self.test_case_path, "preprocessed.csv")))
            self.time = self.preprocessed_data[:, 0]
            self.a = self.preprocessed_data[:, 1:4]
            self.la = self.preprocessed_data[:, 4:7]
            self.gs = self.preprocessed_data[:, 7:10]
            self.m = self.preprocessed_data[:, 10:13]

        # 8. 前 10% 的 Location_input 的数据
        self.len_input = int(len(self.time_location) * 0.1)
        self.location = np.array(
            self.pd_location_input[self.pd_location_input.columns[1:]][:self.len_input])
        self.latitude = self.location[:, 0]
        self.longitude = self.location[:, 1]
        # 9. 选取前 10% 中最后一个数据作为经纬度原点
        self.origin = (self.latitude[-1], self.longitude[-1])
        # 10. 对 Location 进行相同的处理
        if self.have_location_valid:
            self.location_valid = np.array(
                self.pd_location[self.pd_location.columns[1:]])
        # 11. 对 Location_output 进行相同的处理
        if self.have_location_output:
            self.location_output = np.array(
                self.pd_location_output[self.pd_location_output.columns[1:]])


    def generate_data(self):
        '''
        生成一些其他相关的数据
        '''
        # 为 a, la, gs, m 扩充成 a_x, a_y, a_z, la_x, la_y, la_z, gs_x, gs_y, gs_z, m_x, m_y, m_z
        self.a_x = self.a[:, 0]
        self.a_y = self.a[:, 1]
        self.a_z = self.a[:, 2]
        self.la_x = self.la[:, 0]
        self.la_y = self.la[:, 1]
        self.la_z = self.la[:, 2]
        self.gs_x = self.gs[:, 0]
        self.gs_y = self.gs[:, 1]
        self.gs_z = self.gs[:, 2]
        self.m_x = self.m[:, 0]
        self.m_y = self.m[:, 1]
        self.m_z = self.m[:, 2]
        self.a_mag = self.magnitude(self.a)
        self.la_mag = self.magnitude(self.la)
        self.gs_mag = self.magnitude(self.gs)
        self.m_mag = self.magnitude(self.m)
        
        # 7. 通过 a - la 算出它自带的 g
        self.g = self.a - self.la
        self.g_x = self.g[:, 0]
        self.g_y = self.g[:, 1]
        self.g_z = self.g[:, 2]
        self.g_mag = self.magnitude(self.g)

        # 处理 Location
        self.latitude = self.location[:, 0]
        self.longitude = self.location[:, 1]
        self.height = self.location[:, 2]
        self.velocity = self.location[:, 3]
        self.direction = self.location[:, 4]
        self.horizontal_accuracy = self.location[:, 5]
        self.vertical_accuracy = self.location[:, 6]
        # 对经纬度进行处理: 减去原点后乘以 K
        self.K = 1e5
        self.x = (self.latitude - self.origin[0]) * self.K
        self.y = (self.longitude - self.origin[1]) * self.K

        if self.have_location_valid:
            self.latitude_valid = self.location_valid[:, 0]
            self.longitude_valid = self.location_valid[:, 1]
            self.height_valid = self.location_valid[:, 2]
            self.velocity_valid = self.location_valid[:, 3]
            self.direction_valid = self.location_valid[:, 4]
            self.horizontal_accuracy_valid = self.location_valid[:, 5]
            self.vertical_accuracy_valid = self.location_valid[:, 6]
            self.x_valid = (self.latitude_valid - self.origin[0]) * self.K
            self.y_valid = (self.longitude_valid - self.origin[1]) * self.K
        
        if self.have_location_output:
            self.latitude_output = self.location_output[:, 0]
            self.longitude_output = self.location_output[:, 1]
            self.height_output = self.location_output[:, 2]
            self.velocity_output = self.location_output[:, 3]
            self.direction_output = self.location_output[:, 4]
            self.horizontal_accuracy_output = self.location_output[:, 5]
            self.vertical_accuracy_output = self.location_output[:, 6]
            self.x_output = (self.latitude_output - self.origin[0]) * self.K
            self.y_output = (self.longitude_output - self.origin[1]) * self.K


    def slice(self, start, end):
        """
        切片，返回一个新的 TestCase 对象
        start 和 end 是与 time_location 对齐的
        """
        if end <= 0:
            end = len(self.time_location) + end
        new_test_case = TestCase(self.test_case_path, is_slice=True)
        new_test_case.slice_start = start
        new_test_case.slice_end = end
        _start, _end = 50 * start, 50 * end
        # 切片
        new_test_case.time_location = self.time_location[start:end]
        new_test_case.time = self.time[_start:_end]
        new_test_case.preprocessed_data = self.preprocessed_data[_start:_end]
        new_test_case.a = self.a[_start:_end]
        new_test_case.la = self.la[_start:_end]
        new_test_case.gs = self.gs[_start:_end]
        new_test_case.m = self.m[_start:_end]
        # 前 10% 的 Location_input 的数据
        start_input = start if start < self.len_input else self.len_input
        end_input = end if end < self.len_input else self.len_input
        new_test_case.origin = self.origin
        new_test_case.location = self.location[start_input:end_input]
        new_test_case.len_input = end_input - start_input
        # 对 Location 进行相同的处理
        new_test_case.have_location_valid = self.have_location_valid
        if new_test_case.have_location_valid:
            new_test_case.location_valid = self.location_valid[start:end]
        # 对 Location_output 进行相同的处理
        new_test_case.have_location_output = self.have_location_output
        if new_test_case.have_location_output:
            new_test_case.location_output = self.location_output[start:end]
        # 重新处理
        new_test_case.generate_data()
        return new_test_case
    

    # 取绝对值
    @staticmethod
    def magnitude(x, y=None, z=None):
        if y is not None and z is not None:
            return np.sqrt(x ** 2 + y ** 2 + z ** 2)
        else:
            return np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2)


    # 画路线图
    def draw_route(self, number_of_arrows=25, draw_type="valid", x=None, y=None, direction=None):
        if x and y and direction:
            _x = x
            _y = y
            _direction = direction
        elif draw_type == "valid" and self.have_location_valid:
            _x = self.x_valid
            _y = self.y_valid
            _direction = self.direction_valid
        elif draw_type == "output" and self.have_location_output:
            _x = self.x_output
            _y = self.y_output
            _direction = self.direction_output
        else:
            _x = self.x
            _y = self.y
            _direction = self.direction
        plt.plot(_x, _y)

        # 每隔 n / number_of_arrows 个点画一个方向箭头
        n = len(_x)
        period = n // number_of_arrows
        head_width = min((max(_x) - min(_x)) / 100, (max(_y) - min(_y)) / 100)
        head_length = head_width
        if period == 0:
            print("Error: too many arrows or too few points.")
            return
        for i in range(0, n - period, period):
            length = ((_x[i + period] - _x[i]) ** 2 + (_y[i + period] - _y[i]) ** 2) ** 0.5
            # deg2rag
            angle = _direction[i] * np.pi / 180
            dx, dy = length * np.cos(angle), length * np.sin(angle)
            plt.arrow(_x[i], _y[i], dx, dy, head_width=head_width, head_length=head_length, fc='r', ec='r')

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
    new_test_case = test_case.slice(100, 500)
    print(f"{new_test_case.time_location.shape = }")
    print(f"{new_test_case.time.shape = }")
    new_test_case.draw_route()


if __name__ == "__main__":
    unit_test()