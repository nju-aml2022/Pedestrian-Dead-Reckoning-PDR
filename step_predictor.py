from dataloader import TestCase
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal


def filter(range, data):
    filter = np.ones(range) / range
    return np.convolve(data, filter, mode="same")


def try_model(model: str):

    if model == 'DecisionTree':
        #### 3.1决策树回归####
        from sklearn import tree
        model_output = tree.DecisionTreeRegressor()
    #### 3.2线性回归####
    elif model == 'Linear':
        from sklearn import linear_model
        model_output = linear_model.LinearRegression()
    #### 3.3SVM回归####
    elif model == 'SVR':
        from sklearn import svm
        model_output = svm.SVR()
    #### 3.4KNN回归####
    elif model == 'KNeighbors':
        from sklearn import neighbors
        model_output = neighbors.KNeighborsRegressor()
    #### 3.5随机森林回归####
    elif model == 'RandomForest':
        from sklearn import ensemble
        model_output = ensemble.RandomForestRegressor(
            n_estimators=20)  # 这里使用20个决策树
    #### 3.6Adaboost回归####
    elif model == 'AdaBoost':
        from sklearn import ensemble
        model_output = ensemble.AdaBoostRegressor(
            n_estimators=50)  # 这里使用50个决策树
    #### 3.7GBRT回归####
    elif model == 'GradientBoosting':
        from sklearn import ensemble
        model_output = ensemble.GradientBoostingRegressor(
            n_estimators=100)  # 这里使用100个决策树
    #### 3.8Bagging回归####
    elif model == 'Bagging':
        from sklearn.ensemble import BaggingRegressor
        model_output = BaggingRegressor()
    #### 3.9ExtraTree极端随机树回归####
    elif model == 'ExtraTree':
        from sklearn.tree import ExtraTreeRegressor
        model_output = ExtraTreeRegressor()
    else:
        raise Exception('No such model.')

    return model_output


def step_process_regression(test_data: TestCase, model_str, write=False, distance_frac_step=5):
    model = try_model(model_str)
    x = []
    y = []
    filtered_a = filter(10, test_data.a_mag)
    num_peak_3 = signal.find_peaks(filtered_a, distance=20)
    mean_peak = sum(filtered_a[num_peak_3[0]])/len(num_peak_3[0])
    real_peak = num_peak_3[0][np.where(
        filtered_a[num_peak_3[0]] > mean_peak*0.8)]

    step_index = 0

    # 表示每隔几步计算一次步长、sigma、f

    for i in range(1, len(test_data.x)//distance_frac_step):
        # new_test_case = test_data.slice((i-1)*5, i*5+1)
        # filtered_a = new_test_case.a_mag
        last_step_index = step_index

        while test_data.time[real_peak[step_index]] <= test_data.time_location[i*distance_frac_step]:
            step_index += 1

        distance = 0

        for k in range((i-1)*distance_frac_step, i*distance_frac_step):
            distance += np.sqrt((test_data.y[k+1] - test_data.y[k])
                                ** 2 + (test_data.x[k+1] - test_data.x[k])**2)

        y.append(distance/(step_index - last_step_index))

        f = (step_index - last_step_index) / \
            (test_data.time[real_peak[step_index]] -
             test_data.time[real_peak[last_step_index]])
        # print(f)
        sigma = np.var(
            filtered_a[real_peak[last_step_index]:real_peak[step_index+1]])
        # print(sigma)
        x.append([f, sigma])

    model.fit(x, y)
    # sum_up = 0
    if write == True:
        for i in range(len(y)):
            print(y[i], model.predict([x[i]])[0])
    #     sum_up += y[i][0]

    # print(sum_up/len(y))
    # # print("w值为:",model.coef_)
    # # print("b截距值为:",model.intercept_)
    return model
