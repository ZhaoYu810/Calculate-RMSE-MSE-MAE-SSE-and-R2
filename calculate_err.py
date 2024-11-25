import glob
import os
import pandas as pd
import numpy as np
from bokeh.layouts import column
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import math
import SaveFunc


def text2dict(file_path):
    result_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            key = line.strip()
            value = f.readline().rstrip()

            result_dict[key] = value
    return result_dict

def print_dic_key(input_dic):
    for key in input_dic.keys():
        print(key)

def read_column_to_list(file_path, sheet_name, column_name):
    # 读取Excel文件
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    # 获取指定列的数据并转换为列表
    column_list = df[column_name].tolist()
    return column_list

def fit_function_exp(x, a):
    return np.exp(-a * np.array(x))

def fit_exp(file_path, sheet_name, column_name, start, end, show_bool):
    Q = read_column_to_list(file_path, sheet_name, column_name)
    print(f"Q={Q}")

    Q_stage1 = Q[start:end]
    Q0 = max(Q_stage1)
    print(f"Q0={Q0}")

    Q_stage1 = [i / Q0 for i in Q_stage1 ]
    XQ = [(i + start + 1, value) for i, value in enumerate(Q_stage1)]
    print(XQ)

    x = []
    y = []
    for index, item in enumerate(XQ):
        x.append(XQ[index][0])
        y.append(XQ[index][1])

    popt, pcov = curve_fit(fit_function_exp, x, y, maxfev=1000)
    a = popt[0]
    print(f"拟合参数 a={a}")

    # 生成拟合曲线的 x 值
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = fit_function_exp(x_fit, a)

    if show_bool == 1:
        # 绘制原始数据点和拟合曲线
        plt.scatter(x, y, label='Original Data')
        plt.plot(x_fit, y_fit, color='red', label='Fitted Curve')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()

    return a

def find_excel_by_txt_name(pathA, pathB):
    """
    根据输入的txt文件路径，在指定路径下查找同名的excel文件并返回其路径。

    :param pathA: txt文件的路径。
    :param pathB: 查找excel文件的目标路径。
    :return: 同名excel文件的路径，如果未找到则返回None。
    """
    txt_file_name = os.path.basename(pathA)
    if not txt_file_name.endswith('.txt'):
        raise ValueError("输入的pathA不是txt文件路径，请检查。")

    target_excel_name = txt_file_name[:-4] + '-deal-delete-sum.xlsx'
    for root, dirs, files in os.walk(pathB):
        for file in files:
            if file == target_excel_name:
                return os.path.join(root, file)

    return None

def fit_linear_Hyperbolic(x, y):
    def custom_func(x, a):
        return a * x + 1

    popt, pcov = curve_fit(custom_func, x, y)
    slope = popt[0]

    # 生成拟合直线的 x 值
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = slope * x_fit + 1

    # 绘制原始数据点和拟合直线
    plt.scatter(x, y, label='Original Data')
    plt.plot(x_fit, y_fit, color='red', label='Fitted Line')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    # plt.show()

    return slope

def process_data_and_fit_Hyperbolic (file_path, sheet_name, column_name, start, end, n):
    Q = read_column_to_list(file_path, sheet_name, column_name)

    Q_stage1 = Q[start:end]
    Q0 = max(Q_stage1)

    Q_stage1 = [i / Q0 for i in Q_stage1]
    Q_stage1 = [math.pow(i, -n) for i in Q_stage1]

    XQ = [(i + start + 1, value) for i, value in enumerate(Q_stage1)]

    x = []
    y = []
    for index, item in enumerate(XQ):
        x.append(XQ[index][0])
        y.append(XQ[index][1])

    slope = fit_linear_Hyperbolic(x, y)
    ai = slope / n

    print(f"斜率: {slope}, ai: {ai}, 截距固定为1")

    return slope, ai

def fit_linear_Harmonic(x, y):
    def custom_func(x, a):
        return a * x + 1

    popt, pcov = curve_fit(custom_func, x, y)
    slope = popt[0]

    # 生成拟合直线的 x 值
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = slope * x_fit + 1

    # 绘制原始数据点和拟合直线
    plt.scatter(x, y, label='Original Data')
    plt.plot(x_fit, y_fit, color='red', label='Fitted Line')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    # plt.show()

    return slope

def process_data_and_fit_Harmonic (file_path, sheet_name, column_name, start, end):
    Q = read_column_to_list(file_path, sheet_name, column_name)

    Q_stage1 = Q[start:end]
    Q0 = max(Q_stage1)

    Q_stage1 = [Q0 / i for i in Q_stage1]

    XQ = [(i + start + 1, value) for i, value in enumerate(Q_stage1)]

    x = []
    y = []
    for index, item in enumerate(XQ):
        x.append(XQ[index][0])
        y.append(XQ[index][1])

    slope = fit_linear_Harmonic(x, y)
    ai = slope

    print(f"斜率: {slope}, ai: {ai}, 截距固定为1")

    return slope, ai

def mean_squared_error(arr1, arr2):
    values1 = np.array(list(arr1.values()))
    values2 = np.array(list(arr2.values()))
    diff = values1 - values2
    squared_diff = diff**2
    return np.mean(squared_diff)

def crop_list(input_list, target_length):
    if target_length >= len(input_list):
        return input_list
    else:
        return input_list[:target_length]

def r_squared(arr1, arr2):
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)

    y_mean = np.mean(arr2)
    ss_res = np.sum((arr2 - arr1)**2)
    ss_tot = np.sum((arr2 - y_mean)**2)

    return 1 - (ss_res / ss_tot)

def calculate_sse(arr1, arr2):
    if len(arr1)!= len(arr2):
        raise ValueError("两个列表的长度不同，无法计算误差指标")

    arr1 = np.array(arr1)
    arr2 = np.array(arr2)

    return np.sum((arr1 - arr2) ** 2)

def calculate_mae(arr1, arr2):
    if len(arr1)!= len(arr2):
        raise ValueError("两个列表的长度不同，无法计算误差指标")

    arr1 = np.array(arr1)
    arr2 = np.array(arr2)

    return np.mean(np.abs(arr1 - arr2))

def calculate_mse(arr1, arr2):
    if len(arr1)!= len(arr2):
        raise ValueError("两个列表的长度不同，无法计算误差指标")

    arr1 = np.array(arr1)
    arr2 = np.array(arr2)

    n = len(arr1)
    sse = np.sum((arr1 - arr2) ** 2)
    return sse / n

def calculate_rmse(arr1, arr2):
    mse = calculate_mse(arr1, arr2)
    return np.sqrt(mse)

if __name__ == "__main__":
    input_path = "D:\\Pycharm\\project\\reformExcelData\\OceanOil\\check\\saveSlope.xlsx"
    sheet_name = "Sheet1"
    column_name_ori = "计算可采储量"
    column_name_deal = "算法可采储量"
    list_length = 9

    ori_data = read_column_to_list(input_path, sheet_name, column_name_ori)
    deal_data = read_column_to_list(input_path, sheet_name, column_name_deal)
    ori_data = crop_list(ori_data, list_length)
    deal_data = crop_list(deal_data, list_length)
    print(type(ori_data), type(deal_data))
    print(f"计算量{ori_data}")
    print(f"算法量{deal_data}")
    # 注意：这篇脚本中的方法是修改后的，输入参数应当是list

    R2 = r_squared(ori_data, deal_data)
    MSE = calculate_mse(ori_data, deal_data)
    SSE = calculate_sse(ori_data, deal_data)
    MAE = calculate_mae(ori_data, deal_data)
    RMSE = calculate_rmse(ori_data, deal_data)

    print(f'SSE: {SSE}')
    print(f'MSE: {MSE}')
    print(f'MAE: {MAE}')
    print(f'RMSE: {RMSE}')
    print(f'R2: {R2}')