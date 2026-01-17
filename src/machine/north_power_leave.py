import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime

from sklearn.neighbors import KNeighborsClassifier

from utils.common import translation, save_dict_to_json
from utils.log import Logger
# from utils.common import load_data
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib



class PowerLeaveModel(object):
    def __init__(self, filename):
        # 配置日志记录
        logfile_name = "train_" + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.logfile = Logger('../', logfile_name).get_logger()
        # 获取数据源
        # self.data_source = load_data(filename)

def data_analysis(data):
    # 看数据
    # 去掉EmployeeNumber、Over18、StandardHours这几个无效列
    data.drop(['EmployeeNumber','Over18', 'StandardHours'], axis=1, inplace=True)


    # print(data.info())
    # print(data.describe())
    # print(data.isnull().sum())


    return data


    # print(data.info())

def feature_engineering(data):
    # 特征工程
    pass

def model_train(data):
    # 模型训练
    # 划分特征和目标变量
    x = data.drop('Attrition', axis=1)
    print(x.head(20))
    y = data['Attrition']
    print(y.head(20))
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # 初始化XGBoost模型
    # model = XGBRegressor()
    model = KNeighborsClassifier()
    # 训练模型
    model.fit(x_train, y_train)
    # 保存模型
    joblib.dump(model, '../../model/xgb_model.pth')

    # 模型评估
    # 预测测试集
    model = joblib.load('../../model/xgb_model.pth')
    y_pred = model.predict(x_test)
    # 评估模型
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    pass

if __name__ == '__main__':
    # 日志配置、数据源获取 

    # 数据分析
    data_analysed = data_analysis(pd.read_csv("../../data/train.csv"))
    # data_analysis(pd.read_csv("../../data/test2.csv"))



    # 特征工程

    # 模型训练 评估、保存
    # model_train(data_analysed)

    # # 翻译列名
    # translation(data_analysed)
    # # 保存look_dict到json文件
    # look_dict = {}
    # for col in data_analysed.columns:
    #     print(data_analysed[col].value_counts())
    #     look_dict[col] = data_analysed[col].value_counts().to_dict()
    # save_dict_to_json(look_dict, '../../data/look_dict.json')
    # # 查看look_dict.json文件中的月收入的最大最小值 19999 1009
    # js = json.load(open('../../data/look_dict.json', 'r', encoding='utf-8'))
    # l = []
    # for key, value in js['月收入'].items():
    #     l.append(int(key))
    # print(max(l))       # 19999
    # print(min(l))       # 1009

