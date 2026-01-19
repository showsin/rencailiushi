import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from utils.common import translation, save_dict_to_json
from utils.log import Logger
# from utils.common import load_data
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score
import joblib



# class PowerLeavePredict(object):
#     def __init__(self, filename):
#         # 配置日志记录
#         logfile_name = "train_" + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
#         self.logfile = Logger('../../', logfile_name).get_logger()
#         # 获取数据源
#         # self.data_source = load_data(filename)


def data_analysis(data):
    # 看数据
    # 去掉EmployeeNumber、Over18、StandardHours这几个无效列
    data.drop(['EmployeeNumber', 'Over18', 'StandardHours'], axis=1, inplace=True)

    # print(data.info())
    # print(data.describe())
    # print(data.isnull().sum())

    return data

    # print(data.info())


def feature_engineering(data):
    # 特征工程
    df = data.copy()
    # 员工在每个公司的工作年限占比(跳槽倾向)
    df['all_company_total_years'] = df['NumCompaniesWorked'] / (df['TotalWorkingYears'] + 1e-5)
    # 员工本公司工作年限占比
    df['company_total_years'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1e-5)
    # 员工在当前岗位的工作年限占比
    df['curr_role_total_years'] = df['YearsInCurrentRole'] / (df['TotalWorkingYears'] + 1e-5)
    # 员工当前岗位本公司工作年限占比
    df['curr_role_company_total_years'] = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 1e-5)
    # 员工距上次晋升年限占比
    df['since_last_promotion_total_years'] = df['YearsSinceLastPromotion'] / (df['TotalWorkingYears'] + 1e-5)
    # 晋升频率
    df['promotion_frequency'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1e-5)

    # 出差频率编码
    df['business_travel_encoded'] = df['BusinessTravel'].map(
        {'Travel_Rarely': 2, 'Travel_Frequently': 1, 'Non-Travel': 0}).fillna(0)
    # 加班情况编码
    df['overtime_encoded'] = df['OverTime'].map({'Yes': 1, 'No': 0}).fillna(0)
    # 出差加班负荷
    df['business_overtime_load'] = df['business_travel_encoded'] + df['overtime_encoded']
    # 通勤加班成本
    df['commute_overtime_cost'] = df['DistanceFromHome'] * df['overtime_encoded']

    # 职级薪酬匹配度
    df['job_level_salary_match'] = df['MonthlyIncome'] / (df['JobLevel'] + 1e-5)
    # 工作投入度匹配度
    df['job_involvement_match'] = df['MonthlyIncome'] / (df['JobInvolvement'] + 1e-5)
    # 绩效涨幅匹配度
    df['performance_increase_match'] = df['PercentSalaryHike'] / (df['PerformanceRating'] + 1e-5)
    # 股权激励绑定度
    df['stock_option_level_match'] = (df['StockOptionLevel'] / (df['JobLevel'] + 1e-5)) * df['YearsAtCompany']

    # 整体满意度
    df['avg_overall_satisfaction'] = (df['EnvironmentSatisfaction'] + df['JobSatisfaction'] + df[
        'RelationshipSatisfaction'] + df['WorkLifeBalance']) / 4
    # print(df['avg_overall_satisfaction'].head(20))

    # 培训晋升匹配度
    df['training_promotion_match'] = df['TrainingTimesLastYear'] / (df['YearsSinceLastPromotion'] + 1e-5)

    # 年龄区间编码
    # age_group = df['Age'].map({18: 0, 25: 1, 35: 2, 45: 3, 55: 4, 65: 5})
    # age_group = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 100], labels=[0, 1, 2, 3, 4]).astype(float)
    df['age_group'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 100], labels=[0, 1, 2, 3, 4]).astype(float)
    # 性别编码
    df['gender_encoded'] = df['Gender'].map({'Male': 0, 'Female': 1}).fillna(0)
    # 婚姻状况编码
    df['marital_status_encoded'] = df['MaritalStatus'].map({'Single': 0, 'Married': 1, 'Divorced': 2}).fillna(0)

    # 人员流失（离职状态）编码
    # df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})  # 将 Yes/No 转换为 1/0

    # 工作年限  df['TotalWorkingYears']
    # 年龄    df['Age']
    # 性别    df['Gender']
    # 距离    df['DistanceFromHome']
    # 受教育程度  df['Education']

    # 岗位角色编码
    # job_role_encoded = df['JobRole'].map({
    #     'Sales Executive': 0, 'Research Scientist': 1, 'Laboratory Technician': 2,
    #     'Manufacturing Director': 3, 'Healthcare Representative': 4, 'Manager': 5,
    #     'Sales Representative': 6, 'Research Director': 7, 'Human Resources': 8
    # })
    new_df = pd.concat([
        df['all_company_total_years'],
        df['company_total_years'],
        df['curr_role_company_total_years'],
        df['curr_role_total_years'],
        df['since_last_promotion_total_years'],
        df['promotion_frequency'],
        df['business_travel_encoded'],
        df['overtime_encoded'],
        df['business_overtime_load'],
        df['commute_overtime_cost'],
        df['job_level_salary_match'],
        df['job_involvement_match'],
        df['performance_increase_match'],
        df['stock_option_level_match'],
        df['avg_overall_satisfaction'],
        df['training_promotion_match'],
        df['age_group'],
        df['gender_encoded'],
        df['marital_status_encoded'],
        df['TotalWorkingYears'],
        df['Age'],
        df['DistanceFromHome'],
        df['Education'],
        df['Attrition']
    ], axis=1)

    new_df = new_df.fillna(new_df.mean(numeric_only=True))  # 用数值列的均值填充

    return new_df

def model_pred(data):
    # 模型训练
    # 划分特征和目标变量
    x = data.drop('Attrition', axis=1)
    # x = data
    # print(x.head(20))
    y = data['Attrition']
    # y = target
    # print(y.head(20))
    x = x.astype('float64')
    y = y.astype('int')
    # 划分训练集和测试集
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    # 初始化XGBoost模型
    # model = XGBClassifier(random_state=42,
    #     eval_metric='auc',
    #     n_estimators=100,
    #     scale_pos_weight=None,  # 处理不平衡数据
    #     learning_rate=0.1,
    #     max_depth=6)
    # model = KNeighborsClassifier()
    # 训练模型
    # model.fit(x_train, y_train)
    # 保存模型
    # joblib.dump(model, '../../model/xgb_model.pth')

    # 模型评估
    # 预测测试集
    # transfer = StandardScaler()
    # x = transfer.fit_transform(x)

    model = joblib.load(f'../../model/XGBClassifier_model.pth')
    y_pred_proba = model.predict_proba(x)[:, 1]  # 获取正类别的概率
    y_pred = model.predict(x)
    print(y_pred_proba)
    print(y_pred)

    # 评估模型
    # mse = mean_squared_error(y_test, y_pred)
    # mae = mean_absolute_error(y_test, y_pred)
    # print(f"Mean Squared Error: {mse}")
    # print(f"Mean Absolute Error: {mae}")
    # AUC
    auc = roc_auc_score(y, y_pred_proba)
    print(f"AUC: {auc}")

if __name__ == '__main__':
    # 日志配置、数据源获取

    # 数据分析
    data_analysed = data_analysis(pd.read_csv("../../data/test2.csv"))
    # data_analysis(pd.read_csv("../../data/test2.csv"))

    # 特征工程
    fea_df = feature_engineering(data_analysed)

    # 模型训练 评估、保存
    model_pred(fea_df)