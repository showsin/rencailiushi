import json

import pandas as pd


# def load_data(file_path):
#     data = pd.read_csv(file_path)
#     # print(data['Attrition'])
#     # print(data['Attrition'].value_counts())
#     return data

def translation(data):
    employee_field_mapping = {
        "Attrition": "人员流失（离职状态）",
        "Age": "年龄",
        "BusinessTravel": "商务出差频率",
        "Department": "部门",
        "DistanceFromHome": "通勤距离（公里）",
        "Education": "教育程度",
        "EducationField": "专业领域",
        # "EmployeeNumber": "员工编号",
        "EnvironmentSatisfaction": "工作环境满意度",
        "Gender": "性别",
        "JobInvolvement": "工作投入度",
        "JobLevel": "职级",
        "JobRole": "岗位角色",
        "JobSatisfaction": "工作满意度",
        "MaritalStatus": "婚姻状况",
        "MonthlyIncome": "月收入",
        "NumCompaniesWorked": "曾任职公司数量",
        # "Over18": "年满18周岁",
        "OverTime": "加班情况",
        "PercentSalaryHike": "薪资涨幅百分比",
        "PerformanceRating": "绩效评级",
        "RelationshipSatisfaction": "人际关系满意度",
        # "StandardHours": "标准工时",
        "StockOptionLevel": "股权激励等级",
        "TotalWorkingYears": "总工龄",
        "TrainingTimesLastYear": "去年培训次数",
        "WorkLifeBalance": "工作生活平衡度",
        "YearsAtCompany": "在本公司任职年限",
        "YearsInCurrentRole": "现任岗位任职年限",
        "YearsSinceLastPromotion": "距上次晋升年限",
        "YearsWithCurrManager": "与现任主管共事年限"
    }
        # 翻译列名
    data.columns = [employee_field_mapping.get(col, col) for col in data.columns]
    # print(data.head())


def save_dict_to_json(dict, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dict, f, ensure_ascii=False, indent=4)


# if __name__ == '__main__':
    # load_data('../data/train.csv')
    # load_data('../data/test2.csv')

