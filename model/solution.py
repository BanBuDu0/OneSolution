# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn import svm, metrics
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# train_base = pd.read_csv("./data/train/base_train_sum.csv", encoding="gbk")
# train_knowledge = pd.read_csv("./data/train/knowledge_train_sum.csv", encoding="gbk")
# train_money = pd.read_csv("./data/train/money_report_train_sum.csv", encoding="gbk")
# train_year = pd.read_csv("./data/train/year_report_train_sum.csv", encoding="gbk")
# verify_base = pd.read_csv("./data/verify/base_verify1.csv", encoding="gbk")
# verify_money = pd.read_csv("./data/verify/money_information_verify1.csv", encoding="gbk")
# verify_knowledge = pd.read_csv("./data/verify/paient_information_verify1.csv", encoding="gbk")
# verify_year = pd.read_csv("./data/verify/year_report_verify1.csv", encoding="gbk")
# test_base = pd.read_csv("./data/test/base_test_sum.csv", encoding="gbk")
# test_money = pd.read_csv("./data/test/money_report_test_sum.csv", encoding="gbk")
# test_knowledge = pd.read_csv("./data/test/knowledge_test_sum.csv", encoding="gbk")
# test_year = pd.read_csv("./data/test/year_report_test_sum.csv", encoding="gbk")

train_base = pd.read_csv("D:\\jupyter_project\\OneSolution\\data\\train\\base_train_sum.csv", encoding="gbk")
train_knowledge = pd.read_csv("D:\\jupyter_project\\OneSolution\\data\\train\\knowledge_train_sum.csv", encoding="gbk")
train_money = pd.read_csv("D:\\jupyter_project\\OneSolution\\data\\train\\money_report_train_sum.csv", encoding="gbk")
train_year = pd.read_csv("D:\\jupyter_project\\OneSolution\\data\\train\\year_report_train_sum.csv", encoding="gbk")
verify_base = pd.read_csv("D:\\jupyter_project\\OneSolution\\data\\verify\\base_verify1.csv", encoding="gbk")
verify_money = pd.read_csv("D:\\jupyter_project\\OneSolution\\data\\verify\\money_information_verify1.csv", encoding="gbk")
verify_knowledge = pd.read_csv("D:\\jupyter_project\\OneSolution\\data\\verify\\paient_information_verify1.csv", encoding="gbk")
verify_year = pd.read_csv("D:\\jupyter_project\\OneSolution\\data\\verify\\year_report_verify1.csv", encoding="gbk")
test_base = pd.read_csv("D:\\jupyter_project\\OneSolution\\data\\test\\base_test_sum.csv", encoding="gbk")
test_money = pd.read_csv("D:\\jupyter_project\\OneSolution\\data\\test\\money_report_test_sum.csv", encoding="gbk")
test_knowledge = pd.read_csv("D:\\jupyter_project\\OneSolution\\data\\test\\knowledge_test_sum.csv", encoding="gbk")
test_year = pd.read_csv("D:\\jupyter_project\\OneSolution\\data\\test\\year_report_test_sum.csv", encoding="gbk")


def merge_base_knowledge():
    df_test = pd.merge(test_knowledge, test_base, on='ID')

    df_train = pd.merge(train_knowledge, train_base, on='ID')
    df_verify = pd.merge(verify_knowledge, verify_base, on='ID')
    # do some preprocess for these two data
    # drop redundant column
    df_verify.drop('控制人ID', axis=1, inplace=True)

    return df_train, df_verify, df_test


def merge_money_year_mean():
    df_test = pd.merge(test_money, test_year, on=['ID', 'year'])

    train_money.loc[:, 'year'] = train_money['year'].fillna(method='bfill')
    train_year.loc[:, 'year'] = train_year['year'].fillna(method='bfill')
    df_train = pd.merge(train_money, train_year, on=['ID', 'year'])

    verify_money.loc[:, 'year'] = verify_money['year'].fillna(method='bfill')
    verify_year.loc[:, 'year'] = verify_year['year'].fillna(method='bfill')
    df_verify = pd.merge(verify_money, verify_year, on=['ID', 'year'])

    # do some preprocess for these two data
    # drop redundant column
    for column in list(df_train.columns[df_train.isnull().sum() > 0]):
        mean_val = df_train[column].mean()
        df_train[column].fillna(mean_val, inplace=True)
    for column in list(df_verify.columns[df_verify.isnull().sum() > 0]):
        mean_val = df_verify[column].mean()
        df_verify[column].fillna(mean_val, inplace=True)
    df_train.sort_values('ID', inplace=True)
    df_verify.sort_values('ID', inplace=True)

    df_test.sort_values('ID', inplace=True)

    df_train_mean = df_train.groupby('ID').mean().reset_index()
    df_verify_mean = df_verify.groupby('ID').mean().reset_index()
    df_test_mean = df_test.groupby('ID').mean().reset_index()
    return df_train_mean, df_verify_mean, df_test_mean


if __name__ == '__main__':
    print('--------> Data Preprocess')
    train_base_knowledge, verify_base_knowledge, test_base_knowledge = merge_base_knowledge()
    train_mean, verify_mean, test_mean = merge_money_year_mean()

    # merge all data
    df_train = pd.merge(train_mean, train_base_knowledge, on='ID')
    df_verify = pd.merge(verify_mean, verify_base_knowledge, on='ID')
    df_test = pd.merge(test_mean, test_base_knowledge, on='ID')

    values1 = {'注册时间': int(df_train['注册时间'].mean()),
               '注册资本': int(df_train['注册资本'].mean()),
               '控制人持股比例': df_train['控制人持股比例'].mean(),
               '行业': 'other',
               '区域': 'other',
               '企业类型': 'other',
               '控制人类型': 'other',
               '专利': 0,
               '商标': 0,
               '著作权': 0, }
    values2 = {'注册时间': int(df_verify['注册时间'].mean()),
               '注册资本': int(df_verify['注册资本'].mean()),
               '控制人持股比例': df_verify['控制人持股比例'].mean(),
               '行业': 'other',
               '区域': 'other',
               '企业类型': 'other',
               '控制人类型': 'other',
               '专利': 0,
               '商标': 0,
               '著作权': 0, }
    df_train.fillna(value=values1, inplace=True)
    df_verify.fillna(value=values2, inplace=True)

    # drop flag na
    df_verify.dropna(inplace=True)

    # deal with chinese data
    print('--------> Handling Chinese Data Problems')
    encoder = OneHotEncoder(sparse=False)
    X_train = df_train.to_numpy()
    X_verify = df_verify.to_numpy()
    X_test = df_test.to_numpy()

    # split label
    Y_train = X_train[:, -1]
    X_train = np.delete(X_train, -1, axis=1)
    Y_verify = X_verify[:, -1]
    X_verify = np.delete(X_verify, -1, axis=1)

    # get the chinese column
    zh_train = X_train[:, -5:-1]
    X_train = np.hstack((X_train[:, :-5], X_train[:, -1].reshape(-1, 1))).astype('float32')
    zh_verify = X_verify[:, -5:-1]
    X_verify = np.hstack((X_verify[:, :-5], X_verify[:, -1].reshape(-1, 1))).astype('float32')
    zh_test = X_test[:, -5:-1]
    X_test = np.hstack((X_test[:, :-5], X_test[:, -1].reshape(-1, 1))).astype('float32')

    # fit the chinese data
    encoder.fit(zh_train)
    ans_train = encoder.transform(zh_train)
    ans_verify = encoder.transform(zh_verify)
    ans_test = encoder.transform(zh_test)

    # re add these column
    X_train = np.hstack((X_train, ans_train)).astype('float32')
    X_verify = np.hstack((X_verify, ans_verify)).astype('float32')
    X_test = np.hstack((X_test, ans_test)).astype('float32')

    # because the train data do not have all label,so need do clustering first
    print('--------> Clustering on Training Data')
    km = KMeans(n_clusters=2, random_state=0)
    km.fit(X_train)
    for i in range(len(km.labels_)):
        if km.labels_[i] == 1:
            Y_train[i] = 0
        else:
            Y_train[i] = 1

    # do min max scaler
    print('--------> MinMaxScaler Process')
    X_train_m = MinMaxScaler().fit_transform(X_train)
    X_verify_m = MinMaxScaler().fit_transform(X_verify)
    X_test_m = MinMaxScaler().fit_transform(X_test)

    Y_train = Y_train.astype('int')
    Y_verify = Y_verify.astype('int')

    # do pca to find the important feature
    print('--------> PCA Process')
    pca = PCA(n_components=10)
    pca.fit(X_train_m)
    X_train_pca = pca.transform(X_train_m)
    X_verify_pca = pca.transform(X_verify_m)
    test_t1 = pca.transform(X_test_m)

    # use gridSearch to find the best param of SVM
    print('--------> GridSearch Process')
    param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'gamma': ['scale', 'auto']}
    clf = GridSearchCV(svm.SVC(degree=5, max_iter=10000), cv=3, param_grid=param_grid, refit=True, )

    clf.fit(X_train_pca, Y_train)
    predict = clf.predict(X_verify_pca)

    best_parameters = clf.best_estimator_.get_params()
    print('--------> Best Parameter of SVM')
    for para, val in list(best_parameters.items()):
        print(para, val)

    accuracy_rate = metrics.accuracy_score(Y_verify, predict)
    # show the verify accuracy
    print('--------> Verify Accuracy: %s' % accuracy_rate)

    # predict test label
    print('--------> Predict Test Data')
    res = clf.predict(test_t1)

    # write csv
    id_column = X_test[:, 0].astype('int')
    res_df = pd.DataFrame(res, id_column).reset_index()
    res_df.columns = ['企业ID', '分类结果']
    res_df.to_csv('D:\\jupyter_project\\OneSolution\\data\\result.csv', index=False, encoding='utf_8_sig')
    print('--------> Write Result Success')
