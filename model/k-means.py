import numpy as np
import random
import math
from sklearn.feature_selection import VarianceThreshold
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer


def Liner_SVM(data_train, data_test, lable_train, lable_test):
    # 用网格搜索法来获取最优的C
    # 搜索最优的C值
    param_grid = {'C': [1, 10, 100], 'kernel': ['linear']}
    clf = GridSearchCV(svm.SVC(degree=5, max_iter=10000), cv=3, param_grid=param_grid, refit=True, )
    clf.fit(data_train, lable_train)
    # 预测数据
    predict = clf.predict(data_test)
    # 生成准确率
    accuracy_rate = metrics.accuracy_score(lable_test, predict)
    print('精度为%s' % accuracy_rate)

def analyse(path_train,path_test):
    file_train = pd.read_csv(path_train,encoding="utf-8", skipinitialspace=True)
    file_test = pd.read_csv(path_test, encoding="utf-8", skipinitialspace=True)
    df_test = pd.DataFrame(file_test)
    df_train = pd.DataFrame(file_train)

    df_train = df_train.dropna()
    df_test = df_test.dropna()
    # 删除整行缺失值

    X_train, Y_train = np.split(np.array(df_train), (-1,), axis=1)
    X_test, Y_test = np.split(np.array(df_test), (-1,), axis=1)
    Y_train = Y_train.astype('int')
    Y_test =Y_test.astype('int')


    list = ['行业','区域', '企业类型','控制人类型']

    trainx = X_train[:,19:]
    testx = X_test[:,19:]

    trainx = Normalizer().fit_transform(trainx)
    testx = Normalizer().fit_transform(testx)

    # # 标准化
    # Scaler_X = preprocessing.MinMaxScaler()
    # VarianceThreshold(threshold=3).fit_transform(iris.data)
    # X_train_scaler = Scaler_X.fit_transform(X_train)
    # X_test_scaler = Scaler_X.transform(X_test)
    #
    le = LabelEncoder()
    enc = OneHotEncoder(handle_unknown = 'ignore')

    #对每一列中文进行赋值
    i=2
    while(i<6):

        le.fit(X_train[:,i].reshape(-1, 1))
        X_train[:,i] = le.transform(X_train[:,i].reshape(-1, 1))
        X_test[:,i] = le.transform(X_test[:,i].reshape(-1, 1))

        enc.fit(X_train[:,i].reshape(-1, 1))
        kkk1 = enc.transform(X_train[:,i].reshape(-1, 1))
        kkk2 = enc.transform(X_test[:,i].reshape(-1, 1))
        kkk1 = sparse.lil_matrix(kkk1)
        kkk2 = sparse.lil_matrix(kkk2)
        trainx = sparse.hstack((trainx, kkk1))
        testx = sparse.hstack((testx,kkk2))
        i +=1

    trainx.toarray()
    testx.toarray()
    print(trainx.shape)
    # data_train = PCA(n_components=2).fit_transform(trainx)
    # df_test = PCA(n_components=2).fit_transform(testx)

    # 用来计算哪几项没区分度，算出来后发现就最后几项有，就在前面直接改了
    # data_train = VarianceThreshold(threshold=0.005).fit_transform(trainx)

    # data_test = VarianceThreshold(threshold=0.005).fit_transform(testx)

    return  trainx,testx,Y_train,Y_test

if __name__ == "__main__":
    path_train = "./HuiZong1.csv"
    path_test = "./HuiZong_test.csv"
    data_train,data_test,Y_train,Y_test = analyse(path_train,path_test)
    Liner_SVM(data_train,data_test,Y_train,Y_test)











