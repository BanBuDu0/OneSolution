import pandas as pd
import numpy as np


def Load(pathlist):

    for i in range(len(pathlist)):
        path = "./data/train/" + pathlist[i]
        if i == 0:
            file_base = pd.read_csv(path,encoding="gbk",skipinitialspace=True)
            df_base = pd.DataFrame(file_base)
            df_base.loc[:, 'flag'] = df_base['flag'].fillna(0)
            #将flag单独提出，等会放在最后一列,第一个参数为行，第二个参数为列
            dfFlag = df_base.iloc[:,8]

            df = df_base.iloc[:,:8]

        elif i == 1 :
            file_konwledge = pd.read_csv(path,encoding='gbk',skipinitialspace=True)
            df_konwledge = pd.DataFrame(file_konwledge)
            df = pd.merge(df,df_konwledge , on='ID')


        elif i == 2:
            file_money = pd.read_csv(path, encoding='gbk', skipinitialspace=True)
            df_money = pd.DataFrame(file_money)
            df_money = df_money.sort_values(by='ID')
            df_money_aveg = pd.DataFrame(columns=df_money.columns)
            j = 0
            # 没三行进行一次计算
            while j < 45150:
                df1 = df_money.iloc[j:j+3,:]
                j += 3
                df2 = pd.DataFrame(columns=df_money.columns)
                #计算平均值
                df2.loc[0] = df1.apply(lambda x: x.mean())
                df_money_aveg = df_money_aveg.append(df2,ignore_index=True)
            df = pd.merge(df, df_money_aveg, on='ID')

        elif i == 3:
            file_year = pd.read_csv(path, encoding='gbk', skipinitialspace=True)
            df_year = pd.DataFrame(file_year)
            df_year = df_year.sort_values(by='ID')
            df_year_aveg = pd.DataFrame(columns=df_year.columns)
            j = 0
            # 没三行进行一次计算
            while j < 45150:
                df1 = df_year.iloc[j:j + 3, :]
                j += 3
                df2 = pd.DataFrame(columns=df_year.columns)
                # 计算平均值
                df2.loc[0] = df1.apply(lambda x: x.mean())
                df_year_aveg = df_year_aveg.append(df2, ignore_index=True)
            df = pd.merge(df, df_year_aveg, on=['ID','year'])

    df = df.join(dfFlag)


    outputpath = './data/train/HuiZong.csv'
    df.to_csv(outputpath, sep=',', index=True, header=True)





if __name__ == "__main__":
    pathlist = ['base_train_sum.csv', 'knowledge_train_sum.csv',
                'money_report_train_sum.csv', 'year_report_train_sum.csv']
    Load(pathlist)

