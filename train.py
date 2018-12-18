# encoding: utf-8
# titanic 問題。 通常データ train/ test.csv の使用して検証する。
# ロジスティック回帰、
# 標準化の処理、なし。
#
# 評価
# train : % 
# test  : %

# 途中で使用するため、あらかじめ読み込んでおいてください。
# データ加工・処理・分析モジュール
import numpy as np
import numpy.random as random
import scipy as sp
from pandas import Series, DataFrame
import pandas as pd

# 可視化モジュール
import matplotlib.pyplot as plt
import matplotlib as mpl
# 機械学習モジュール
import sklearn

#
def get_subData(src ):
    sub=src
    sub["Age"] = src["Age"].fillna( src["Age"].median())
    sub = sub.dropna()
    sub["Embark_flg"] = sub["Embarked"].values
    sub["Embark_flg"] = sub["Embark_flg"].map(lambda x: str(x).replace('C', '0') )
    sub["Embark_flg"] = sub["Embark_flg"].map(lambda x: str(x).replace('Q', '1') )
    sub["Embark_flg"] = sub["Embark_flg"].map(lambda x: str(x).replace('S', '2') )
    sub.groupby("Embark_flg").size()
    # convert, num
    sub = sub.assign( Embark_flg=pd.to_numeric( sub.Embark_flg ))
    sub["Sex_flg"] = sub["Sex"].map(lambda x: 0 if x =='female' else 1)
    return sub

# 標準化対応、学習。
# 学習データ
train_data = pd.read_csv("train.csv" )
test_data = pd.read_csv("test.csv" )
print( train_data.shape )
#print( train_data.head() )
#
# 前処理 ,欠損データ 中央値で置き換える
train2  = train_data[["PassengerId","Survived","Sex","Age" , "Embarked" ,"SibSp" ,"Parch" ]]
test2   = test_data[ ["PassengerId","Sex","Age" , "Embarked" ,"SibSp" ,"Parch" ]]

train_sub =get_subData(train2 )
test_sub =get_subData(test2 )
print(train_sub.info() )
print(test_sub.info() )
#quit()

# ロジスティック回帰
from sklearn.linear_model import LogisticRegression

# 説明変数と目的変数
X_train= train_sub[["Sex_flg","Age" , "Embark_flg" ,"SibSp" ,"Parch" ]]
y_train= train_sub['Survived']
X_test = test_sub[["Sex_flg","Age" , "Embark_flg" ,"SibSp" ,"Parch" ]]

# 学習データとテストデータに分ける
print(X_train.shape, y_train.shape )
print(X_test.shape  )
#quit()

# ロジスティック回帰のインスタンス
model = LogisticRegression()

# fit
clf = model.fit(X_train,y_train)

print("train result:",clf.score(X_train,y_train))
#quit()
#
# 予測をしてCSVへ書き出す
pred = model.predict(X_test)
PassengerId = np.array( test_data["PassengerId"]).astype(int)
df = pd.DataFrame(pred, PassengerId, columns=["Survived"])
df.head()

#
df.to_csv("out_res.csv", index_label=["PassengerId"])


