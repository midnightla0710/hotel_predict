import pandas as pd
import numpy as np
# NAIBE BAYES
from sklearn.naive_bayes import GaussianNB
# KNN
from sklearn.neighbors import KNeighborsClassifier
# RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
# LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
# DECISON TREE
from sklearn.tree import DecisionTreeClassifier
# XGBOOST
from xgboost import XGBClassifier
# AdaBoosting Classifier
from sklearn.ensemble import AdaBoostClassifier
# GradientBoosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
# HistGradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import warnings
import time

#顯示所有列
pd.set_option('display.max_columns', None)
#顯示所有行
pd.set_option('display.max_rows', None)
#測試value的顯示長度為100
pd.set_option('max_colwidth',100)

# 計算建立模型的時間(起點)
start = time.time()

#讀取CSV檔，將資料放入dataframe
train = pd.read_csv(r'C:\Users\88691\PycharmProjects\hotel_sales\data-question\train.csv', header=None)
test = pd.read_csv(r'C:\Users\88691\PycharmProjects\hotel_sales\data-question\test.csv', header=None)
warnings.filterwarnings("ignore")

#train資料集preprocessing
#顯示資料集
print(train)
#顯示資料筆數
print(train.shape)
#顯示資料統計資訊
print(train.describe())
#顯示欄位屬性
print(train.dtypes)

#檢查是train否具有遺漏值
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum() / train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data)

# 發現遺漏值欄位為類別變數，且資料筆數佔比遠小於1%，故刪除含有NA的樣本
train_no_na = train.dropna()
print(train_no_na)

# 刪除第0列(去除欄位名稱)
train_new = train_no_na.drop([0])
print(train_new)

#轉換train資料型態為float
train_new = train_new.astype('float')

#顯示資料統計資訊
print(train_new.describe())

#承上，發現最小值有負數，刪除不合理的樣本(機率應介於0~1之間)
train_new[train_new < 0] = np.nan
train_new = train_new.dropna()
print(train_new)

#儲存清洗後之train資料集為CSV檔
train_new.to_csv(r'C:\Users\88691\PycharmProjects\hotel_sales\data-question\train_new.csv', index = False)

#分配train資料集
X_train, X_test, y_train, y_test = train_test_split(train_new.iloc[1:, 1:18], train_new.iloc[1:, 18:],
                                                    test_size=0.30, random_state=101)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

# 利用StratifiedKFold做交叉驗證，相較於KFold，StratifiedKFold會照比例每個data set中抽取資料作驗證
sk_fold = StratifiedKFold(10, shuffle=True, random_state=42)

g_nb = GaussianNB()
knn = KNeighborsClassifier()  # 參數:n_neighbors(鄰居數:預設為5)、weights(權重,預設為uniform)、leaf_size(葉的大小:預設為30)
ran_for = RandomForestClassifier()
# n_estimators:樹的顆數、max_depth:最大深度，剪枝用，超過全部剪掉。
# min_samples_leaf:搭配max_depth使用，一個節點在分枝後每個子節點都必須包含至少min_samples_leaf個訓練樣本
# bootstrap:重新取樣原有Data產生新的Data，取樣的過程是均勻且可以重複取樣
log_reg = LogisticRegression()  #penalty:懲罰函數(預設L2)、Ｃ:正則強度倒數，預設為1.0、solver:解決器(默認='lbfgs')，saga對所有懲罰都可以使用
tree = DecisionTreeClassifier()
xgb = XGBClassifier()  # https://www.itread01.com/content/1536594984.html 參數詳解
ada_boost = AdaBoostClassifier()  # https://ask.hellobi.com/blog/zhangjunhong0428/12405 參數詳解
grad_boost = GradientBoostingClassifier(n_estimators=100)  # https://www.itread01.com/content/1514358146.html 參數詳解
hist_grad_boost = HistGradientBoostingClassifier()  # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html

#訓練模型之參數設定
clf = [("Naive Bayes", g_nb, {}), \
       ("K Nearest", knn, {"n_neighbors": [3, 5, 6, 7, 8, 9, 10], "leaf_size": [25, 30, 35]}), \
       ("Random Forest", ran_for,
        {"n_estimators": [10, 50, 100, 200, 400], "max_depth": [3, 10, 20, 40], "random_state": [99],
         "min_samples_leaf": [5, 10, 20, 40, 50], "bootstrap": [False]}), \
       ("Logistic Regression", log_reg, {"penalty": ['l2'], 'max_iter':[10, 20],"C": [100, 10, 1.0, 0.1, 0.01], "solver": ['saga']}), \
 \
       ("Decision Tree", tree, {}), \
       ("XGBoost", xgb,
        {"n_estimators": [200], "max_depth": [3, 4, 5], "learning_rate": [.01, .1, .2], "subsample": [.8],
         "colsample_bytree": [1], "gamma": [0, 1, 5], "lambda": [.01, .1, 1]}), \
 \
       ("Adapative Boost", ada_boost, {"n_estimators": [100], "learning_rate": [.6, .8, 1]}), \
       ("Gradient Boost", grad_boost, {}), \
 \
       ("Histogram GB", hist_grad_boost,
        {"loss": ["binary_crossentropy"], "min_samples_leaf": [5, 10, 20, 40, 50], "l2_regularization": [0, .1, 1]})]

#創建stack_list儲存模型Train Score與Test Score
stack_list = []
#創建train_scores儲存模型儲存Train Score, Test Score(Accuracy), Precision, Sensitivity, Specificity, F1_score
train_scores = pd.DataFrame(columns = ["Name", "Train Score", "Test Score(Accuracy)", "Precision", "Sensitivity", "Specificity", "F1_score"])

#利用GridSearchCV自動優化參數
i = 0
for name, clf1, param_grid in clf:
    clf = GridSearchCV(clf1, param_grid=param_grid, scoring="accuracy", cv=sk_fold, return_train_score=True, n_jobs=7) #啟用7核心加速運算
    clf.fit(X_train, y_train.values.ravel())
    y_pred = clf.best_estimator_.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("=====================================")
    #儲存Train Score, Test Score(Accuracy), Precision, Sensitivity, Specificity, F1_score到train_scores dataframe中
    train_scores.loc[i] = [name, clf.best_score_, (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1]),
                               (cm[0, 0] / (cm[0, 0] + cm[0, 1])), (cm[0, 0] / (cm[0, 0] + cm[1, 1])),
                               (cm[1, 1] / (cm[1, 1] + cm[0, 1])),
                               ((2 * cm[0, 0]) / (2 * cm[0, 0] + cm[1, 0] + cm[0, 1]))]
    stack_list.append(clf.best_estimator_)
    i = i + 1

est = [("g_nb", stack_list[0]), \
       ("knn", stack_list[1]), \
       ("ran_for", stack_list[2]), \
       ("log_reg", stack_list[3]), \
       ("dec_tree", stack_list[4]), \
       ("XGBoost", stack_list[5]), \
       ("ada_boost", stack_list[6]), \
       ("grad_boost", stack_list[7]), \
       ("hist_grad_boost", stack_list[8])]

#集成學習
sc = StackingClassifier(estimators=est,final_estimator = None,cv=sk_fold,passthrough=False)
sc.fit(X_train,y_train)
y_pred = sc.predict(X_test)
cm1 = confusion_matrix(y_test,y_pred)
y_pred_train = sc.predict(X_train)
cm2 = confusion_matrix(y_train,y_pred_train)
stacking = pd.Series(["Stacking",(cm2[0,0]+cm2[1,1])/(cm2[0,0]+cm2[0,1]+cm2[1,0]+cm2[1,1]), (cm1[0, 0] + cm1[1, 1]) / (cm1[0, 0] + cm1[0, 1] + cm1[1, 0] + cm1[1, 1]),
                           (cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])), (cm1[0, 0] / (cm1[0, 0] + cm1[1, 1])),
                           (cm1[1, 1] / (cm1[1, 1] + cm1[0, 1])), ((2 * cm1[0, 0]) / (2 * cm1[0, 0] + cm1[1, 0] + cm1[0, 1]))],
              index=["Name", "Train Score", "Test Score(Accuracy)", "Precision", "Sensitivity", "Specificity", "F1_score"])
train_scores = train_scores.append(stacking, ignore_index=True)
print(train_scores)

#test資料集preprocessing
#顯示資料集
print(test)
#顯示資料筆數
print(test.shape)
#顯示資料統計資訊
print(test.describe())
#顯示欄位屬性
print(test.dtypes)

#檢查test是否具有遺漏值(未發現NA)
total_test = test.isnull().sum().sort_values(ascending=False)
percent_test = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)
missing_data_test = pd.concat([total_test, percent_test], axis=1, keys=['Total', 'Percent'])
print(missing_data_test)

# 刪除第0列(去除欄位名稱)
test_new = test.drop([0])
print(test_new)

#轉換資料型態為float
test_new = test_new.astype('float')

#顯示數據敘述統計
print(test_new.describe())

#儲存清洗後之test資料集為CSV檔
test_new.to_csv(r'C:\Users\88691\PycharmProjects\hotel_sales\data-question\test_new.csv', index = False)

#選擇最佳模型，並輸出預測結果
#觀察Train Score, Test Score(Accuracy), Precision, Sensitivity, Specificity, F1_score後，選擇Random Forest作為預測模型
X_submit = test_new.iloc[:,1:]
stack_list[2].fit(train_new.iloc[:,1:18],train_new.iloc[:,18:])
y_submit = stack_list[2].predict(X_submit)

y_submit= pd.DataFrame(y_submit)
y_submit.index +=1

# FRAMING OUR SOLUTION
y_submit.columns = ['HasRevenue']
y_submit['ID'] = np.arange(1,y_submit.shape[0]+1)
y_submit = y_submit[['ID', 'HasRevenue']]

y_submit.to_csv(r'C:\Users\88691\PycharmProjects\hotel_sales\Submission.csv', index = False)

# 計算建立模型的時間(終點)
end = time.time()
spend = end - start
hour = spend // 3600
minu = (spend - 3600 * hour) // 60
sec = int(spend - 3600 * hour - 60 * minu)
print(f'一共花費了{hour}小時{minu}分鐘{sec}秒')
