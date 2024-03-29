# hotel_predict
## 目的
依照GA助理提供的Google Analytics流量統計資料，預測訪客是否會訂房。

## 要求
1.預測訪客是否訂房：
>1.1 請依照流量統計資料train.csv，訓練一個分類模型或回歸模型，預測test.csv中每位訪客是否會訂房消費。

>1.2 依照sample_submission.csv的格式提交預測結果。
 
2.藉由Model A與Model B之預測結果，可得到二個混亂矩陣，如何評估哪個較適合用於本案例。
![](https://github.com/midnightla0710/hotel_predict/blob/main/data-question/%E6%B7%B7%E6%B7%86%E7%9F%A9%E9%99%A3.jpg)

## 預測訪客是否訂房
### predit.py: (predit.ipynb與前者相同，僅為jupyter notebook檔案格式)
1.讀取CSV檔，將資料放入dataframe

2.train資料集preprocessing
>2.1觀察train之遺漏值為類別變數，且資料筆數佔比遠小於1%，故刪除含有NA的樣本

>2.2發現train最小值有負數，刪除不合理的樣本(ex.機率應介於0, 1之間)

>2.3將preprocessing後的資料儲存為train_new.csv

3.分配train資料集，利用StratifiedKFold按比例從每個data set中抽取資料作驗證

4.使用sklearn訓練模型
>4.1預計訓練的模型：Naive Bayes, K Nearest, Random Forest, Decision Tree, XGBoost, Adapative Boost,  Gradient Boost, Ensemble Learning

>4.2模型變數宣告與參數設定

>4.3創建stack_list儲存clf.best_estimator_

>4.4創建train_scores儲存Train Score, Test Score(Accuracy), Precision, Sensitivity, Specificity, F1_score

>4.5顯示train_scores，觀察模型的各項指標，選出表現最佳者

5.test資料集preprocessing
>5.1觀察test未發現遺漏值

>5.2觀察test未發現不合理的樣本(ex.機率應介於0~1之間)

>5.3將preprocessing後的資料儲存為test_new.csv

6.承4.5最終選擇Random Forest模型，預測test.csv中訪客是否會訂房
![](https://github.com/midnightla0710/hotel_predict/blob/main/data-question/predit.jpg)

### predit_produce_new_col.py: 
1.流程大致上如predit.py所述。

2.僅在train與test資料集preprocessing時，選擇部分特徵進行運算以產生新特徵。
>2.1創造3個新欄位(計算每個瀏覽頁面平均瀏覽時間)

>2.2運算時，若除數為0(頁面瀏覽數量為0)，計算結果為NA，代表未實際使用頁面，故將NA值補0

>2.3將preprocessing後的資料儲存為train_new_col.csv與test_new_col.csv

3.觀察模型的各項指標後，發現Gradient Boost模型最佳。但仍劣於predit.py中Random Forest模型表現，故最終採用predit.py預測結果。
![](https://github.com/midnightla0710/hotel_predict/blob/main/data-question/predit_produce_new_col.jpg)

### Submission.csv: 
1.以predit.py中Random Forest模型，預測test.csv中訪客是否會訂房，並輸出預測結果Submission.csv

## 模型選用策略
1.分析過程
![](https://github.com/midnightla0710/hotel_predict/blob/main/data-question/%E8%A9%95%E4%BC%B0%E6%B7%B7%E6%B7%86%E7%9F%A9%E9%99%A3.jpg)

2.結論：若是著重於「所有會訂房的訪客中，到底有多少人能被成功預測出會訂房」，則採用Model A；否則通常採用Model B
