import pandas as pd
import numpy as np
task4 = pd.read_csv("xxxxx", index_col = 0)

X = task4.drop(columns=["class"])
Y = task4["class"]

""" 前處理 """

# 將未知數轉換為nan，轉換資料型態、切分XY
import numpy as np
X.replace('?', np.nan, inplace = True)
X_num = X.iloc[:, 8:16].astype(float)
X_obj = X.drop(columns = X.iloc[:, 8:16].columns)
X = pd.concat([X_num, X_obj], axis = 1)

# 切分訓練/驗證集
from sklearn.model_selection import train_test_split
train_X, val_X, train_Y, val_Y = train_test_split(X, Y, test_size = 0.2, random_state = 50)
num_col = train_X.select_dtypes(exclude="object").columns
obj_col = train_X.select_dtypes(include="object").columns

# 以平均數填補數值型資料、以眾數填補非數值型資料
from sklearn.impute import SimpleImputer
imputer_num = SimpleImputer(strategy="mean")
imputer_obj = SimpleImputer(strategy="most_frequent")

train_X.loc[:, num_col] = imputer_num.fit_transform(train_X.loc[:, num_col])
val_X.loc[:, num_col] = imputer_num.transform(val_X.loc[:, num_col])
train_X.loc[:, obj_col] = imputer_obj.fit_transform(train_X.loc[:, obj_col])
val_X.loc[:, obj_col] = imputer_obj.transform(val_X.loc[:, obj_col])

# 使用one_hot_encoder將類別型資料轉換為數值型
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
obj_train_oh = ohe.fit_transform(train_X.loc[:, obj_col]) # array
obj_val_oh = ohe.transform(val_X.loc[:, obj_col])
train_obj = pd.DataFrame(obj_train_oh, index = train_X.index)
val_obj = pd.DataFrame(obj_val_oh, index = val_X.index)
train_X = pd.concat([train_X.drop(obj_col, axis = 1), train_obj], axis = 1)
val_X = pd.concat([val_X.drop(obj_col, axis = 1), val_obj], axis = 1)
train_X = train_X.astype(float)
val_X = val_X.astype(float)

""" 機器學習 """
# 使用GridSearchCV 抓取特定參數中的最佳參數組合 
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
params = {
    "n_estimators" : [50, 100, 150, 300],
    "max_depth" : [3, 5],
    "learning_rate" : [0.05, 0.1, 0.5],
    "gamma" : [0.02, 0.05]
}

# Y轉換為數字型態
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
train_Y = encoder.fit_transform(train_Y)
val_Y = encoder.transform(val_Y)

model_xgb = XGBClassifier()
model = GridSearchCV(model_xgb, params, cv=10, scoring='accuracy')
model.fit(train_X, train_Y)
Y_pred = model.predict(val_X)
print(model.best_params_) # {'gamma': 0.05, 'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 300}
print(model.best_score_) # 0.7583333

print(f"Accuracy: {accuracy_score(Y_pred, val_Y)}") # Accuracy: 0.7916666

# 依照最佳參數重新建置模型
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
train_Y = encoder.fit_transform(train_Y)
val_Y = encoder.transform(val_Y)

model = XGBClassifier(n_estimators = 300, max_depth = 3, learning_rate = 0.05, gamma = 0.05, early_stopping_rounds = 5)
model.fit(train_X, train_Y, eval_set = [(val_X, val_Y)], verbose = False)

Y_p = model.predict(val_X)
print(f"Accuracy: {accuracy_score(Y_p, val_Y)}") # 0.790625

""" 前處理測試資料、模型預測 """
test = pd.read_csv("xxxxx", index_col = 0)

test.replace('?', np.nan, inplace = True)
test_num = test.iloc[:, 8:16].astype(float)
test_obj = test.drop(columns = test.iloc[:, 8:16].columns)
test = pd.concat([test_num, test_obj], axis = 1)
test_obj_col = test.select_dtypes(include="object").columns
test_num_col = test.select_dtypes(exclude="object").columns
test.loc[:, test_num_col] = imputer_num.transform(test.loc[:, test_num_col])
test.loc[:, test_obj_col] = imputer_obj.transform(test.loc[:, test_obj_col])
obj_test_oh = ohe.transform(test.loc[:, obj_col])
test_obj = pd.DataFrame(obj_test_oh, index = test.index)
test = pd.concat([test.drop(obj_col, axis = 1), test_obj], axis = 1)
test = test.astype(float)

# 使用模型預測測試資料
pred = model.predict(test)
pred = pd.DataFrame(pred)
pred.replace([0, 1, 2, 3, 4, 5], ["C0", "C1", "C2", "C3", "C4", "C5"], inplace = True)
pred.rename(columns = {0 :"class"}, inplace = True)
pred.index.names = ["id"]

# 輸出預測結果
# pred.to_csv("xxxxx")
