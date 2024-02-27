import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
from lightgbm import LGBMClassifier
import numpy as np
from xgboost import XGBClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
origin = pd.read_csv('origin.csv')

origin.rename(columns = {'RowNumber':'id'}, inplace=True)

test_id = test['id']

def drop_useless(df):
    df.drop(['id', 'CustomerId', 'Surname'], axis=1, inplace=True)
    return df

drop_useless(train)
drop_useless(origin)
drop_useless(test)

# train과 원본 데이터 병합
train = pd.concat([train, origin], ignore_index=True)

# 중복 열 제거
train.drop_duplicates(inplace=True)

# 각 열의 최빈값(mode)을 구하고 이를 딕셔너리에 저장
mode_values_train = train.mode().iloc[0]

# 각 열의 결측치를 해당 열의 최빈값으로 대체
train.fillna(mode_values_train, inplace=True)

# test에 대해서 동일하게 적용
mode_values_test = test.mode().iloc[0]
test.fillna(mode_values_test, inplace=True)

# 범주형 데이터를 구분 후 각각의 변수에 저장
# train과 test 두 데이터에 대해 각각 적용
num_cols = list(train.select_dtypes(exclude=['object']).columns)

num_cols_test = list(test.select_dtypes(exclude=['object']).columns)

# Encoding이 필요하지 않은 경우 저장하지 않음
num_cols = [col for col in num_cols if col not in ['Exited', 'IsActiveMember', 'HasCrCard']]
num_cols_test = [col for col in num_cols_test if col not in ['Exited', 'IsActiveMember', 'HasCrCard']]

# 두 방법 중 하나 선택 후 비교
# StandardScaler를 통해 수치형 데이터를 표준화
# scaler = StandardScaler()
# train[num_cols] = scaler.fit_transform(train[num_cols])
# test[num_cols_test] = scaler.transform(test[num_cols_test])

# MinMaxScaler 통해 수치형 데이터를 표준화 - 조금 더 높은 성능을 보임
scaler = MinMaxScaler()
train[num_cols] = scaler.fit_transform(train[num_cols])
test[num_cols_test] = scaler.transform(test[num_cols_test])

labelencoder = LabelEncoder()
train['Gender']=labelencoder.fit_transform(train['Gender'])
test['Gender']=labelencoder.fit_transform(test['Gender'])

# Geography OneHotEncoding
train = pd.get_dummies(train, columns=['Geography'])
test = pd.get_dummies(test, columns=['Geography'])

X = train.drop('Exited', axis=1)
y = train['Exited']

# 훈련 데이터 분할
X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.2,random_state=42)

param_grid = {
     'max_depth': [5, 6, 7],
     'learning_rate': [0.1, 0.5, 1],
     'gamma': [1.0, 1.5, 2],
     'reg_lambda': [10.0, 25.0,50.0],
     'scale_pos_weight': [5, 7, 9]
}

model = GridSearchCV(estimator=XGBClassifier(), param_grid=param_grid, scoring='roc_auc', verbose=3, n_jobs=10, cv=3)
model.fit(X_train, y_train)

#best_params = {'gamma': 1.0, 'learning_rate': 0.1, 'max_depth': 5, 'reg_lambda': 25.0, 'scale_pos_weight': 5}

print("Best parameters:", model.best_params_)

best_model = XGBClassifier(**model.best_params_)
best_model.fit(X_train, y_train)

br = best_model.predict_proba(test)

y_pred = best_model.predict(X_valid)
accuracy_score(y_valid, y_pred)

best_result = best_model.predict_proba(test)[:, 1]

sub = pd.DataFrame()
sub['id'] = test_id

sub['Exited'] = best_result

xgb = XGBClassifier()
cb = CatBoostClassifier()
lg = LGBMClassifier()

xgb.fit(X_train, y_train)
cb.fit(X_train, y_train)
lg.fit(X_train, y_train)

y_score_xgb = xgb.predict_proba(X_valid)[:, 1]
y_score_cb = cb.predict_proba(X_valid)[:, 1]
y_score_lg = lg.predict_proba(X_valid)[:, 1]

fpr_xgb, tpr_xgb, _ = roc_curve(y_valid, y_score_xgb)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

fpr_cb, tpr_cb, _ = roc_curve(y_valid, y_score_cb)
roc_auc_cb = auc(fpr_cb, tpr_cb)

fpr_lg, tpr_lg, _ = roc_curve(y_valid, y_score_lg)
roc_auc_lg = auc(fpr_lg, tpr_lg)

plt.figure(figsize=(8, 6))
plt.plot(fpr_xgb, tpr_xgb, color='blue', lw=2, label='XGBoost (AUC = %0.2f)' % roc_auc_xgb)
plt.plot(fpr_cb, tpr_cb, color='red', lw=2, label='CatBoost (AUC = %0.2f)' % roc_auc_cb)
plt.plot(fpr_lg, tpr_lg, color='green', lw=2, label='LightGBM (AUC = %0.2f)' % roc_auc_lg)

plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
