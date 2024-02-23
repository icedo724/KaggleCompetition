import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

# 데이터 로드
# https://www.kaggle.com/competitions/playground-series-s4e2
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Competition의 원본 데이터
# https://www.kaggle.com/datasets/aravindpcoder/obesity-or-cvd-risk-classifyregressorcluster
origin = pd.read_csv('origin.csv')

# train에서 id는 필요 없기 때문에 drop
train.drop('id', axis=1, inplace=True)

# train과 원본 데이터 병합
train = pd.concat([train, origin], ignore_index=True)

# 중복 열 제거
train.drop_duplicates(inplace=True)

# 범주형 데이터와 수치형 데이터를 구분 후 각각의 변수에 저장
# train과 test 두 데이터에 대해 각각 적용
num_cols = list(train.select_dtypes(exclude=['object']).columns)
cat_cols = list(train.select_dtypes(include=['object']).columns)

num_cols_test = list(test.select_dtypes(exclude=['object']).columns)
cat_cols_test = list(test.select_dtypes(include=['object']).columns)

# test의 경우 id는 제외하고 저장함
num_cols_test = [col for col in num_cols_test if col not in ['id']]

# StandardScaler를 통해 수치형 데이터를 표준화
scaler = StandardScaler()
train[num_cols] = scaler.fit_transform(train[num_cols])
test[num_cols_test] = scaler.transform(test[num_cols_test])

# LabelEncoder를 통해 범주형 데이터를 정수형으로 인코딩
labelencoder = LabelEncoder()
object_columns = train.select_dtypes(include='object').columns.difference(['NObeyesdad'])

# train 인코딩
for col_name in object_columns:
    if train[col_name].dtypes=='object':
        train[col_name]=labelencoder.fit_transform(train[col_name])

# test 인코딩
for col_name in test.columns:
    if test[col_name].dtypes=='object':
        test[col_name]=labelencoder.fit_transform(test[col_name])

# X에 feature 초기화
X = train.drop(['NObeyesdad'], axis=1)

# y에 target 초기화
y = train['NObeyesdad']

# y에 초기화 된 target을 LabelEncoder를 통해 인코딩
y = labelencoder.fit_transform(y)

# X_test에 test의 feature 초기화
X_test = test.drop(["id"],axis=1)

# 훈련 데이터 분할
X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.2,random_state=42)

# 파라미터 지정(최적 파라미터 탐색 권장)
best_param = {"objective": "multiclass",
    "metric": "multi_logloss",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "random_state": 42,
    "num_class": 7,
    'learning_rate': 0.030962211546832760,
    'n_estimators': 500,
    'lambda_l1': 0.009667446568254372,
    'lambda_l2': 0.04018641437301800,
    'max_depth': 10,
    'colsample_bytree': 0.40977129346872643,
    'subsample': 0.9535797422450176,
    'min_child_samples': 26}

# 모델 초기화 후 파라미터 할당
model_lgb = lgb.LGBMClassifier(**best_param,verbose=100)

# 모델에 훈련 데이터 학습
model_lgb.fit(X_train, y_train)

# 훈련 된 모델을 통해 예측 - 예측값 반환
pred_lgb = model_lgb.predict(X_valid)

# 훈련 된 모델을 통해 예측 - 각 클래스에 속할 확률 반환
pred_proba = model_lgb.predict_proba(X_valid)

# accuracy score 계산
accuracy = accuracy_score(y_valid, pred_lgb)
print("Accuracy Score:", accuracy)

# Optuna를 통해 클래스별 임계값을 최적화

# objective 함수는 입력값의 임계값을 추출하고 thresholds 딕셔너리에 추가
def objective(trial):
    # Define the thresholds for each class
    thresholds = {}
    for i in range(num_classes):
        thresholds[f'threshold_{i}'] = trial.suggest_uniform(f'threshold_{i}', 0.0, 1.0)

    # apply_thresholds에서 변환된 값을 y_pred에 초기화
    y_pred = apply_thresholds(pred_proba, thresholds)

    # y_valid와 y_pred를 통해 accuracy score 계산
    accuracy = accuracy_score(y_valid, y_pred)

    return accuracy

# apply_thresholds 함수는 입력된 임계값을 통해 확률을 예측 레이블로 변환
def apply_thresholds(y_proba, thresholds):
    y_pred_labels = np.argmax(y_proba, axis=1)
    for i in range(y_proba.shape[1]):
        y_pred_labels[y_proba[:, i] > thresholds[f'threshold_{i}']] = i

    return y_pred_labels

# num_classes 정의(best_param에 따름)
num_classes = 7

# Optuna의 Study 객체를 생성
study = optuna.create_study(direction='maximize')

# Objective 함수를 최적화
study.optimize(objective, n_trials=100)

# 최적 임계값을 초기화
best_thresholds = study.best_params

# 최적 임계값 출력
print("Best Thresholds:", best_thresholds)

# X_test에 대한 예측 확률을 test_label에 초기화
test_label = model_lgb.predict_proba(X_test)

# test_label에 초기화된 예측 확률을 apply_thresholds 함수를 통해 예측값으로 변환
test_label = apply_thresholds(test_label, best_thresholds)

# test_label에 초기화된 예측값을 원래대로 디코딩
pred = labelencoder.inverse_transform(test_label)

# 디코딩 된 예측값을 DataFrame 형식으로 변환하고 제출 양식에 맞게 변환
sub_lgb = pd.DataFrame({'id': test.id, 'NObeyesdad': pred})

# 파일 export
sub_lgb.to_csv('sub_lgb_final_3.csv', index=False)
