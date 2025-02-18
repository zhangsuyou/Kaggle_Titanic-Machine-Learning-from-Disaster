import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# 1. 数据加载
train = pd.read_csv('d:/Visual Studio Code/code/kaggle/Titanic/train.csv')
test = pd.read_csv('d:/Visual Studio Code/code/kaggle/Titanic/test.csv')

# 2. 特征工程
def feature_engineering(df):
    # 提取称呼
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 
                                       'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # 家庭特征
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
    
    # 票价分箱
    df['FareBin'] = pd.qcut(df['Fare'], 4, labels=False)
    
    # 年龄分箱
    df['AgeBin'] = pd.cut(df['Age'], bins=[0,12,20,40,120], labels=False)
    
    # 客舱特征
    df['HasCabin'] = df['Cabin'].notnull().astype(int)
    
    # 删除原始特征
    return df.drop(['Name', 'Ticket', 'Cabin', 'Age', 'Fare'], axis=1)

train = feature_engineering(train)
test = feature_engineering(test)

# 3. 预处理管道
numeric_features = ['SibSp', 'Parch', 'FamilySize']
categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'FareBin', 'AgeBin', 'HasCabin']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# 4. 模型构建
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 5. 参数网格搜索
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 5, 10],
    'classifier__min_samples_split': [2, 5]
}

X = train.drop(['Survived', 'PassengerId'], axis=1)
y = train['Survived']

# 划分验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 网格搜索
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# 最佳模型
best_model = grid_search.best_estimator_

# 验证集评估
val_pred = best_model.predict(X_val)
print(f"Best params: {grid_search.best_params_}")
print(f"Validation Accuracy: {accuracy_score(y_val, val_pred):.4f}")

# 6. 预测测试集
test_pred = best_model.predict(test.drop('PassengerId', axis=1))

# 7. 生成提交文件
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': test_pred
})
submission.to_csv('submission.csv', index=False)