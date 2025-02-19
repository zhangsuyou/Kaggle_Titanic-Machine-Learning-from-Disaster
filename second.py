import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
# 选择模型（随机森林）
from sklearn.model_selection import train_test_split, GridSearchCV
# train_test_split 的作用是将原有的训练集分成两个部分，一部分用来训练，一部分用来验证
# GridSearchCV 的作用是用网格法寻找最best的那一组参数
from sklearn.pipeline import Pipeline
# 预处理管道，用来对数据进行预处理的
from sklearn.compose import ColumnTransformer
# 可以将不同的预处理管道进行一个组合
from sklearn.impute import SimpleImputer
# 将缺失部分进行充填
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# 独热编码和标准化放在预处理管道中，都是为了将我们处理好的数据进行进一步的处理，变成模型更加喜欢的数据
from sklearn.metrics import accuracy_score
# 这是一个函数，用来表示在验证集得出答案后的准确率 accuracy_score(y_val, val_pred)

'''
选择随机森林的原因
1、显然对于原始数据而言，有较多的缺失和异常情况出现
   而随机森林对缺失值不敏感，通过多数投票机制，可以容忍部分缺失值
   因其鲁棒性，减少异常值对模型的影响
2、再看一下数据内容，年龄与生存之间呈现非线性关系（儿童存活率显著高于青少年和中年人，而老年人存活率又较低。）
   ，票价与存活关系也呈现分段非线性关系（高票价乘客存活率高，但中等票价存活率骤降，呈现分段非线性。）
   随机森林对处理非线性问题比逻辑回归与SVM方便很多
3、相互交互作用，不同特征之间有不同的影响，如性别与舱位的交互作用或者家庭规模与是否独行的组合影响
   随机森林可以很好的自动组合不同特征，但逻辑回归或单棵决策树可能就麻烦很多
4、高维稀疏   

'''

train = pd.read_csv('d:/Visual Studio Code/code/kaggle/Titanic/train.csv')
test = pd.read_csv('d:/Visual Studio Code/code/kaggle/Titanic/test.csv')

#### 对数据进行第一遍预处理，将一些不同格式、文本等进行一个整合
def feature_engineering(df):
    df['Title'] = df['Name'].str.extract('([A-Za-z]+\.)',expand=False)                  #将头衔提取出来
    df['Title'] = df['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr',
                                       'Major','Rev','Sir','Jonkheer','Dona'],'Rare')   #将头衔中一些不常见的给归到一类
    df['Title'] = df['Title'].replace(['Mlle','Ms'],'Miss')                             #将女士统一
    df['Title'] = df['Title'].replace(['Mme','Mrs'])                                    #男士统一
    
    
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1                                    #将每个人的家庭数量计算出来
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1,'IsAlone'] = 1                                         #通过家庭数量判断是否为一个人
    
    
    df['FareBin'] = pd.qcut(df['Fare'], 4, labels=False)                                #对票价进行分箱操作
    
    
    df['AgeBin'] = pd.cut(df['Age'], bins = [0,12,20,40,120], labels=False)             #对年龄进行分箱操作    
    
    
    df['HasCabin'] = df['Cabin'].notnull().astype(int)                                  #如果仓位信息不空的话，就填充一个有

    
    return df.drop(['Name', 'Ticket', 'Cabin', 'Age', 'Fare'], axis=1)                  #最后返回一个整理好的数据

#### 将原有数据进行上面定义的预处理进行第一遍处理
train = feature_engineering(train)
test = feature_engineering(test)

#### 搭建预处理管道
## 将两种方式处理的信息分开
numeric_features = ['SibSp','Parch','FamilySize']                           #有数据的预处理组                        
categorical_features = ['Pclass','Sex','Embarked','Title','FareBin',        #抽象概念的预处理组
                        'AgeBin','HasCabin']

## 分别搭建管道
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),    
                                      ('scaler', StandardScaler())])                    
'''
有数据的预处理：先将空缺部分用’median’进行填充
选择median的原因是数据中的票价与年龄都存在少量极端例子，所以用中位数的话中位数不受极端值影响
更接近大多数乘客的真实票价。

此外，均值（Mean）：对异常值敏感，导致填充值偏离真实分布。
     众数（Mode）：适用于分类特征，但对数值特征可能不适用（如年龄的众数可能是 24，但大部分年龄分布在20-40之间）。
     固定值（Constant）：如填充 0 或 -1，会引入人为偏差，破坏特征含义。
'''

categorical_transformer = Pipeline(steps=[ ('imputer', SimpleImputer(strategy='most_frequent')),
                                          ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
'''
抽象概念等的预处理：先将空缺部分用’most_frequent’进行填充
选择most frequent的原因是，这些数据大部分是离散的，用均值和中位数没有实际意义
而新增类别会引入噪点，对后面的独热编码产生影响
'''

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

'''
columntransformer的作用是针对数据集中不同类型的特征（如数值型、分类型）分别进行不同的预处理
同时将所有处理后的特征合并为一个统一的特征矩阵，供后续模型直接使用。
'''

####构建模型
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

####网络搜索的参数设置
param_grid = {
    'classifier__n_estimators': [100, 200],                     #会自动组成配对的参数进行测试
    'classifier__max_depth': [None, 5, 10],
    'classifier__min_samples_split': [2, 5]
}

####将原始数据进行划分，方便训练
X = train.drop(['Survived', 'PassengerId'], axis=1)
y = train['Survived']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)         #测试集与验证集的比例为5/1

####网格搜索
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')             #用cv=5去多次训练搜索
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_                        #得出最佳模型

####用之前分出来的验证集进行一个评估
val_pred = best_model.predict(X_val)
print(f"Best params: {grid_search.best_params_}")
print(f"Validation Accuracy: {accuracy_score(y_val, val_pred):.4f}")

####后续测试集，以及更新提交的数据的csv
test_pred = best_model.predict(test.drop('PassengerId', axis=1))

submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': test_pred
})
submission.to_csv('submission.csv', index=False)