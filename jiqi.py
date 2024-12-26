import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV

# 读取训练数据
train_df = pd.read_csv('train_set.csv', sep='\t', nrows=15000)

# 特征提取：使用 TfidfVectorizer，支持 n-grams 和停用词过滤
vectorizer = TfidfVectorizer(
    max_features=5000,  # 增加特征数量
    ngram_range=(1, 2),  # 支持一元词和二元词
    stop_words='english'  # 去除常见无意义单词
)
train_test = vectorizer.fit_transform(train_df['text'])

# 数据划分
X_train, X_val, y_train, y_val = train_test[:10000], train_test[10000:], train_df['label'].values[:10000], train_df['label'].values[10000:]

# 调参：使用 GridSearchCV 寻找最佳参数
param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0]}  # 调整正则化强度
grid_search = GridSearchCV(RidgeClassifier(), param_grid, scoring='f1_macro', cv=5)
grid_search.fit(X_train, y_train)

# 使用最佳参数训练模型
best_alpha = grid_search.best_params_['alpha']
clf = RidgeClassifier(alpha=best_alpha)
clf.fit(X_train, y_train)

# 验证
val_pred = clf.predict(X_val)
f1 = f1_score(y_val, val_pred, average='macro')
print(f"F1 Score on Validation Set: {f1:.4f}")

# 读取测试数据
test_df = pd.read_csv('test_a.csv', sep='\t', nrows=50000)
test = vectorizer.transform(test_df['text'])  # 使用相同的 vectorizer

# 预测
test_predictions = clf.predict(test)

# 保存预测结果
predictions_df = pd.DataFrame(test_predictions, columns=['label'])
predictions_df.to_csv('jiqi2.csv', index=False)

print("预测完成，结果已保存至 jiqi2.csv")