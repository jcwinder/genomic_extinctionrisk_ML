import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from shapely.geometry import Point, Polygon
from scipy.spatial import ConvexHull
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score,
    roc_auc_score
)
from sklearn.inspection import permutation_importance

# Timepoint: at start of bottleneck (100 years after start of simulation)
y0_target = pd.read_csv("/path/to/file/dist_target_310.csv", index_col=0)
y0_features = pd.read_csv("/path/to/file/disty100.csv", index_col=0)
y0_q5_features = y0_features.drop(
    y0_features.filter(regex="q10_|q30|q50|q70|q90").columns,
    axis=1
)
y0_features_p = y0_q5_features[y0_q5_features['na']!=100]
y0_target = y0_target[y0_target['match_col'].isin(y0_features_p['match_col'])]
y0_target = y0_target.drop(['match_col'], axis = 1).astype('int').replace(500,1)
y0_target = y0_target.astype('category').to_numpy()
y0_features_t = y0_features_p.drop(['match_col', 'mu','na', 'nb', 'littersize'], axis = 1)

# Logistic regression
X= y0_features_t
y = np.ravel(y0_target)
olr_mod1 = LogisticRegression(solver="saga", max_iter=4000).fit(X, y)
y_pred = olr_mod1.predict(X)
y_proba = olr_mod1.predict_proba(X)[:, 0]   # probability of positive class
precision = precision_score(y, y_pred, pos_label=0)
recall = recall_score(y, y_pred, pos_label=0)
f1 = f1_score(y, y_pred, pos_label=0)
auc = roc_auc_score(y, y_proba)
accuracy = olr_mod1.score(X, y)

print(f"Logreg y0 - Accuracy:  {accuracy:.3f}")
print(f"Logreg y0 - Precision: {precision:.3f}")
print(f"Logreg y0 - Recall:    {recall:.3f}")
print(f"Logreg y0 - F1 score:  {f1:.3f}")
print(f"Logreg y0 - AUC ROC:   {auc:.3f}")

# Random forest: 
X = y100_features_t.values
y = np.ravel(y100_target)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_index, test_index = next(sss.split(X, y))
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42
    ))
])

param_grid = {
    "clf__n_estimators": [200, 500, 1000], 
    "clf__max_depth": [None, 5, 10, 20],
    "clf__min_samples_split": [2, 5, 10]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
print("RF y100 pb: Best parameters:", grid.best_params_)
print("RF y100 pb: Best CV accuracy:", grid.best_score_)
best_model = grid.best_estimator_
test_acc = best_model.score(X_test, y_test)
print("RF y100 pb: Test set accuracy (best model):", test_acc)

# SVC
X = y0_features_t.values
y = np.ravel(y0_target)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_index, test_index = next(sss.split(X, y))
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index] 
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(kernel='rbf', C=1.0, gamma='scale', class_weight = 'balanced', probability=True ))  # You can tune C and gamma later
])

param_grid = {
    'clf__C': [0.1, 1, 10, 100],
    'clf__gamma': ['scale', 0.01, 0.001] # for example 
}
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print("SVC y0 pb: Best parameters:", grid.best_params_)
print("SVC y0 pb: Best cross-val accuracy:", grid.best_score_)

print("SVC y0 pb: Test set accuracy (best model):", grid.score(X_test, y_test))
best_svc = grid.best_estimator_

y_pred = best_svc.predict(X_test)
y_proba = best_svc.predict_proba(X_test)[:, 0]   # probability of the positive class

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("SVC Confusion Matrix")
plt.show()
plt.savefig(f"/path/to/file/y0_SVC_CM.png", dpi=300)
precision = precision_score(y_test, y_pred, pos_label=0)
recall = recall_score(y_test, y_pred, pos_label=0)
f1 = f1_score(y_test, y_pred, pos_label=0)
auc = roc_auc_score(y_test, y_proba)

print(f"SVC y0 - Precision: {precision:.3f}")
print(f"SVC y0 - Recall:    {recall:.3f}")
print(f"SVC y0 - F1 score:  {f1:.3f}")
print(f"SVC y0 - AUC ROC:   {auc:.3f}")

# Feature importance
feature_names = y0_features_t.columns  
result = permutation_importance(
    best_svc, X_test, y_test, random_state=42, n_jobs=-1
)
importances_df = pd.DataFrame({
    "feature": feature_names,
    "importance": result.importances_mean
}).sort_values("importance", ascending=False)
print(importances_df.head(20))

# LDA
X = y0_features_t.values  # or your NumPy array / DataFrame
y = np.ravel(y0_target)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_index, test_index = next(sss.split(X, y))
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]
lda_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LinearDiscriminantAnalysis())
])
cv_scores = cross_val_score(lda_pipeline, X_train, y_train, cv=5, scoring='accuracy')
print("y0 pb:LDA 5-fold CV accuracy on training set: %.3f ± %.3f" % (cv_scores.mean(), cv_scores.std()))
lda_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LinearDiscriminantAnalysis())
])
cv_scores = cross_val_score(lda_pipeline, X_train, y_train, cv=5, scoring='accuracy')
print("y0pb: LDA 5-fold CV accuracy on training set: %.3f ± %.3f" % (cv_scores.mean(), cv_scores.std()))
lda_pipeline.fit(X_train, y_train)
test_accuracy = lda_pipeline.score(X_test, y_test)
print("y0pb: LDA test set accuracy:", test_accuracy)
lda = LinearDiscriminantAnalysis()
X_lda = lda.fit(X_train, y_train).transform(X_train)

plt.figure(figsize=(8, 4))
plt.hist(X_lda[y_train == 0], bins=30, alpha=0.7, label='0', color='navy')
plt.hist(X_lda[y_train == 1], bins=30, alpha=0.7, label='500', color='turquoise')
plt.axvline(0, color='k', linestyle='--')
plt.xlabel("Linear Discriminant 1")
plt.ylabel("Frequency")
plt.title("LDA projection (1D): class separation")
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig(f"/path/to/file/y0_LDA.png", dpi=300)

y_pred = lda.predict(X_test)
y_proba = lda.predict_proba(X_test)[:, 0]   # probability of the positive class

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("LDA Confusion Matrix")
plt.show()
plt.savefig(f"/path/to/file/y0_LDA_CM.png", dpi=300)
precision = precision_score(y_test, y_pred, pos_label=0)
recall = recall_score(y_test, y_pred, pos_label=0)
f1 = f1_score(y_test, y_pred, pos_label=0)
auc = roc_auc_score(y_test, y_proba)

print(f"y0 LDA - Precision: {precision:.3f}")
print(f"y0 LDA - Recall:    {recall:.3f}")
print(f"y0 LDA - F1 score:  {f1:.3f}")
print(f"y0 LDA - AUC ROC:   {auc:.3f}")

# XGB 
X = y0_features_t.values 
y = np.ravel(y0_target)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_index, test_index = next(sss.split(X, y))
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

cv_scores = cross_val_score(xgb, X_train, y_train, cv=5, scoring='accuracy')
print("XGBoost 5-fold CV accuracy on training set: %.3f ± %.3f" %
     (cv_scores.mean(), cv_scores.std()))
xgb.fit(X_train, y_train)
test_acc = xgb.score(X_test, y_test)
print("Test set accuracy:", test_acc)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}
grid = GridSearchCV(XGBClassifier(eval_metric='logloss', random_state=42),
                    param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
print("XGB y0 pb: Best parameters:", grid.best_params_)
print("XGB y0 pb: Best CV accuracy:", grid.best_score_)
best_model = grid.best_estimator_
test_acc = best_model.score(X_test, y_test)
print("XGB y0 pb: Test set accuracy (best model):", test_acc)
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 0]
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("XGB Confusion Matrix")
plt.show()
plt.savefig(f"/path/to/file/y0_XGB_CM.png", dpi=300)
precision = precision_score(y_test, y_pred, pos_label=0)
recall = recall_score(y_test, y_pred, pos_label=0)
f1 = f1_score(y_test, y_pred, pos_label=0)
auc = roc_auc_score(y_test, y_proba)
print(f"y0 XGB - Precision: {precision:.3f}")
print(f"y0 XGB - Recall:    {recall:.3f}")
print(f"y0 XGB - F1 score:  {f1:.3f}")
print(f"y0 XGB - AUC ROC:   {auc:.3f}")

# Feature importance
feature_names = y0_features_t.columns  
result = permutation_importance(
    best_xgb, X_test, y_test, random_state=42, n_jobs=-1
)
importances_df = pd.DataFrame({
    "feature": feature_names,
    "importance": result.importances_mean
}).sort_values("importance", ascending=False)
print(importances_df.head(20))

# KNN
X = y100_features_t.values
y = np.ravel(y100_target)
ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_index, test_index = next(sss.split(X, y))
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])
param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9, 15],
    "knn__weights": ["uniform", "distance"],
    "knn__p": [1, 2]  # 1 = Manhattan distance, 2 = Euclidean distance
}
grid = GridSearchCV(pipeline,param_grid,cv=5,scoring="accuracy")

grid.fit(X_train, y_train)

print("KNN y100 pb: Best parameters:", grid.best_params_)
print("KNN y100 pb: Best CV accuracy:", grid.best_score_)

best_model = grid.best_estimator_
test_acc = best_model.score(X_test, y_test)

print("KNN y100 pb: Test set accuracy (best model):", test_acc)

# Year 100 + bottleneck population size + fecundity
y0nbfec_target = pd.read_csv("/path/to/file/dist_target_310.csv", index_col=0)
y0nbfec_features = pd.read_csv("/path/to/file/disty100.csv", index_col=0)
y0nbfec_q5_features = y0nbfec_features.drop(
    y0nbfec_features.filter(regex="q10_|q30|q50|q70|q90").columns,
    axis=1
)
y0nbfec_features_p = y0nbfec_q5_features[y0nbfec_q5_features['na']!=100]
y0nbfec_target = y0nbfec_target[y0nbfec_target['match_col'].isin(y0nbfec_features_p['match_col'])]
y0nbfec_target = y0nbfec_target.drop(['match_col'], axis = 1).astype('int').replace(500,1)
y0nbfec_target = y0nbfec_target.astype('category').to_numpy()
y0nbfec_features_t = y0nbfec_features_p.drop(['match_col', 'mu','na'], axis = 1)

# Logistic regression
X = y0nbfec_features_t
y = np.ravel(y0nbfec_target)
olr_mod1 = LogisticRegression(solver="saga", max_iter=4000).fit(X, y)
y_pred = olr_mod1.predict(X)
y_proba = olr_mod1.predict_proba(X)[:, 0]   # probability of positive class
precision = precision_score(y, y_pred, pos_label=0)
recall = recall_score(y, y_pred, pos_label=0)
f1 = f1_score(y, y_pred, pos_label=0)
auc = roc_auc_score(y, y_proba)
accuracy = olr_mod1.score(X, y)
print(f"y0-nbfec Logreg - Accuracy:  {accuracy:.3f}")
print(f"y0-nbfec Logreg - Precision: {precision:.3f}")
print(f"y0-nbfec Logreg - Recall:    {recall:.3f}")
print(f"y0-nbfec Logreg - F1 score:  {f1:.3f}")
print(f"y0-nbfec Logreg - AUC ROC:   {auc:.3f}")

# Random forest: 
X = y0nbfec_features_t.values
y = np.ravel(y0nbfec_target)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_index, test_index = next(sss.split(X, y))
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42
    ))
])

param_grid = {
    "clf__n_estimators": [200, 500, 1000], 
    "clf__max_depth": [None, 5, 10, 20],
    "clf__min_samples_split": [2, 5, 10]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
print("RF y100 pb: Best parameters:", grid.best_params_)
print("RF y100 pb: Best CV accuracy:", grid.best_score_)
best_model = grid.best_estimator_
test_acc = best_model.score(X_test, y_test)
print("RF y100 pb: Test set accuracy (best model):", test_acc)

# SVC:
X = y0nbfec_features_t.values
y = np.ravel(y0nbfec_target)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_index, test_index = next(sss.split(X, y))
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(kernel='rbf', C=1.0, gamma='scale', class_weight = 'balanced', probability=True)) # C and gamma were tuned here
param_grid = {
    'clf__C': [0.1, 1, 10, 100],
    'clf__gamma': ['scale', 0.01, 0.001]
}
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print("SVC y0 pb + nbfec: Best parameters:", grid.best_params_)
print("SVC y0 pb + nbfec: Best cross-val accuracy:", grid.best_score_)
print("SVC y0 pb + nbfec: Test set accuracy (best model):", grid.score(X_test, y_test))
best_svc = grid.best_estimator_
y_pred = best_svc.predict(X_test)
y_proba = best_svc.predict_proba(X_test)[:, 0]
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("SCV Confusion Matrix")
plt.show()
plt.savefig(f"/path/to/file/y0NBFec_SVC_CM.png", dpi=300)
precision = precision_score(y_test, y_pred, pos_label=0)
recall = recall_score(y_test, y_pred, pos_label=0)
f1 = f1_score(y_test, y_pred, pos_label=0)
auc = roc_auc_score(y_test, y_proba)
print(f"y0 Nbfec SVC - Precision: {precision:.3f}")
print(f"y0 Nbfec SVC - Recall:    {recall:.3f}")
print(f"y0 Nbfec SVC - F1 score:  {f1:.3f}")
print(f"y0 Nbfec SVC - AUC ROC:   {auc:.3f}")

# Feature importance
feature_names = y0nbfec_features_t.columns  
result = permutation_importance(
    best_svc, X_test, y_test, random_state=42, n_jobs=-1
)
importances_df = pd.DataFrame({
    "feature": feature_names,
    "importance": result.importances_mean
}).sort_values("importance", ascending=False)
print(importances_df.head(20))


# LDA
X = y0nbfec_features_t.values 
y = np.ravel(y0nbfec_target)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_index, test_index = next(sss.split(X, y))
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]
lda_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LinearDiscriminantAnalysis())
])

cv_scores = cross_val_score(lda_pipeline, X_train, y_train, cv=5, scoring='accuracy')
print("y0 pb + nbfec:LDA 5-fold CV accuracy on training set: %.3f ± %.3f" % (cv_scores.mean(), cv_scores.std()))

# Fit and evaluate on test set
lda_pipeline.fit(X_train, y_train)
test_accuracy = lda_pipeline.score(X_test, y_test)
print("y0 pb + nbfec: LDA test set accuracy:", test_accuracy)
lda = LinearDiscriminantAnalysis()
X_lda = lda.fit(X_train, y_train).transform(X_train)
plt.figure(figsize=(8, 4))
plt.hist(X_lda[y_train == 0], bins=30, alpha=0.7, label='0', color='navy')
plt.hist(X_lda[y_train == 1], bins=30, alpha=0.7, label='500', color='turquoise')
plt.axvline(0, color='k', linestyle='--')
plt.xlabel("Linear Discriminant 1")
plt.ylabel("Frequency")
plt.title("LDA projection (1D): class separation")
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig(f"/path/to/file/y0NBFec_LDA.png", dpi=300)
y_pred = lda.predict(X_test)
y_proba = lda.predict_proba(X_test)[:, 0]
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("LDA Confusion Matrix")
plt.show()
plt.savefig(f"/path/to/file/y0NBFec_LDA_CM.png", dpi=300)

precision = precision_score(y_test, y_pred, pos_label=0)
recall = recall_score(y_test, y_pred, pos_label=0)
f1 = f1_score(y_test, y_pred, pos_label=0)
auc = roc_auc_score(y_test, y_proba)

print(f"y0 Nbfec LDA - Precision: {precision:.3f}")
print(f"y0 Nbfec LDA - Recall:    {recall:.3f}")
print(f"y0 Nbfec LDA - F1 score:  {f1:.3f}")
print(f"y0 Nbfec LDA - AUC ROC:   {auc:.3f}")

# XGB:
X = y0nbfec_features_t.values
y = np.ravel(y0nbfec_target)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_index, test_index = next(sss.split(X, y))
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

XGBoost model
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
cv_scores = cross_val_score(xgb, X_train, y_train, cv=5, scoring='accuracy')
print("XGBoost 5-fold CV accuracy on training set: %.3f ± %.3f" %
     (cv_scores.mean(), cv_scores.std()))
xgb.fit(X_train, y_train)
test_acc = xgb.score(X_test, y_test)
print("Test set accuracy:", test_acc)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}
grid = GridSearchCV(XGBClassifier(eval_metric='logloss', random_state=42),
                    param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
print("XGB y0 pb + nbfec: Best parameters:", grid.best_params_)
print("XGB y0 pb + nbfec: Best CV accuracy:", grid.best_score_)
best_model = grid.best_estimator_
test_acc = best_model.score(X_test, y_test)
print("XGB y0 pb + nbfec: Test set accuracy (best model):", test_acc)
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 0]
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("LDA Confusion Matrix")
plt.show()
plt.savefig(f"/path/to/file/y0NBFec_LDA_CM.png", dpi=300)
precision = precision_score(y_test, y_pred, pos_label=0)
recall = recall_score(y_test, y_pred, pos_label=0)
f1 = f1_score(y_test, y_pred, pos_label=0)
auc = roc_auc_score(y_test, y_proba)
print(f"y0 + nbfec XGB - Precision: {precision:.3f}")
print(f"y0 + nbfec XGB - Recall:    {recall:.3f}")
print(f"y0 + nbfec XGB - F1 score:  {f1:.3f}")
print(f"y0 + nbfec XGB - AUC ROC:   {auc:.3f}")


# Feature importance 
best_xgb = grid.best_estimator_
feature_names = y0nbfec_features_t.columns  
result = permutation_importance(
    best_xgb, X_test, y_test, random_state=42, n_jobs=-1
)
importances_df = pd.DataFrame({
    "feature": feature_names,
    "importance": result.importances_mean
}).sort_values("importance", ascending=False)
print(importances_df.head(20))

#KNN
X = y0nbfec_features_t.values
y = np.ravel(y0nbfec_target)
ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_index, test_index = next(sss.split(X, y))
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])
param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9, 15],
    "knn__weights": ["uniform", "distance"],
    "knn__p": [1, 2]  # 1 = Manhattan distance, 2 = Euclidean distance
}
grid = GridSearchCV(pipeline,param_grid,cv=5,scoring="accuracy")

grid.fit(X_train, y_train)

print("KNN y0 + nbfec: Best parameters:", grid.best_params_)
print("KNN y0 + nbfec: Best CV accuracy:", grid.best_score_)

best_model = grid.best_estimator_
test_acc = best_model.score(X_test, y_test)

print("KNN y0 + nbfec: Test set accuracy (best model):", test_acc)


# repeated for different year timepoints