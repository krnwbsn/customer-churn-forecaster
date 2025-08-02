from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report

def cross_validate(model, X, y, n_splits=5):
    # set up stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    # evaluate ROC-AUC over each fold
    aucs = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
    # display mean and standard deviation of ROC-AUC
    print(f"ROC-AUC CV: {aucs.mean():.3f} ± {aucs.std():.3f}")
    return aucs

def train_final(model, X, y):
    # train the model on the full training set
    model.fit(X, y)
    return model

def evaluate(model, X_test, y_test):
    # get predicted probabilities for the positive (churn) class
    y_proba = model.predict_proba(X_test)[:, 1]
    # get predicted class labels
    y_pred = model.predict(X_test)
    # compute ROC-AUC on the test set
    auc = roc_auc_score(y_test, y_proba)
    print(f"Test ROC-AUC  : {auc:.3f}")
    # print precision, recall, and f1-score
    print("\nClassification Report:\n",
          classification_report(y_test, y_pred, digits=3))
