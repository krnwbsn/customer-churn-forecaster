from sklearn.model_selection import RandomizedSearchCV

def tune_pipeline(pipeline, param_dist, X, y, n_iter=20, cv=5):
    """
    Run randomized search over hyperparameter distributions
    to optimize the pipeline for ROC-AUC.
    - pipeline: a scikit-learn Pipeline object
    - param_dist: dictionary of parameter distributions to sample
    - X, y: feature matrix and target vector
    - n_iter: number of parameter settings to try
    - cv: number of cross-validation folds
    Returns the best-estimator pipeline with tuned hyperparameters.
    """
    # set up RandomizedSearchCV with ROC-AUC as the scoring metric
    search = RandomizedSearchCV(
        estimator=pipeline, # pipeline containing preprocessing and classifier
        param_distributions=param_dist, # hyperparameters to test
        n_iter=n_iter, # how many parameter combinations to try
        scoring='roc_auc', # performance metric
        cv=cv, # number of cross-validation folds
        random_state=42, # ensure reproducible results
        n_jobs=-1, # use all CPU cores
        verbose=1 # show progress during search
    )

    # run the search on the training data
    search.fit(X, y)

    # print out the best parameters and associated CV score
    print("Best params:", search.best_params_)
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")

    # return the pipeline configured with the best-found hyperparameters
    return search.best_estimator_
