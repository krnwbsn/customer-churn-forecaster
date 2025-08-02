import shap
import matplotlib.pyplot as plt
from src.utils import save_fig
from src.config import OUTPUT_DIR

def explain_model(pipeline, X):
    """
    Generate SHAP summary plots for either a linear model or a tree based model.
    """
    # get the preprocessing step from the pipeline
    preproc = pipeline.named_steps['preproc']
    # get the classifier step from the pipeline
    clf = pipeline.named_steps['clf']

    # transform the original feature set into the numeric matrix
    X_trans = preproc.transform(X)
    # take a small random sample to use as background data for SHAP
    background = shap.sample(X_trans, 100, random_state=42)

    # choose the right SHAP explainer depending on the model type
    if hasattr(shap, "TreeExplainer") and clf.__class__.__name__.endswith("Classifier"):
        # use TreeExplainer for tree based models like LightGBM or XGBoost
        explainer = shap.TreeExplainer(clf, data=background)
        shap_values = explainer.shap_values(X_trans)
    else:
        # use LinearExplainer for linear models
        explainer = shap.LinearExplainer(
            clf,
            masker=background,
            feature_perturbation="interventional"
        )
        # shap_values is a 2D array for binary classification
        shap_values = explainer.shap_values(X_trans)

    # create a bar chart of mean absolute SHAP values
    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_values,
        X_trans,
        feature_names=preproc.get_feature_names_out(),
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    save_fig(plt.gcf(), OUTPUT_DIR + 'shap_summary_bar.png')
    plt.close()

    # create a dot plot showing distribution of SHAP values per feature
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_trans,
        feature_names=preproc.get_feature_names_out(),
        plot_type="dot",
        show=False
    )
    plt.tight_layout()
    save_fig(plt.gcf(), OUTPUT_DIR + 'shap_summary_dot.png')
    plt.close()
