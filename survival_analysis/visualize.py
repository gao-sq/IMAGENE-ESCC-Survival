
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

from scipy.stats import chi2_contingency
from sklearn.metrics import confusion_matrix

# Survival analysis specific library
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

from utils.logger import logger

def visualize_survival_results(results, model_type='Cox', output_dir='outputs/survival'):
    """Generate survival analysis visualization results"""
    logger.info(f"Generating {model_type} model survival analysis visualization...")
    
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    test_df = results['test_df']
    
    # 1. Kaplan-Meier survival curve (grouped by risk)
    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()
    
    high_risk = test_df[test_df['risk_group'] == 1]
    low_risk = test_df[test_df['risk_group'] == 0]
    
    kmf.fit(high_risk['time'], high_risk['status'], label='high_risk')
    ax = kmf.plot(ci_show=False, color='red')
    
    kmf.fit(low_risk['time'], low_risk['status'], label='low_risk')
    kmf.plot(ax=ax, ci_show=False, color='blue')
    
    # Add statistical test results
    results = logrank_test(
        high_risk['time'], low_risk['time'],
        high_risk['status'], low_risk['status']
    )
    p_value = results.p_value
    
    plt.title(f'Kaplan-Meier Survival Curve by Risk Group (Log-rank p={p_value:.4e})')
    plt.xlabel('Time (days)')
    plt.ylabel('Survival Probability')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/km_curve_{model_type}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Risk Score Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=test_df, x='risk_score', hue='status', 
                 multiple='stack', palette=['skyblue', 'salmon'])
    plt.axvline(x=test_df['risk_score'].median(), color='red', linestyle='--')
    plt.title('Risk Score Distribution')
    plt.xlabel('Risk Score')
    plt.ylabel('Number of Patients')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/risk_distribution_{model_type}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Save analysis results
    test_df.to_csv(f"{output_dir}/risk_predictions_{model_type}.csv", index=False)

def visualize_feature_importance(importance_df, model_type='Cox', output_dir='outputs/survival'):
    """Generate feature importance visualization"""
    logger.info("Generating feature importance visualization...")
    
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Feature importance bar chart
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(15)
    
    # Beautify feature names
    top_features = top_features.copy()
    top_features['feature'] = top_features['feature'].apply(
        lambda x: x.replace('region_', 'R').replace('density_type_', 'D')
                   .replace('ratio_type_', 'RT').replace('nn_', 'NN_')
    )
    
    sns.barplot(x='importance', y='feature', data=top_features)
    plt.title(f'Top 15 Feature Importance ({model_type} model)')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance_{model_type}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save feature importance data
    importance_df.to_csv(f"{output_dir}/feature_importance_{model_type}.csv", index=False)

def plot_km_curve(time1, time2, event_observed1, event_observed2, label1, label2, output_path):
    """
    Plot KM curve and save as image

    Args:
    time1 (array-like): First group time data
    time2 (array-like): Second group time data
    event_observed1 (array-like): First group event occurrence data
    event_observed2 (array-like): Second group event occurrence data
    label1 (str): First group label
    label2 (str): Second group label
    output_path (str): Image save path
    """
    p_value = logrank_test(time1, time2, event_observed1, event_observed2).p_value
    logger.info(f"Log-rank test P-value: {p_value:.4f}")
    kmf = KaplanMeierFitter()
    kmf.fit(time1, event_observed1, label=label1)
    ax = kmf.plot_survival_function()
    kmf.fit(time2, event_observed2, label=label2)
    kmf.plot_survival_function(ax=ax)
    plt.legend()
    plt.title(f"KM curve\nLog-rank P-value: {p_value:.4f}")
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrix(args, df):
    """Plot confusion matrix for 6 most inconsistent features"""
    feature_cm = {}
    feature_risk = {}
    for feature_col in df.columns:
        if feature_col in ['time', 'status', 'risk_group']:
            continue
        # Univariate Cox regression for grouping
        model = CoxPHFitter()
        model.fit(df[[feature_col, 'time', 'status']], duration_col='time', event_col='status')
        score = model.predict_partial_hazard(df[[feature_col]])
        df['feature_group'] = (score > score.median()).astype(int)

        # Calculate chi-square test p-value
        p_value = chi2_contingency(pd.crosstab(df['feature_group'], df['risk_group']))[1]
        feature_risk[feature_col] = p_value
        feature_cm[feature_col] = confusion_matrix(df['feature_group'], df['risk_group'])

    # Sort by p-value
    feature_risk = {k: v for k, v in sorted(feature_risk.items(), key=lambda item: item[1])}
    # Find 6 most inconsistent features
    top6_features = list(feature_risk.keys())[:6]
    logger.info(f"6 most inconsistent features: {top6_features}")

    # 6 feature confusion matrices, 2 rows 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, ax in enumerate(axes.flat):
        sns.heatmap(feature_cm[top6_features[i]], annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Confusion Matrix for {top6_features[i]}')
        ax.set_xlabel('feature_group')
        ax.set_ylabel('risk_group')
    plt.tight_layout()
    plt.savefig(f'{args.full_output_dir}/confusion_matrix.png')
    plt.close()

