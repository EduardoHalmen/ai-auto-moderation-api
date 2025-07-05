import matplotlib.pyplot as plt
from itertools import cycle
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def plot_roc_curves(class_names, fpr, tpr, roc_auc, figsize=(15, 10)):
    """
    Plot ROC curves for each toxicity category.

    Parameters:
    class_names (list): List of class names.
    fpr (dict): Dictionary of false positive rates for each class.
    tpr (dict): Dictionary of true positive rates for each class.
    roc_auc (dict): Dictionary of ROC AUC scores for each class.
    figsize (tuple): Tuple specifying the figure size (default is (15, 10)).

    Returns:
    None: Displays the plot.
    """
    distinct_colors = ['#9b59b6', '#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#1abc9c', '#d35400']
    plt.figure(figsize=figsize)
    for class_name, color in zip(class_names, cycle(distinct_colors)):
        plt.plot(fpr[class_name], tpr[class_name], color=color, lw=2, label=f'{class_name} (AUC = {roc_auc[class_name]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Toxicity Category')
    plt.legend(loc="lower right", bbox_to_anchor=(1.15, 0))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_pr_curves(class_names, recall, precision, avg_precision, figsize=(15, 10)):
    """
    Plot precision-recall curves for each toxicity category.

    Parameters:
    class_names (list): List of class names.
    recall (dict): Dictionary of recall values for each class.
    precision (dict): Dictionary of precision values for each class.
    avg_precision (dict): Dictionary of average precision scores for each class.
    figsize (tuple): Tuple specifying the figure size (default is (15, 10)).

    Returns:
    None: Displays the plot.
    """
    distinct_colors = ['#9b59b6', '#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#1abc9c', '#d35400']
    plt.figure(figsize=figsize)
    for class_name, color in zip(class_names, cycle(distinct_colors)):
        plt.plot(recall[class_name], precision[class_name], color=color, lw=2, label=f'{class_name} (AP = {avg_precision[class_name]:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for Each Toxicity Category')
    plt.legend(loc="lower left", bbox_to_anchor=(1.15, 0))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_metrics_heatmap(metrics_df, figsize=(10, 6)):
    """
    Plot a heatmap of classification metrics for each toxicity category.

    Parameters:
    metrics_df (DataFrame): DataFrame containing the classification metrics.
    figsize (tuple): Tuple specifying the figure size (default is (10, 6)).

    Returns:
    None: Displays the heatmap.
    """
    vibrant_purple_palette = ['#4A148C', '#6A1B9A', '#8E24AA', '#AB47BC', '#CE93D8']
    plt.figure(figsize=figsize)
    vibrant_cmap = LinearSegmentedColormap.from_list("vibrant_purple", vibrant_purple_palette[::-1])
    sns.heatmap(metrics_df, annot=True, cmap=vibrant_cmap, fmt='.3f', cbar_kws={'label': 'Score'}, vmin=0, vmax=1)
    plt.title('Classification Metrics per Toxicity Category', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrices(class_names, results, figsize=(15, 10)):
    """
    Plot confusion matrices for each toxicity category.

    Parameters:
    class_names (list): List of class names.
    results (dict): Dictionary containing the results, including confusion matrices for each class.
    figsize (tuple): Tuple specifying the figure size (default is (15, 10)).

    Returns:
    None: Displays the confusion matrices.
    """
    purple_palette = ['#9b59b6', '#AF7AC5', '#C39BD3', '#D7BDE2', '#E8DAEF']
    dark_purple_palette = ['#6A0DAD', '#8E44AD', '#9B59B6', '#B19CD9', '#D2B4DE']
    n_classes = len(class_names)
    n_cols = 3
    n_rows = np.ceil(n_classes / n_cols)
    plt.figure(figsize=(15, 4*n_rows))
    custom_cmaps = []
    for i in range(len(class_names)):
        if i % 2 == 0:
            custom_cmaps.append(LinearSegmentedColormap.from_list(f"custom_purple_{i}", purple_palette[::-1]))
        else:
            custom_cmaps.append(LinearSegmentedColormap.from_list(f"dark_purple_{i}", dark_purple_palette[::-1]))
    for i, class_name in enumerate(class_names):
        cm = np.array([[results['per_class'][class_name]['confusion_matrix']['tn'], results['per_class'][class_name]['confusion_matrix']['fp']],
                       [results['per_class'][class_name]['confusion_matrix']['fn'], results['per_class'][class_name]['confusion_matrix']['tp']]])
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        safe_row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_percentage = cm.astype('float') / safe_row_sums * 100
        plt.subplot(n_rows, n_cols, i+1)
        sns.heatmap(cm_percentage, annot=True, fmt=',.1f', cmap=custom_cmaps[i], xticklabels=['Non-Toxic', 'Toxic'], yticklabels=['Non-Toxic', 'Toxic'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{class_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        for t in plt.gca().texts:
            t.set_text(t.get_text() + "%")
    plt.tight_layout()
    plt.show()
    
def plot_mean_metrics(results, figsize=(12, 6)):
    """
    Plot mean performance metrics across all toxicity categories.

    Parameters:
    results (dict): Dictionary containing the overall mean metrics.
    figsize (tuple): Tuple specifying the figure size (default is (12, 6)).

    Returns:
    None: Displays the bar plot.
    """
    plt.figure(figsize=figsize)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC AUC', 'PR AUC']
    values = [
        results['overall']['mean_accuracy'],
        results['overall']['mean_precision'],
        results['overall']['mean_recall'],
        results['overall']['mean_f1_score'],
        results['overall']['mean_roc_auc'],
        results['overall']['mean_average_precision']
    ]
    
    colors = ['#D7BDE2'] * 5 + ['#6A1B9A']
    
    bars = plt.barh(metrics, values, color=colors)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                 va='center', fontweight='bold')
    
    plt.xlim(0, 1.1)
    plt.title('Mean Performance Metrics', fontsize=16, fontweight='bold')
    plt.xlabel('Score', fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
def print_classification_report(class_names, results):
    """
    Print a detailed classification report for each toxicity category.

    Parameters:
    class_names (list): List of class names.
    results (dict): Dictionary containing the results, including precision, recall, F1-score, accuracy, ROC AUC, and average precision for each class.

    Returns:
    None: Prints the report.
    """
    print("\nDetailed Classification Report:")
    print("-" * 50)
    for class_name in class_names:
        print(f"\nCategory: {class_name}")
        print(f"Precision: {results['per_class'][class_name]['precision']:.3f}")
        print(f"Recall: {results['per_class'][class_name]['recall']:.3f}")
        print(f"F1-score: {results['per_class'][class_name]['f1_score']:.3f}")
        print(f"Accuracy: {results['per_class'][class_name]['accuracy']:.3f}")
        print(f"ROC AUC: {results['per_class'][class_name]['roc_auc']:.3f}")
        print(f"Average Precision: {results['per_class'][class_name]['average_precision']:.3f}")
    
    print("\nMean Metrics Across All Toxicity Categories:")
    print("-" * 50)
    for metric, value in results['overall'].items():
        if isinstance(value, dict):
            print(f"{metric}:")
            for sub_metric, sub_value in value.items():
                print(f"  {sub_metric}: {sub_value:.3f}")
        else:
            print(f"{metric}: {value:.3f}")