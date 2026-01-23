import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os

def generate_comparative_charts():
    # Load results
    if not os.path.exists('comparison_results.json'):
        print("Error: comparison_results.json not found.")
        return
        
    with open('comparison_results.json', 'r') as f:
        results = json.load(f)

    # Prepare data for Accuracy Correlation
    data_acc = []
    for exp_ref, exp_data in results.items():
        for model_name, metrics in exp_data.items():
            data_acc.append({
                'Experiment': 'Exp I' if exp_ref == 'Exp1' else 'Exp II',
                'Model': model_name,
                'Accuracy': metrics['accuracy']
            })
    df_acc = pd.DataFrame(data_acc)

    # 1. Bar Chart: Accuracy Comparison (SVM vs RF vs XGBoost)
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    ax = sns.barplot(x='Experiment', y='Accuracy', hue='Model', data=df_acc, palette='muted')
    plt.title('Accuracy Comparison: SVM vs Random Forest vs XGBoost', fontsize=16)
    plt.ylabel('Accuracy Score', fontsize=14)
    plt.ylim(0, 1.1)
    
    # Add labels on top of bars
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f'{p.get_height():.2f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 9), 
                        textcoords='offset points',
                        fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('multi_model_accuracy.png')
    plt.close()
    print("Saved 'multi_model_accuracy.png'")

    # 2. Grouped Bar Chart: F1-Score for Dyslexic Class (Label '1')
    data_f1 = []
    for exp_ref, exp_data in results.items():
        for model_name, metrics in exp_data.items():
            # In some cases label might be '1' or 1 depending on JSON
            f1 = metrics['report'].get('1', {}).get('f1-score', 0)
            data_f1.append({
                'Experiment': 'Exp I' if exp_ref == 'Exp1' else 'Exp II',
                'Model': model_name,
                'F1-Score (Dyslexic)': f1
            })
    df_f1 = pd.DataFrame(data_f1)

    plt.figure(figsize=(12, 7))
    ax_f1 = sns.barplot(x='Experiment', y='F1-Score (Dyslexic)', hue='Model', data=df_f1, palette='pastel')
    plt.title('F1-Score (Dyslexic Class): Model Comparison', fontsize=16)
    plt.ylabel('F1-Score', fontsize=14)
    plt.ylim(0, 1.1)

    for p in ax_f1.patches:
        if p.get_height() > 0:
            ax_f1.annotate(f'{p.get_height():.2f}', 
                           (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha='center', va='center', 
                           xytext=(0, 9), 
                           textcoords='offset points',
                           fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('multi_model_f1_score.png')
    plt.close()
    print("Saved 'multi_model_f1_score.png'")

    # 3. Grid of Confusion Matrices for Experiment II (The most challenging one)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    model_names = ['SVM', 'Random Forest', 'XGBoost']
    
    for i, name in enumerate(model_names):
        cm = np.array(results['Exp2'][name]['cm'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Control', 'Dyslexic'], 
                    yticklabels=['Control', 'Dyslexic'])
        axes[i].set_title(f'{name} (Exp II)', fontsize=14)
        axes[i].set_xlabel('Predicted', fontsize=12)
        axes[i].set_ylabel('True', fontsize=12)

    plt.suptitle('Confusion Matrices for Experiment II: Cross-Dataset Generalization', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('multi_model_cm_grid.png')
    plt.close()
    print("Saved 'multi_model_cm_grid.png'")

if __name__ == "__main__":
    generate_comparative_charts()
