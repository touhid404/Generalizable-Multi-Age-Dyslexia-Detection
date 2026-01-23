import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA
from data_loader import load_data

def run_experiments():
    print("Loading all datasets...")
    data_map = load_data(".")
    
    etdd_list = data_map['etdd70']
    kron_list = data_map['kronoberg']
    adult_list = data_map['adult']
    
    print(f"\nLoaded Samples: ETDD70={len(etdd_list)}, Kronoberg={len(kron_list)}, Adult={len(adult_list)}")
    
    # helper to df
    def to_df(data):
        if not data: return pd.DataFrame()
        return pd.DataFrame(data)
        
    df_etdd = to_df(etdd_list)
    df_kron = to_df(kron_list)
    df_adult = to_df(adult_list)
    
    # Feature columns (Domain-Invariant)
    feature_cols = [
        'fixation_duration_mean', 'fixation_duration_std', 'fixation_duration_max',
        'saccade_length_mean', 'saccade_length_std', 'fix_sac_ratio', 'regression_ratio'
    ]
    
    if df_etdd.empty:
        print("CRITICAL: No ETDD70 data loaded. Cannot run Experiment 1.")
        return

    # Imputer
    imputer = SimpleImputer(strategy='mean')
    
    # --- Experiment I: Intra-Dataset Baseline (ETDD70) ---
    print("\n=== Experiment I: Intra-Dataset Baseline (ETDD70) ===")
    X_etdd = df_etdd[feature_cols].values
    y_etdd = df_etdd['label'].values
    
    # 80/20 Split
    X_train, X_test, y_train, y_test = train_test_split(X_etdd, y_etdd, test_size=0.2, random_state=42)
    
    # Prepare results storage
    results = {
        'Exp1': {},
        'Exp2': {}
    }

    # Models to compare
    models = {
        'SVM': make_pipeline(imputer, StandardScaler(), SVC(kernel='rbf', C=1.0, random_state=42)),
        'Random Forest': make_pipeline(imputer, RandomForestClassifier(n_estimators=100, random_state=42)),
        'XGBoost': make_pipeline(imputer, XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    }

    for name, clf in models.items():
        print(f"\nTraining {name} for Experiment I...")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results['Exp1'][name] = {
            'accuracy': acc,
            'report': classification_report(y_test, y_pred, output_dict=True)
        }
        print(f"{name} Accuracy: {acc}")
    
    # --- Experiment II: Cross-Dataset Generalization (Kronoberg) ---
    print("\n=== Experiment II: Cross-Dataset Generalization (Kronoberg) ===")
    if not df_kron.empty:
        X_kron = df_kron[feature_cols].values
        y_kron = df_kron['label'].values
        
        # DOMAIN ADAPTATION: Quantile Normalization
        qt_etdd = QuantileTransformer(output_distribution='normal', n_quantiles=min(len(X_etdd), 100), random_state=42)
        X_etdd_imp = imputer.fit_transform(X_etdd)
        X_etdd_scaled = qt_etdd.fit_transform(X_etdd_imp)
        
        qt_kron = QuantileTransformer(output_distribution='normal', n_quantiles=min(len(X_kron), 100), random_state=42)
        X_kron_imp = imputer.transform(X_kron)
        X_kron_scaled = qt_kron.fit_transform(X_kron_imp)
        
        # Models for Experiment II (Using Domain Adaptation/DA principles)
        for name, model_obj in models.items():
            print(f"\nEvaluating {name} for Experiment II...")
            
            # Simple Domain Adaptation: Quantile Normalization
            qt_etdd = QuantileTransformer(output_distribution='normal', n_quantiles=min(len(X_etdd), 100), random_state=42)
            X_etdd_imp = imputer.fit_transform(X_etdd)
            X_etdd_scaled = qt_etdd.fit_transform(X_etdd_imp)
            
            qt_kron = QuantileTransformer(output_distribution='normal', n_quantiles=min(len(X_kron), 100), random_state=42)
            X_kron_imp = imputer.transform(X_kron)
            X_kron_scaled = qt_kron.fit_transform(X_kron_imp)
            
            # Extract basic estimator if pipeline
            if hasattr(model_obj, 'named_steps'):
                base_model = model_obj.steps[-1][1]
            else:
                base_model = model_obj
            
            # Re-fit on full scaled ETDD for generalization
            base_model.fit(X_etdd_scaled, y_etdd)
            y_pred_kron = base_model.predict(X_kron_scaled)
            
            acc_kron = accuracy_score(y_kron, y_pred_kron)
            results['Exp2'][name] = {
                'accuracy': acc_kron,
                'report': classification_report(y_kron, y_pred_kron, output_dict=True),
                'cm': confusion_matrix(y_kron, y_pred_kron).tolist()
            }
            print(f"{name} Accuracy on Kronoberg: {acc_kron}")

        # Save all results for visualization
        import json
        with open('comparison_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        print("\nComparative results saved to 'comparison_results.json'")
    else:
        print("No Kronoberg data for Experiment II.")

    # --- Experiment III: Unsupervised Cross-Age Analysis ---
    print("\n=== Experiment III: Unsupervised Cross-Age Analysis ===")
    # Combine all
    combined_frames = []
    if not df_etdd.empty: combined_frames.append(df_etdd)
    if not df_kron.empty: combined_frames.append(df_kron)
    if not df_adult.empty: combined_frames.append(df_adult)
    
    if combined_frames:
        df_all = pd.concat(combined_frames, ignore_index=True)
        actual_cols = [c for c in feature_cols if c in df_all.columns]
        X_all = df_all[actual_cols].values
        labels = df_all['group'].values
        
        X_all_imp = imputer.fit_transform(X_all)
        X_all_scaled = StandardScaler().fit_transform(X_all_imp)
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_all_scaled)
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels, style=labels)
        plt.title("PCA of Eye-Tracking Features (Cross-Age)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.savefig("pca_analysis.png")
        print("PCA plot saved to 'pca_analysis.png'")
        
        for grp in df_all['group'].unique():
             mask = df_all['group'] == grp
             center = X_pca[mask].mean(axis=0)
             print(f"Centroid {grp}: {center}")
    else:
        print("No data for Experiment III.")

if __name__ == "__main__":
    run_experiments()
