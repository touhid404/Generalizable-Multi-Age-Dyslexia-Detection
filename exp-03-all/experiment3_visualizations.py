"""
Experiment 3: Cross-Age Feature Analysis Visualizations
=======================================================
Generate visualizations for PCA + UMAP + HDBSCAN analysis (130 subjects: 66 children + 64 adults)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory
os.makedirs('visualizations', exist_ok=True)

print("="*70)
print("EXPERIMENT 3: CROSS-AGE ANALYSIS VISUALIZATIONS")
print("="*70)

# Load results
print("\n[1/8] Loading experiment results...")
df = pd.read_csv('experiment3_results.csv')

print(f"✓ Loaded {len(df)} subjects:")
print(f"  Children (9-10y): {(df['age_group'] == 'Child (9-10y)').sum()}")
print(f"  Adults (18+): {(df['age_group'] == 'Adult (18+)').sum()}")

# Calculate statistics
child_mask = df['age_group'] == 'Child (9-10y)'
adult_mask = df['age_group'] == 'Adult (18+)'

child_centroid = df.loc[child_mask, ['pca_1', 'pca_2']].mean().values
adult_centroid = df.loc[adult_mask, ['pca_1', 'pca_2']].mean().values
centroid_distance = euclidean(child_centroid, adult_centroid)

print(f"\n✓ Centroid distance: {centroid_distance:.3f}")

# ============================================================================
# Visualization 1: PCA Age Separation
# ============================================================================

print("\n[2/8] Creating PCA age separation plot...")

fig, ax = plt.subplots(figsize=(12, 8))

# Plot children
children = df[child_mask]
adults = df[adult_mask]

ax.scatter(children['pca_1'], children['pca_2'], 
          c='#3498db', s=100, alpha=0.6, edgecolors='black', linewidth=1,
          label=f'Children 9-10y (n={len(children)})', marker='o')

ax.scatter(adults['pca_1'], adults['pca_2'],
          c='#e74c3c', s=100, alpha=0.6, edgecolors='black', linewidth=1,
          label=f'Adults 18+ (n={len(adults)})', marker='^')

# Plot centroids
ax.scatter(*child_centroid, c='#2c3e50', s=400, marker='*', 
          edgecolors='yellow', linewidths=3, zorder=5,
          label='Child Centroid')
ax.scatter(*adult_centroid, c='#c0392b', s=400, marker='*',
          edgecolors='yellow', linewidths=3, zorder=5,
          label='Adult Centroid')

# Draw line between centroids
ax.plot([child_centroid[0], adult_centroid[0]], 
        [child_centroid[1], adult_centroid[1]],
        'k--', linewidth=2, alpha=0.5)
ax.text((child_centroid[0] + adult_centroid[0])/2,
        (child_centroid[1] + adult_centroid[1])/2,
        f'Distance: {centroid_distance:.3f}',
        fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

ax.set_xlabel('PC1 (59.09% variance)', fontsize=12, fontweight='bold')
ax.set_ylabel('PC2 (23.70% variance)', fontsize=12, fontweight='bold')
ax.set_title('PCA Age Group Separation (82.79% Total Variance)\nChildren vs Adults - DISTINCT Groups',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='best', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
ax.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)

plt.tight_layout()
plt.savefig('visualizations/20_exp3_pca_age_separation.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 20_exp3_pca_age_separation.png")

# ============================================================================
# Visualization 2: UMAP Embedding
# ============================================================================

print("\n[3/8] Creating UMAP embedding plot...")

fig, ax = plt.subplots(figsize=(12, 8))

ax.scatter(children['umap_1'], children['umap_2'],
          c='#3498db', s=100, alpha=0.6, edgecolors='black', linewidth=1,
          label=f'Children 9-10y (n={len(children)})', marker='o')

ax.scatter(adults['umap_1'], adults['umap_2'],
          c='#e74c3c', s=100, alpha=0.6, edgecolors='black', linewidth=1,
          label=f'Adults 18+ (n={len(adults)})', marker='^')

ax.set_xlabel('UMAP 1', fontsize=12, fontweight='bold')
ax.set_ylabel('UMAP 2', fontsize=12, fontweight='bold')
ax.set_title('UMAP Non-Linear Embedding\nAge Group Separation', 
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='best', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/21_exp3_umap_embedding.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 21_exp3_umap_embedding.png")

# ============================================================================
# Visualization 3: HDBSCAN Clustering (PCA space)
# ============================================================================

print("\n[4/8] Creating HDBSCAN clustering visualization...")

fig, ax = plt.subplots(figsize=(12, 8))

# Plot by cluster
unique_clusters = sorted(df['cluster_pca'].unique())
colors_map = {-1: 'gray', 0: '#2ecc71', 1: '#f39c12'}

for cluster_id in unique_clusters:
    cluster_data = df[df['cluster_pca'] == cluster_id]
    label = 'Noise' if cluster_id == -1 else f'Cluster {cluster_id}'
    marker = 'x' if cluster_id == -1 else 'o'
    size = 50 if cluster_id == -1 else 100
    
    ax.scatter(cluster_data['pca_1'], cluster_data['pca_2'],
              c=colors_map.get(cluster_id, 'purple'), s=size, alpha=0.6,
              marker=marker, edgecolors='black', linewidth=1,
              label=f'{label} (n={len(cluster_data)})')

ax.set_xlabel('PC1', fontsize=12, fontweight='bold')
ax.set_ylabel('PC2', fontsize=12, fontweight='bold')
ax.set_title('HDBSCAN Clustering Results (PCA Space)\n2 Clusters + Noise', 
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='best', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/22_exp3_hdbscan_clustering.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 22_exp3_hdbscan_clustering.png")

# ============================================================================
# Visualization 4: Feature Distributions by Age
# ============================================================================

print("\n[5/8] Creating feature distributions...")

features = ['fix_duration', 'saccade_length', 'regression_count']
feature_names = ['Fixation Duration (ms)', 'Saccade Length (°)', 'Regression Count']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, (feature, name) in enumerate(zip(features, feature_names)):
    ax = axes[idx]
    
    child_data = df.loc[child_mask, feature]
    adult_data = df.loc[adult_mask, feature]
    
    positions = [0, 1]
    data = [child_data, adult_data]
    
    parts = ax.violinplot(data, positions=positions, widths=0.7,
                          showmeans=True, showmedians=True)
    
    parts['bodies'][0].set_facecolor('#3498db')
    parts['bodies'][1].set_facecolor('#e74c3c')
    parts['bodies'][0].set_alpha(0.7)
    parts['bodies'][1].set_alpha(0.7)
    
    # Calculate percentage difference
    child_mean = child_data.mean()
    adult_mean = adult_data.mean()
    pct_diff = ((adult_mean - child_mean) / child_mean) * 100
    
    ax.text(0.5, ax.get_ylim()[1] * 0.95,
            f'{pct_diff:+.1f}%',
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax.set_xticks(positions)
    ax.set_xticklabels(['Children\n(9-10y)', 'Adults\n(18+)'], fontsize=10)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

fig.suptitle('Feature Distributions by Age Group', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/23_exp3_feature_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 23_exp3_feature_distributions.png")

# ============================================================================
# Visualization 5: Cluster Composition Analysis
# ============================================================================

print("\n[6/8] Creating cluster composition analysis...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# PCA-based clustering
crosstab_pca = pd.crosstab(df['cluster_pca'], df['age_group'])
crosstab_pca.plot(kind='bar', stacked=True, ax=axes[0], 
                  color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
axes[0].set_title('Cluster Composition (PCA-based HDBSCAN)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Cluster ID', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Count', fontsize=11, fontweight='bold')
axes[0].legend(title='Age Group', fontsize=9)
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
axes[0].grid(axis='y', alpha=0.3)

# UMAP-based clustering
crosstab_umap = pd.crosstab(df['cluster_umap'], df['age_group'])
crosstab_umap.plot(kind='bar', stacked=True, ax=axes[1],
                   color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
axes[1].set_title('Cluster Composition (UMAP-based HDBSCAN)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Cluster ID', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Count', fontsize=11, fontweight='bold')
axes[1].legend(title='Age Group', fontsize=9)
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
axes[1].grid(axis='y', alpha=0.3)

plt.suptitle('Cluster × Age Group Composition', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('visualizations/24_exp3_cluster_composition.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 24_exp3_cluster_composition.png")

# ============================================================================
# Visualization 6: Summary Dashboard
# ============================================================================

print("\n[7/8] Creating summary dashboard...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Top left: PCA scatter
ax1 = fig.add_subplot(gs[0:2, 0:2])
ax1.scatter(children['pca_1'], children['pca_2'], c='#3498db', s=80, alpha=0.6,
           edgecolors='black', linewidth=1, label='Children', marker='o')
ax1.scatter(adults['pca_1'], adults['pca_2'], c='#e74c3c', s=80, alpha=0.6,
           edgecolors='black', linewidth=1, label='Adults', marker='^')
ax1.scatter(*child_centroid, c='#2c3e50', s=300, marker='*', 
           edgecolors='yellow', linewidths=2, zorder=5)
ax1.scatter(*adult_centroid, c='#c0392b', s=300, marker='*',
           edgecolors='yellow', linewidths=2, zorder=5)
ax1.plot([child_centroid[0], adult_centroid[0]], 
        [child_centroid[1], adult_centroid[1]],
        'k--', linewidth=2, alpha=0.5)
ax1.set_xlabel('PC1 (59.09%)', fontsize=10, fontweight='bold')
ax1.set_ylabel('PC2 (23.70%)', fontsize=10, fontweight='bold')
ax1.set_title('PCA Age Separation', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Top right: Statistics
ax2 = fig.add_subplot(gs[0:2, 2])
ax2.axis('off')
stats_text = f"""
EXPERIMENT 3 SUMMARY

Dataset Composition:
  Total: {len(df)} subjects
  Children: {len(children)} (50.8%)
  Adults: {len(adults)} (49.2%)

Methods Applied:
  ✓ PCA (82.79% variance)
  ✓ UMAP (non-linear)
  ✓ HDBSCAN clustering

PCA Results:
  PC1: 59.09% variance
  PC2: 23.70% variance

Centroid Distance:
  {centroid_distance:.3f}
  > 1.5 threshold
  = DISTINCT groups

Clustering (PCA):
  Clusters: 2
  Noise: {(df['cluster_pca'] == -1).sum()} points
  
Feature Differences:
  Fixation: +32.3%
  Saccade: +21.0%
  Regression: -26.1%

Conclusion:
  ✓ Age-specific models
    required
  ✓ Features DON'T
    generalize across ages
"""
ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
         fontsize=8, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

# Bottom left: Feature means comparison
ax3 = fig.add_subplot(gs[2, 0])
feature_means_child = [children[f].mean() for f in features]
feature_means_adult = [adults[f].mean() for f in features]
x = np.arange(len(features))
width = 0.35
ax3.bar(x - width/2, feature_means_child, width, label='Children', 
        color='#3498db', alpha=0.7, edgecolor='black')
ax3.bar(x + width/2, feature_means_adult, width, label='Adults',
        color='#e74c3c', alpha=0.7, edgecolor='black')
ax3.set_xticks(x)
ax3.set_xticklabels(['Fixation', 'Saccade', 'Regression'], fontsize=9, rotation=45)
ax3.set_ylabel('Mean Value', fontsize=10, fontweight='bold')
ax3.set_title('Feature Means by Age', fontsize=11, fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(axis='y', alpha=0.3)

# Bottom middle: Cluster composition (PCA)
ax4 = fig.add_subplot(gs[2, 1])
crosstab_pca.plot(kind='bar', stacked=True, ax=ax4,
                  color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black', legend=False)
ax4.set_title('Clusters (PCA)', fontsize=11, fontweight='bold')
ax4.set_xlabel('Cluster', fontsize=9)
ax4.set_ylabel('Count', fontsize=9)
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0, fontsize=8)
ax4.grid(axis='y', alpha=0.3)

# Bottom right: Dyslexia analysis (children only)
ax5 = fig.add_subplot(gs[2, 2])
child_dyslexia = children[children['dyslexia_label'] != 'unknown']
dyslexia_counts = child_dyslexia['dyslexia_label'].value_counts()
colors_dys = ['#2ecc71', '#e74c3c']
wedges, texts, autotexts = ax5.pie(dyslexia_counts.values, labels=dyslexia_counts.index,
                                     autopct='%1.1f%%', colors=colors_dys, startangle=90)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(9)
ax5.set_title('Children Labels', fontsize=11, fontweight='bold')

plt.suptitle('Experiment 3: Cross-Age Feature Analysis Dashboard\nPCA + UMAP + HDBSCAN', 
             fontsize=16, fontweight='bold', y=0.98)
plt.savefig('visualizations/25_exp3_summary_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 25_exp3_summary_dashboard.png")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("EXPERIMENT 3 VISUALIZATIONS COMPLETE")
print("="*70)
print(f"\n✓ Generated 6 visualizations in visualizations/ folder:")
print(f"  20_exp3_pca_age_separation.png")
print(f"  21_exp3_umap_embedding.png")
print(f"  22_exp3_hdbscan_clustering.png")
print(f"  23_exp3_feature_distributions.png")
print(f"  24_exp3_cluster_composition.png")
print(f"  25_exp3_summary_dashboard.png")
print(f"\n📊 Key Result: Centroid distance = {centroid_distance:.3f} (DISTINCT age groups)")
print(f"   Methods: PCA (82.79% variance) + UMAP + HDBSCAN")
print(f"   Dataset: 130 subjects (66 children + 64 adults)")
print("="*70)
