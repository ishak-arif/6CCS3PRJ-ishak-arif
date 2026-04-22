import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# Global style
plt.rcParams.update({
    'figure.dpi':       150,
    'savefig.dpi':      300,
    'font.size':        11,
    'axes.titlesize':   13,
    'axes.labelsize':   12,
    'legend.fontsize':  10,
    'figure.facecolor': 'white',
})

AE_COL  = '#B22222'
IF_COL  = '#4682B4'
OUT_DIR = 'outputs/plots'


def _save(name):
    plt.savefig(f'{OUT_DIR}/{name}.png', bbox_inches='tight')
    plt.close()


def generate_plots(r):
    """Create and save all dissertation plots."""

    os.makedirs(OUT_DIR, exist_ok=True)

    t      = r['training']
    fi     = r['feature_importance']
    con    = r['consistency']
    bst    = r['bootstrap']
    prm    = r['permutation']
    pd_    = r['plot_data']
    fnames = r['feature_names']
    rob    = r['robustness']

    # Class Distribution
    y_test = np.asarray(pd_['y_test'])
    n_normal = (r['data_splits']['n_test_normal']
                + r['data_splits']['n_train']
                + r['data_splits']['n_val'])
    n_fraud = r['data_splits']['n_test_fraud']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(['Normal', 'Fraud'], [n_normal, n_fraud],
           color=[IF_COL, AE_COL], alpha=0.85)
    ax.set_xlabel('Class')
    ax.set_ylabel('Frequency')
    ax.set_title('Transaction Class Distribution')
    for i, v in enumerate([n_normal, n_fraud]):
        ax.text(i, v + 1000, f'{v:,}', ha='center', fontsize=11)
    ax.grid(alpha=0.25, axis='y')
    fig.tight_layout()
    _save('class_distribution')

    # Training History
    fig, ax = plt.subplots(figsize=(9, 5))
    epochs = range(len(t['loss_history']))
    ax.plot(epochs, t['loss_history'],     label='Training Loss',
            color='tab:blue',   lw=2)
    ax.plot(epochs, t['val_loss_history'], label='Validation Loss',
            color='tab:orange', lw=2)
    best = int(np.argmin(t['val_loss_history']))
    ax.axvline(best, color='grey', ls=':', lw=1.2,
               label=f'Best Epoch ({best})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Squared Error')
    ax.set_yscale('log')
    ax.set_title('Autoencoder Training History')
    ax.legend()
    ax.grid(alpha=0.25)
    _save('training_history')

    # Autoencoder Error Distribution
    mse = np.asarray(pd_['test_mse'])
    thr = pd_['threshold']
    pct = pd_['best_percentile']

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(mse[y_test == 0], bins=50, alpha=0.50, color='tab:blue',
            density=True, label='Normal')
    ax.hist(mse[y_test == 1], bins=20, alpha=0.70, color='tab:red',
            density=True, label='Fraud')
    ax.axvline(thr, color='black', ls='--', lw=1.5,
               label=f'{pct}th Percentile Threshold ({thr:.2f})')
    ax.set_yscale('log')
    ax.set_xlabel('Reconstruction Error (MSE)')
    ax.set_ylabel('Density (Log Scale)')
    ax.set_title('Reconstruction Error Distribution: Normal vs. Fraud')
    ax.legend()
    ax.grid(alpha=0.25)
    _save('error_distribution')

    # Isolation Forest Score Distribution 
    iso_sc  = np.asarray(pd_['iso_scores'])
    iso_thr = r['if_threshold']['selected_value']
    iso_pct = r['if_threshold']['selected_percentile']

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(iso_sc[y_test == 0], bins=50, alpha=0.50, color='tab:blue',
            density=True, label='Normal')
    ax.hist(iso_sc[y_test == 1], bins=20, alpha=0.70, color='tab:red',
            density=True, label='Fraud')
    ax.axvline(iso_thr, color='black', ls='--', lw=1.5,
               label=f'{iso_pct}th Percentile Threshold ({iso_thr:.4f})')
    ax.set_yscale('log')
    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Density (Log Scale)')
    ax.set_title('Isolation Forest Score Distribution: Normal vs. Fraud')
    ax.legend()
    ax.grid(alpha=0.25)
    _save('iso_score_distribution')

    # Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(pd_['recall_pts'], pd_['precision_pts'], color=AE_COL, lw=2,
            label=f'Autoencoder  (PR AUC = {r["ae_metrics"]["pr_auc"]:.3f})')
    ax.plot(pd_['iso_recall_pts'], pd_['iso_precision_pts'], color=IF_COL, lw=2,
            label=f'Isolation Forest  (PR AUC = {r["iso_metrics"]["pr_auc"]:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision–Recall Curves')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.25)
    _save('precision_recall_curve')

    # Threshold Sensitivity
    ts     = r['threshold_sensitivity']
    pcts   = [d['percentile'] for d in ts]
    ae_f1s = [d['ae_f1']  for d in ts]
    if_f1s = [d['iso_f1'] for d in ts]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(pcts)), ae_f1s, color=AE_COL, lw=2, marker='o',
            label='Autoencoder')
    ax.plot(range(len(pcts)), if_f1s, color=IF_COL, lw=2, marker='o',
            label='Isolation Forest')
    ax.set_xticks(range(len(pcts)))
    ax.set_xticklabels(['95th', '97th', '99th', '99.5th'])
    ax.set_xlabel('Threshold Percentile')
    ax.set_ylabel('F1 Score')
    ax.set_ylim(0, 1)
    ax.set_title('F1 Score Across Threshold Percentiles')
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save('threshold_sensitivity')

    # Paired Normalised SHAP Comparison
    comp = pd.DataFrame({
        'Feature':         fnames,
        'Autoencoder':     fi['ae_norm'],
        'IsolationForest': fi['iso_norm'],
    })
    comp['Avg'] = (comp['Autoencoder'] + comp['IsolationForest']) / 2
    comp = comp.sort_values('Avg', ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(9, 7))
    y  = np.arange(len(comp))
    bh = 0.35
    ax.barh(y - bh/2, comp['Autoencoder'],    bh, color=AE_COL,
            alpha=0.85, label='Autoencoder')
    ax.barh(y + bh/2, comp['IsolationForest'], bh, color=IF_COL,
            alpha=0.85, label='Isolation Forest')
    ax.set_yticks(y)
    ax.set_yticklabels(comp['Feature'])
    ax.set_xlabel('Normalised SHAP Importance')
    ax.set_title(f'Feature Importance Comparison  '
                 f'(ρ = {con["spearman_raw"]:.3f},  '
                 f'cos = {con["cosine_sim"]:.3f})')
    ax.legend()
    ax.grid(alpha=0.25, axis='x')
    fig.tight_layout()
    _save('shap_comparison_paired')

    # SHAP Beeswarms
    cohort_n = r['config']['cohort_main']

    plt.figure(figsize=(9, 7))
    shap.summary_plot(pd_['ae_cohort_shap'], pd_['fraud_cohort'],
                      feature_names=fnames, show=False)
    plt.title(f'SHAP Beeswarm — Autoencoder  (N = {cohort_n})')
    _save('shap_beeswarm_autoencoder')

    plt.figure(figsize=(9, 7))
    shap.summary_plot(pd_['iso_cohort_shap'], pd_['fraud_cohort'],
                      feature_names=fnames, show=False)
    plt.title(f'SHAP Beeswarm — Isolation Forest  (N = {cohort_n})')
    _save('shap_beeswarm_isolation_forest')

    # Top-k Overlap
    topk     = r['topk']
    ks       = [row['k'] for row in topk]
    overlaps = [row['overlap'] for row in topk]
    jaccards = [row['jaccard'] for row in topk]

    fig, ax = plt.subplots(figsize=(8, 5))
    bw = 0.35
    x  = np.arange(len(ks))
    ax.bar(x - bw/2, overlaps, bw, color=AE_COL, alpha=0.85,
           label='Overlap (shared / k)')
    ax.bar(x + bw/2, jaccards, bw, color=IF_COL, alpha=0.85,
           label='Jaccard (shared / union)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'k = {k}' for k in ks])
    ax.set_ylabel('Agreement')
    ax.set_ylim(0, 1)
    ax.set_title('Top-k Feature Overlap Between Models')
    ax.legend()
    ax.grid(alpha=0.25, axis='y')
    fig.tight_layout()
    _save('topk_overlap')

    # Bootstrap Distributions
    rho  = con['spearman_raw']
    cs   = con['cosine_sim']
    s_ci = bst['spearman_ci']
    c_ci = bst['cosine_ci']
    n_b  = r['config']['n_bootstrap']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.hist(bst['spearman_dist'], bins=50, color='tab:purple',
             alpha=0.7, edgecolor='white')
    ax1.axvline(rho,     color='black', ls='--', lw=2,
                label=f'Observed ρ = {rho:.3f}')
    ax1.axvline(s_ci[0], color='red',   ls=':',  lw=1.4,
                label=f'95 % CI [{s_ci[0]:.3f}, {s_ci[1]:.3f}]')
    ax1.axvline(s_ci[1], color='red',   ls=':',  lw=1.4)
    ax1.set_xlabel('Spearman ρ')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Bootstrap — Spearman ρ')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.25)

    ax2.hist(bst['cosine_dist'], bins=50, color='tab:green',
             alpha=0.7, edgecolor='white')
    ax2.axvline(cs,     color='black', ls='--', lw=2,
                label=f'Observed = {cs:.3f}')
    ax2.axvline(c_ci[0], color='red',  ls=':',  lw=1.4,
                label=f'95 % CI [{c_ci[0]:.3f}, {c_ci[1]:.3f}]')
    ax2.axvline(c_ci[1], color='red',  ls=':',  lw=1.4)
    ax2.set_xlabel('Cosine Similarity')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Bootstrap — Cosine Similarity')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25)

    fig.suptitle(f'Bootstrap Uncertainty  (B = {n_b})', fontsize=13)
    fig.tight_layout()
    _save('bootstrap_distributions')

    # Permutation Null Distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.hist(prm['spearman_dist'], bins=50, color='grey',
             alpha=0.7, edgecolor='white', label='Null distribution')
    ax1.axvline(rho, color=AE_COL, ls='--', lw=2,
                label=f'Observed ρ = {rho:.3f}')
    ax1.set_xlabel('Spearman ρ')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Permutation Test — Spearman  '
                  f'(p = {prm["spearman_p"]:.4f})')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.25)

    ax2.hist(prm['cosine_dist'], bins=50, color='grey',
             alpha=0.7, edgecolor='white', label='Null distribution')
    ax2.axvline(cs, color=AE_COL, ls='--', lw=2,
                label=f'Observed = {cs:.3f}')
    ax2.set_xlabel('Cosine Similarity')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Permutation Test — Cosine  '
                  f'(p = {prm["cosine_p"]:.4f})')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25)

    fig.suptitle(f'Permutation Null Distributions  (N = {n_b})', fontsize=13)
    fig.tight_layout()
    _save('permutation_null_distributions')

    # Multi-Seed Robustness
    rob_df = pd.DataFrame(rob)
    seeds  = rob_df['seed'].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.scatter(seeds, rob_df['spearman'], color='tab:purple',
                s=70, zorder=3, label='Per-seed ρ')
    ax1.axhline(rob_df['spearman'].mean(), color='tab:purple', ls='--', lw=1.2,
                label=f'Mean = {rob_df["spearman"].mean():.3f}')
    ax1.axhline(rho, color='black', ls=':', lw=1.2,
                label=f'Primary analysis = {rho:.3f}')
    ax1.set_xlabel('Random Seed')
    ax1.set_ylabel('Spearman ρ')
    ax1.set_title('Spearman ρ Across Seeds')
    ax1.set_xticks(seeds)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.25)

    ax2.scatter(seeds, rob_df['cosine'], color='tab:green',
                s=70, zorder=3, label='Per-seed cosine')
    ax2.axhline(rob_df['cosine'].mean(), color='tab:green', ls='--', lw=1.2,
                label=f'Mean = {rob_df["cosine"].mean():.3f}')
    ax2.axhline(cs, color='black', ls=':', lw=1.2,
                label=f'Primary analysis = {cs:.3f}')
    ax2.set_xlabel('Random Seed')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('Cosine Similarity Across Seeds')
    ax2.set_xticks(seeds)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25)

    fig.suptitle(f'Multi-Seed Robustness  ({len(seeds)} full retrains)',
                 fontsize=13)
    fig.tight_layout()
    _save('robustness_across_seeds')

    print(f"Plots saved to {OUT_DIR}/ ({len(os.listdir(OUT_DIR))} files).")