import os
import pandas as pd
import numpy as np


def export_results(r):
    """Write all result CSVs from the pipeline results dictionary."""

    os.makedirs('outputs/results', exist_ok=True)

    # performance metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
    pd.DataFrame({
        'Metric':           metrics,
        'Autoencoder':      [r['ae_metrics'][m] for m in metrics],
        'Isolation_Forest': [r['iso_metrics'][m] for m in metrics],
    }).to_csv('outputs/results/performance_metrics.csv', index=False)

    # feature importance
    fi = pd.DataFrame({
        'Feature':        r['feature_names'],
        'AE_Impact':      r['feature_importance']['ae_raw'],
        'AE_Impact_Norm': r['feature_importance']['ae_norm'],
        'IF_Impact':      r['feature_importance']['iso_raw'],
        'IF_Impact_Norm': r['feature_importance']['iso_norm'],
    })
    fi['AE_Rank'] = fi['AE_Impact'].rank(ascending=False, method='dense').astype(int)
    fi['IF_Rank'] = fi['IF_Impact'].rank(ascending=False, method='dense').astype(int)
    fi = fi.sort_values('AE_Rank')
    fi.to_csv('outputs/results/feature_importance.csv', index=False)

    # consistency metrics
    con = r['consistency']
    bst = r['bootstrap']
    prm = r['permutation']
    pd.DataFrame([
        {'Metric': 'Spearman_rho',
         'Observed': con['spearman_raw'],
         'Analytical_p': con['spearman_p'],
         'Bootstrap_Mean': bst['spearman_mean'],
         'Bootstrap_Std': bst['spearman_std'],
         'Bootstrap_CI_Lower': bst['spearman_ci'][0],
         'Bootstrap_CI_Upper': bst['spearman_ci'][1],
         'Permutation_Null_Mean': prm['spearman_null_mean'],
         'Permutation_p': prm['spearman_p']},
        {'Metric': 'Cosine_Similarity',
         'Observed': con['cosine_sim'],
         'Analytical_p': np.nan,
         'Bootstrap_Mean': bst['cosine_mean'],
         'Bootstrap_Std': bst['cosine_std'],
         'Bootstrap_CI_Lower': bst['cosine_ci'][0],
         'Bootstrap_CI_Upper': bst['cosine_ci'][1],
         'Permutation_Null_Mean': prm['cosine_null_mean'],
         'Permutation_p': prm['cosine_p']},
    ]).to_csv('outputs/results/consistency_metrics.csv', index=False)

    # Top-k overlap
    pd.DataFrame(r['topk']).to_csv('outputs/results/topk_overlap.csv', index=False)

    # Threshold selection (both models)
    rows = []
    for p, v in r['ae_threshold']['percentile_map'].items():
        rows.append({'Model': 'Autoencoder', 'Percentile': p, 'Threshold': v,
                     'Selected': (p == r['ae_threshold']['selected_percentile'])})
    for p, v in r['if_threshold']['percentile_map'].items():
        rows.append({'Model': 'Isolation_Forest', 'Percentile': p, 'Threshold': v,
                     'Selected': (p == r['if_threshold']['selected_percentile'])})
    pd.DataFrame(rows).to_csv('outputs/results/threshold_selection.csv', index=False)

    # threshold sensitivity
    pd.DataFrame(r['threshold_sensitivity']).to_csv(
        'outputs/results/threshold_sensitivity.csv', index=False)

    # stability summary
    stab_rows = []
    for s in r['stability']:
        stab_rows.append({
            'Run':      s['run'],
            'Seed':     s['seed'],
            'Spearman': s['spearman'],
            'Cosine':   s['cosine'],
            'AE_Top5':  '; '.join(s['ae_top5']),
            'IF_Top5':  '; '.join(s['if_top5']),
        })
    stab_df = pd.DataFrame(stab_rows)

    stab_df.to_csv('outputs/results/stability_summary.csv', index=False)

    pd.DataFrame([{
        'Spearman_Mean': stab_df['Spearman'].mean(),
        'Spearman_Std':  stab_df['Spearman'].std(),
        'Cosine_Mean':   stab_df['Cosine'].mean(),
        'Cosine_Std':    stab_df['Cosine'].std(),
    }]).to_csv('outputs/results/stability_summary_stats.csv', index=False)

    # robustness (per seed)
    rob_df = pd.DataFrame(r['robustness'])
    rob_df.to_csv('outputs/results/robustness_per_seed.csv', index=False)

    # Robustness (summary statistics)

    summary_cols = ['ae_f1', 'ae_prauc', 'iso_f1', 'iso_prauc',
                    'spearman', 'cosine', 'top5_overlap']
    rob_summary = []
    for col in summary_cols:
        vals = rob_df[col]
        rob_summary.append({
            'Metric': col,
            'Mean':   vals.mean(),
            'Std':    vals.std(),
            'Min':    vals.min(),
            'Max':    vals.max(),
        })
    pd.DataFrame(rob_summary).to_csv('outputs/results/robustness_summary.csv', index=False)

    # Training summary
    t = r['training']
    pd.DataFrame([{
        'Epochs_Run':       t['epochs_run'],
        'Best_Val_Loss':    t['best_val_loss'],
        'Final_Train_Loss': t['final_train_loss'],
    }]).to_csv('outputs/results/training_summary.csv', index=False)

    # Config
    cfg = r['config']
    cfg_rows = [{'Parameter': k, 'Value': str(v)} for k, v in cfg.items()]
    pd.DataFrame(cfg_rows).to_csv('outputs/results/config.csv', index=False)

    # Data splits
    pd.DataFrame([r['data_splits']]).to_csv('outputs/results/data_splits.csv', index=False)

    print(f"Results exported to outputs/results/ ({len(os.listdir('outputs/results'))} CSV files).")