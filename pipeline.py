import pandas as pd
import numpy as np
import tensorflow as tf
import random
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, precision_recall_curve, auc)
from sklearn.ensemble import IsolationForest
from scipy.stats import spearmanr
import shap


# config
GLOBAL_SEED = 42

# Primary analysis
BACKGROUND_MAIN     = 50
COHORT_MAIN         = 100
NSAMPLES_MAIN       = 300

# Stability check
BACKGROUND_STABILITY = 50
COHORT_STABILITY     = 100
NSAMPLES_STABILITY   = 300
N_SHAP_STABILITY     = 3

# Multi-seed robustness 
BACKGROUND_ROBUSTNESS = 50
COHORT_ROBUSTNESS     = 100
NSAMPLES_ROBUSTNESS   = 300
ROBUSTNESS_SEEDS      = [0, 1, 2, 3, 4]

N_BOOTSTRAP = 1000


def run_pipeline():

    os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    tf.keras.utils.set_random_seed(GLOBAL_SEED)

    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass

    # load dataset
    df = pd.read_csv('data/creditcard.csv')
    print(f"[1/10] Dataset loaded: {df.shape[0]} transactions.")

    # split
    df = df.drop(['Time'], axis=1)
    df['Amount'] = np.log1p(df['Amount'])

    normal_df = df[df['Class'] == 0].drop('Class', axis=1)
    fraud_df  = df[df['Class'] == 1].drop('Class', axis=1)

    normal_trainval, normal_test = train_test_split(
        normal_df, test_size=0.2, random_state=GLOBAL_SEED)
    normal_train, normal_val = train_test_split(
        normal_trainval, test_size=0.2, random_state=GLOBAL_SEED)

    X_val = normal_val.copy()
    X_test = pd.concat([normal_test, fraud_df], axis=0)
    y_test = pd.Series(
        [0] * len(normal_test) + [1] * len(fraud_df), index=X_test.index)

    # preprocessing
    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(normal_train), columns=normal_train.columns)
    X_val = pd.DataFrame(
        scaler.transform(X_val), columns=X_val.columns)
    X_test = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns)

    feature_names = list(X_train.columns)
    input_dim = X_train.shape[1]

    # Cache numpy arrays
    X_train_np = X_train.values
    X_val_np   = X_val.values
    X_test_np  = X_test.values

    print("[2/10] Preprocessing complete.")

    # autoencoder
    autoencoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dense(8,  activation='relu'),
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(input_dim, activation='linear')
    ])
    autoencoder.compile(optimizer='adam', loss='mse')

    print("[3/10] Training Autoencoder...")
    history = autoencoder.fit(
        X_train_np, X_train_np,
        epochs=300, batch_size=256, shuffle=True,
        validation_data=(X_val_np, X_val_np),
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=7, restore_best_weights=True)],
        verbose=0
    )

    epochs_run    = len(history.history['loss'])
    best_val_loss = min(history.history['val_loss'])

    # autoencoder
    val_recon = autoencoder.predict(X_val_np, verbose=0)
    val_mse   = np.mean(np.square(X_val_np - val_recon), axis=1)

    test_recon = autoencoder.predict(X_test_np, verbose=0)
    mse        = np.mean(np.square(X_test_np - test_recon), axis=1)

    ae_thresholds = {p: np.percentile(val_mse, p) for p in [95, 97, 99, 99.5]}
    best_percentile = 99.5
    threshold = ae_thresholds[best_percentile]
    y_pred = (mse > threshold).astype(int)

    precision_pts, recall_pts, _ = precision_recall_curve(y_test, mse)
    pr_auc = auc(recall_pts, precision_pts)

    ae_metrics = {
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall':    recall_score(y_test, y_pred, zero_division=0),
        'f1':        f1_score(y_test, y_pred, zero_division=0),
        'roc_auc':   roc_auc_score(y_test, mse),
        'pr_auc':    pr_auc,
    }
    print(f"[4/10] Autoencoder trained ({epochs_run} epochs). F1={ae_metrics['f1']:.4f}")

    # isolation forest
    iso_forest = IsolationForest(
        n_estimators=100, contamination="auto",
        random_state=GLOBAL_SEED, n_jobs=-1)
    iso_forest.fit(X_train_np)

    iso_val_scores = -iso_forest.score_samples(X_val_np) # negate: higher = more anomalous
    iso_scores     = -iso_forest.score_samples(X_test_np)

    if_thresholds = {p: np.percentile(iso_val_scores, p) for p in [95, 97, 99, 99.5]}
    iso_best_percentile = 99.5
    iso_threshold = if_thresholds[iso_best_percentile]
    y_pred_iso = (iso_scores > iso_threshold).astype(int)

    iso_precision_pts, iso_recall_pts, _ = precision_recall_curve(y_test, iso_scores)
    iso_pr_auc = auc(iso_recall_pts, iso_precision_pts)

    iso_metrics = {
        'accuracy':  accuracy_score(y_test, y_pred_iso),
        'precision': precision_score(y_test, y_pred_iso, zero_division=0),
        'recall':    recall_score(y_test, y_pred_iso, zero_division=0),
        'f1':        f1_score(y_test, y_pred_iso, zero_division=0),
        'roc_auc':   roc_auc_score(y_test, iso_scores),
        'pr_auc':    iso_pr_auc,
    }
    print(f"[5/10] Isolation Forest trained. F1={iso_metrics['f1']:.4f}")

    # SHAP
    def ae_reconstruction_error(data):
        data = np.asarray(data)
        reconstruction = autoencoder.predict(data, verbose=0)
        return np.mean(np.square(data - reconstruction), axis=1)

    def iso_anomaly_score(data):
        return -iso_forest.score_samples(np.asarray(data))

    background = X_train.sample(BACKGROUND_MAIN, random_state=GLOBAL_SEED).values
    ae_explainer  = shap.KernelExplainer(ae_reconstruction_error, background)
    iso_explainer = shap.KernelExplainer(iso_anomaly_score, background)

    fraud_indices  = np.where(y_test.values == 1)[0]
    cohort_indices = np.random.RandomState(GLOBAL_SEED).choice(
        fraud_indices, size=min(COHORT_MAIN, len(fraud_indices)), replace=False)
    fraud_cohort = X_test_np[cohort_indices]

    # shap values
    print(f"[6/10] Computing SHAP values — Autoencoder (N={COHORT_MAIN})...")
    ae_cohort_shap = np.array(
        ae_explainer.shap_values(fraud_cohort, nsamples=NSAMPLES_MAIN))
    if ae_cohort_shap.ndim > 2:
        ae_cohort_shap = ae_cohort_shap.squeeze()
    assert ae_cohort_shap.shape == fraud_cohort.shape

    print(f"[7/10] Computing SHAP values — Isolation Forest (N={COHORT_MAIN})...")
    iso_cohort_shap = np.array(
        iso_explainer.shap_values(fraud_cohort, nsamples=NSAMPLES_MAIN))
    if iso_cohort_shap.ndim > 2:
        iso_cohort_shap = iso_cohort_shap.squeeze()
    assert iso_cohort_shap.shape == fraud_cohort.shape

    # feature importance
    ae_importance      = np.abs(ae_cohort_shap).mean(axis=0)
    iso_importance     = np.abs(iso_cohort_shap).mean(axis=0)
    ae_importance_norm = ae_importance / ae_importance.sum()
    iso_importance_norm = iso_importance / iso_importance.sum()

    ae_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'AE_Impact': ae_importance,
        'AE_Impact_Norm': ae_importance_norm
    }).sort_values('AE_Impact', ascending=False)

    iso_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'IF_Impact': iso_importance,
        'IF_Impact_Norm': iso_importance_norm
    }).sort_values('IF_Impact', ascending=False)

    # consistency
    rho_raw, p_raw = spearmanr(ae_importance, iso_importance)
    cosine_sim = np.dot(ae_importance_norm, iso_importance_norm) / (
        np.linalg.norm(ae_importance_norm) * np.linalg.norm(iso_importance_norm))

    # top-k overlap and jaccard
    ae_ranked  = ae_importance_df['Feature'].values
    iso_ranked = iso_importance_df['Feature'].values

    topk_results = []
    for k in [3, 5, 7, 10]:
        ae_top  = set(ae_ranked[:k])
        iso_top = set(iso_ranked[:k])
        inter   = ae_top & iso_top
        topk_results.append({
            'k': k,
            'overlap': len(inter) / k,
            'jaccard': len(inter) / len(ae_top | iso_top),
            'shared':  ', '.join(sorted(inter)) if inter else 'None'
        })

    # bootstrap
    print("[8/10] Bootstrap + permutation tests...")
    n_cohort = ae_cohort_shap.shape[0]
    rng      = np.random.RandomState(GLOBAL_SEED)

    boot_spearman = np.zeros(N_BOOTSTRAP)
    boot_cosine   = np.zeros(N_BOOTSTRAP)

    for i in range(N_BOOTSTRAP):
        idx = rng.choice(n_cohort, size=n_cohort, replace=True)
        ae_boot  = np.abs(ae_cohort_shap[idx]).mean(axis=0)
        iso_boot = np.abs(iso_cohort_shap[idx]).mean(axis=0)
        rho_b, _ = spearmanr(ae_boot, iso_boot)
        boot_spearman[i] = rho_b
        ae_bn  = ae_boot / ae_boot.sum()
        iso_bn = iso_boot / iso_boot.sum()
        boot_cosine[i] = np.dot(ae_bn, iso_bn) / (
            np.linalg.norm(ae_bn) * np.linalg.norm(iso_bn))

    bootstrap = {
        'spearman_mean': boot_spearman.mean(),
        'spearman_std':  boot_spearman.std(),
        'spearman_ci':   np.percentile(boot_spearman, [2.5, 97.5]),
        'cosine_mean':   boot_cosine.mean(),
        'cosine_std':    boot_cosine.std(),
        'cosine_ci':     np.percentile(boot_cosine, [2.5, 97.5]),
        'spearman_dist': boot_spearman,
        'cosine_dist':   boot_cosine,
    }

    # permutation
    perm_spearman = np.zeros(N_BOOTSTRAP)
    perm_cosine   = np.zeros(N_BOOTSTRAP)

    for i in range(N_BOOTSTRAP):
        iso_shuffled = rng.permutation(iso_importance)
        rho_p, _ = spearmanr(ae_importance, iso_shuffled)
        perm_spearman[i] = rho_p
        iso_n = iso_shuffled / iso_shuffled.sum()
        perm_cosine[i] = np.dot(ae_importance_norm, iso_n) / (
            np.linalg.norm(ae_importance_norm) * np.linalg.norm(iso_n))

    permutation = {
        'spearman_observed':  rho_raw,
        'spearman_null_mean': perm_spearman.mean(),
        'spearman_p':         (np.sum(perm_spearman >= rho_raw) + 1) / (N_BOOTSTRAP + 1),
        'cosine_observed':    cosine_sim,
        'cosine_null_mean':   perm_cosine.mean(),
        'cosine_p':           (np.sum(perm_cosine >= cosine_sim) + 1) / (N_BOOTSTRAP + 1),
        'spearman_dist':      perm_spearman,
        'cosine_dist':        perm_cosine,
    }

    # shap stability
    print(f"[9/10] SHAP stability check ({N_SHAP_STABILITY} reruns)...")
    stability_sample = fraud_cohort[:COHORT_STABILITY]

    stability_results = []
    for run in range(N_SHAP_STABILITY):
        seed = GLOBAL_SEED + run + 1
        np.random.seed(seed)

        bg = X_train.sample(BACKGROUND_STABILITY, random_state=seed).values
        ae_exp  = shap.KernelExplainer(ae_reconstruction_error, bg)
        iso_exp = shap.KernelExplainer(iso_anomaly_score, bg)

        ae_sv = np.array(ae_exp.shap_values(stability_sample, nsamples=NSAMPLES_STABILITY))
        if ae_sv.ndim > 2: ae_sv = ae_sv.squeeze()
        iso_sv = np.array(iso_exp.shap_values(stability_sample, nsamples=NSAMPLES_STABILITY))
        if iso_sv.ndim > 2: iso_sv = iso_sv.squeeze()

        ae_imp  = np.abs(ae_sv).mean(axis=0)
        iso_imp = np.abs(iso_sv).mean(axis=0)

        rho_s, _ = spearmanr(ae_imp, iso_imp)
        ae_sn  = ae_imp / ae_imp.sum()
        iso_sn = iso_imp / iso_imp.sum()
        cos_s = np.dot(ae_sn, iso_sn) / (np.linalg.norm(ae_sn) * np.linalg.norm(iso_sn))

        stability_results.append({
            'run':      run + 1,
            'seed':     seed,
            'spearman': rho_s,
            'cosine':   cos_s,
            'ae_top5':  [feature_names[i] for i in np.argsort(-ae_imp)[:5]],
            'if_top5':  [feature_names[i] for i in np.argsort(-iso_imp)[:5]],
        })

    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)

    # multi seed robustness
    print(f"[10/10] Multi-seed robustness ({len(ROBUSTNESS_SEEDS)} seeds)...")
    robustness_results = []

    for seed in ROBUSTNESS_SEEDS:
        tf.keras.backend.clear_session()
        random.seed(seed)
        np.random.seed(seed)
        tf.keras.utils.set_random_seed(seed)

        
        r_trainval, r_test_normal = train_test_split(
            normal_df, test_size=0.2, random_state=seed)
        r_train, r_val = train_test_split(
            r_trainval, test_size=0.2, random_state=seed)

        r_X_test = pd.concat([r_test_normal, fraud_df], axis=0)
        r_y_test = pd.Series(
            [0] * len(r_test_normal) + [1] * len(fraud_df), index=r_X_test.index)

        
        r_scaler     = StandardScaler()
        r_X_train_np = r_scaler.fit_transform(r_train)
        r_X_val_np   = r_scaler.transform(r_val)
        r_X_test_np  = r_scaler.transform(r_X_test)

        
        r_ae = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(20, activation='relu'),
            tf.keras.layers.Dense(12, activation='relu'),
            tf.keras.layers.Dense(8,  activation='relu'),
            tf.keras.layers.Dense(12, activation='relu'),
            tf.keras.layers.Dense(20, activation='relu'),
            tf.keras.layers.Dense(input_dim, activation='linear')
        ])
        r_ae.compile(optimizer='adam', loss='mse')
        r_hist = r_ae.fit(
            r_X_train_np, r_X_train_np,
            epochs=300, batch_size=256, shuffle=True,
            validation_data=(r_X_val_np, r_X_val_np),
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=7, restore_best_weights=True)],
            verbose=0)

        
        r_val_recon = r_ae.predict(r_X_val_np, verbose=0)
        r_val_mse   = np.mean(np.square(r_X_val_np - r_val_recon), axis=1)
        r_threshold = np.percentile(r_val_mse, 99.5)

        r_test_recon = r_ae.predict(r_X_test_np, verbose=0)
        r_mse        = np.mean(np.square(r_X_test_np - r_test_recon), axis=1)
        r_y_pred_ae  = (r_mse > r_threshold).astype(int)

        r_pr, r_rc, _ = precision_recall_curve(r_y_test, r_mse)

        
        r_iso = IsolationForest(
            n_estimators=100, contamination="auto", random_state=seed, n_jobs=-1)
        r_iso.fit(r_X_train_np)

        r_iso_val   = -r_iso.score_samples(r_X_val_np)
        r_iso_thr   = np.percentile(r_iso_val, 99.5)
        r_iso_sc    = -r_iso.score_samples(r_X_test_np)
        r_y_pred_if = (r_iso_sc > r_iso_thr).astype(int)

        r_iso_pr, r_iso_rc, _ = precision_recall_curve(r_y_test, r_iso_sc)

        
        def r_ae_error(data):
            data = np.asarray(data)
            recon = r_ae.predict(data, verbose=0)
            return np.mean(np.square(data - recon), axis=1)

        def r_iso_score(data):
            return -r_iso.score_samples(np.asarray(data))

        r_bg_idx   = np.random.RandomState(seed).choice(
            len(r_X_train_np), size=BACKGROUND_ROBUSTNESS, replace=False)
        r_bg       = r_X_train_np[r_bg_idx]
        r_ae_exp   = shap.KernelExplainer(r_ae_error, r_bg)
        r_iso_exp  = shap.KernelExplainer(r_iso_score, r_bg)

        r_fraud_idx    = np.where(r_y_test.values == 1)[0]
        r_cohort_idx   = np.random.RandomState(seed).choice(
            r_fraud_idx, size=min(COHORT_ROBUSTNESS, len(r_fraud_idx)), replace=False)
        r_fraud_cohort = r_X_test_np[r_cohort_idx]

        r_ae_shap = np.array(r_ae_exp.shap_values(r_fraud_cohort, nsamples=NSAMPLES_ROBUSTNESS))
        if r_ae_shap.ndim > 2: r_ae_shap = r_ae_shap.squeeze()
        r_iso_shap = np.array(r_iso_exp.shap_values(r_fraud_cohort, nsamples=NSAMPLES_ROBUSTNESS))
        if r_iso_shap.ndim > 2: r_iso_shap = r_iso_shap.squeeze()

        r_ae_imp  = np.abs(r_ae_shap).mean(axis=0)
        r_iso_imp = np.abs(r_iso_shap).mean(axis=0)

        r_rho, _ = spearmanr(r_ae_imp, r_iso_imp)
        r_ae_n   = r_ae_imp / r_ae_imp.sum()
        r_iso_n  = r_iso_imp / r_iso_imp.sum()
        r_cos    = np.dot(r_ae_n, r_iso_n) / (
            np.linalg.norm(r_ae_n) * np.linalg.norm(r_iso_n))

        r_ae_top5  = set(np.argsort(-r_ae_imp)[:5])
        r_iso_top5 = set(np.argsort(-r_iso_imp)[:5])

        robustness_results.append({
            'seed':        seed,
            'ae_f1':       f1_score(r_y_test, r_y_pred_ae, zero_division=0),
            'ae_prec':     precision_score(r_y_test, r_y_pred_ae, zero_division=0),
            'ae_rec':      recall_score(r_y_test, r_y_pred_ae, zero_division=0),
            'ae_prauc':    auc(r_rc, r_pr),
            'iso_f1':      f1_score(r_y_test, r_y_pred_if, zero_division=0),
            'iso_prec':    precision_score(r_y_test, r_y_pred_if, zero_division=0),
            'iso_rec':     recall_score(r_y_test, r_y_pred_if, zero_division=0),
            'iso_prauc':   auc(r_iso_rc, r_iso_pr),
            'spearman':    r_rho,
            'cosine':      r_cos,
            'top5_overlap': len(r_ae_top5 & r_iso_top5) / 5,
            'epochs':      len(r_hist.history['loss']),
        })

    # restore primary seed
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    tf.keras.utils.set_random_seed(GLOBAL_SEED)

    # threshold sensitivity
    threshold_sensitivity = []
    for p in [95, 97, 99, 99.5]:
        ae_pred_p  = (mse > ae_thresholds[p]).astype(int)
        iso_pred_p = (iso_scores > if_thresholds[p]).astype(int)
        threshold_sensitivity.append({
            'percentile': p,
            'ae_f1':      f1_score(y_test, ae_pred_p, zero_division=0),
            'ae_prec':    precision_score(y_test, ae_pred_p, zero_division=0),
            'ae_rec':     recall_score(y_test, ae_pred_p, zero_division=0),
            'iso_f1':     f1_score(y_test, iso_pred_p, zero_division=0),
            'iso_prec':   precision_score(y_test, iso_pred_p, zero_division=0),
            'iso_rec':    recall_score(y_test, iso_pred_p, zero_division=0),
        })

    
    print("Pipeline complete.")

    return {
        'config': {
            'global_seed':            GLOBAL_SEED,
            'background_main':        BACKGROUND_MAIN,
            'cohort_main':            COHORT_MAIN,
            'nsamples_main':          NSAMPLES_MAIN,
            'background_stability':   BACKGROUND_STABILITY,
            'cohort_stability':       COHORT_STABILITY,
            'nsamples_stability':     NSAMPLES_STABILITY,
            'n_shap_stability':       N_SHAP_STABILITY,
            'background_robustness':  BACKGROUND_ROBUSTNESS,
            'cohort_robustness':      COHORT_ROBUSTNESS,
            'nsamples_robustness':    NSAMPLES_ROBUSTNESS,
            'robustness_seeds':       ROBUSTNESS_SEEDS,
            'n_bootstrap':            N_BOOTSTRAP,
        },
        'data_splits': {
            'n_train':       len(normal_train),
            'n_val':         len(normal_val),
            'n_test_normal': len(normal_test),
            'n_test_fraud':  len(fraud_df),
        },
        'training': {
            'epochs_run':       epochs_run,
            'best_val_loss':    best_val_loss,
            'final_train_loss': history.history['loss'][-1],
            'loss_history':     history.history['loss'],
            'val_loss_history': history.history['val_loss'],
        },
        'ae_threshold': {
            'percentile_map':      ae_thresholds,
            'selected_percentile': best_percentile,
            'selected_value':      threshold,
        },
        'if_threshold': {
            'percentile_map':      if_thresholds,
            'selected_percentile': iso_best_percentile,
            'selected_value':      iso_threshold,
        },
        'ae_metrics':  ae_metrics,
        'iso_metrics': iso_metrics,
        'feature_names': feature_names,
        'feature_importance': {
            'ae_df':    ae_importance_df,
            'iso_df':   iso_importance_df,
            'ae_raw':   ae_importance,
            'iso_raw':  iso_importance,
            'ae_norm':  ae_importance_norm,
            'iso_norm': iso_importance_norm,
        },
        'consistency': {
            'spearman_raw': rho_raw,
            'spearman_p':   p_raw,
            'cosine_sim':   cosine_sim,
        },
        'topk':                 topk_results,
        'bootstrap':            bootstrap,
        'permutation':          permutation,
        'stability':            stability_results,
        'robustness':           robustness_results,
        'threshold_sensitivity': threshold_sensitivity,
        'plot_data': {
            'test_mse':          mse,
            'iso_scores':        iso_scores,
            'y_test':            y_test.values,
            'threshold':         threshold,
            'best_percentile':   best_percentile,
            'ae_cohort_shap':    ae_cohort_shap,
            'iso_cohort_shap':   iso_cohort_shap,
            'fraud_cohort':      fraud_cohort,
            'precision_pts':     precision_pts,
            'recall_pts':        recall_pts,
            'iso_precision_pts': iso_precision_pts,
            'iso_recall_pts':    iso_recall_pts,
        },
    }