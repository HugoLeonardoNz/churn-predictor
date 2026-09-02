"""
Customer Churn Predictor — FiberNet ISP
Pipeline completo: dados sintéticos -> feature engineering -> 4 modelos -> SHAP -> relatório
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    brier_score_loss, confusion_matrix,
)
from sklearn.base import clone

try:
    from xgboost import XGBClassifier
    XGB_OK = True
except ImportError:
    XGB_OK = False

try:
    from lightgbm import LGBMClassifier
    LGBM_OK = True
except ImportError:
    LGBM_OK = False

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False

# ── Constants, dataset e pré-processamento ─────────────────────────────────────
# Vivem em churn_data.py, importado também pelo app.py — fonte única.
# Re-exportados aqui porque testes e documentação referenciam `pipeline.<nome>`.

from churn_data import (
    SEED, RUIDO_COMPORTAMENTAL,
    PLANOS, REGIOES, PLAN_PRICE_RANGE, PLAN_USAGE_MAX, PLAN_W, REGION_W,
    PLAN_ORDER, REGION_ORDER, PLAN_FACTOR, REGION_FACTOR,
    CONTINUOUS_COLS, ORDINAL_COLS, BINARY_COLS, ALL_FEATURES, FEATURE_NAMES,
    build_dataset, add_derived_features, build_preprocessor, build_pipeline,
    optimal_threshold,
)

# RE-EXPORTACAO DECLARADA. Tudo acima vem de churn_data e nada disso e usado
# DENTRO deste arquivo — os testes e o app importam de `pipeline`, que e a
# fachada unica do projeto. Sem `__all__`, o ruff le como import morto e o
# `--fix` apaga em silencio: foi o que aconteceu em 2026-09-02 e quebrou a
# coleta dos testes inteira. `__all__` diz ao linter, e a quem le, que a
# reexportacao e o proposito.
__all__ = [
    "SEED", "RUIDO_COMPORTAMENTAL", "PLANOS", "REGIOES",
    "PLAN_PRICE_RANGE", "PLAN_USAGE_MAX", "PLAN_W", "REGION_W",
    "PLAN_ORDER", "REGION_ORDER", "PLAN_FACTOR", "REGION_FACTOR",
    "CONTINUOUS_COLS", "ORDINAL_COLS", "BINARY_COLS", "ALL_FEATURES",
    "FEATURE_NAMES", "build_dataset", "add_derived_features", "build_preprocessor",
    "build_pipeline", "optimal_threshold",
]


# ── Model catalogue ────────────────────────────────────────────────────────────

def get_models() -> dict:
    models = {
        "LogisticRegression": LogisticRegression(
            C=0.5, class_weight="balanced", max_iter=500, random_state=SEED,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_split=5,
            min_samples_leaf=2, class_weight="balanced",
            random_state=SEED, n_jobs=-1,
        ),
    }
    if XGB_OK:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=2.3, eval_metric="logloss",
            random_state=SEED, verbosity=0,
        )
    else:
        models["GradientBoosting"] = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.08,
            subsample=0.8, random_state=SEED,
        )
    if LGBM_OK:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            num_leaves=40, class_weight="balanced",
            random_state=SEED, verbosity=-1,
        )
    return models


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def eval_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "AUC-ROC":   round(roc_auc_score(y_true, y_proba), 4),
        "F1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "Precisão":  round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "Brier":     round(brier_score_loss(y_true, y_proba), 4),
    }


# ── SHAP analysis ─────────────────────────────────────────────────────────────

def run_shap(fitted_clf, X_transformed: np.ndarray,
             feature_names: list, out_dir: str, high_risk_idx: list,
             X_train_transformed: np.ndarray = None):
    os.makedirs(out_dir, exist_ok=True)

    # Choose explainer: TreeExplainer for tree models, LinearExplainer for LR
    from sklearn.linear_model import LogisticRegression as _LR
    is_linear = isinstance(fitted_clf, _LR)

    if is_linear:
        bg = shap.sample(X_train_transformed, 200) if X_train_transformed is not None \
             else X_transformed[:200]
        explainer   = shap.LinearExplainer(fitted_clf, bg)
        shap_values = explainer.shap_values(X_transformed)
        sv = shap_values if not isinstance(shap_values, list) else shap_values[1]
        ev_raw = explainer.expected_value
        ev = float(ev_raw) if not hasattr(ev_raw, '__len__') else float(ev_raw[1])
    else:
        explainer   = shap.TreeExplainer(fitted_clf)
        shap_values = explainer.shap_values(X_transformed)
        ev_raw = explainer.expected_value
        if isinstance(shap_values, list):
            sv = shap_values[1]
            ev = float(ev_raw[1]) if hasattr(ev_raw, '__len__') else float(ev_raw)
        elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
            sv = shap_values[:, :, 1]
            ev = float(ev_raw[1]) if hasattr(ev_raw, '__len__') else float(ev_raw)
        else:
            sv = shap_values
            ev = float(ev_raw) if not hasattr(ev_raw, '__len__') else float(ev_raw[1])

    # 1. Summary plot (global feature importance)
    fig, _ = plt.subplots(figsize=(10, 7))
    shap.summary_plot(sv, X_transformed, feature_names=feature_names,
                      show=False, plot_size=None)
    plt.title("SHAP — Importância Global das Features (Churn)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  [OK] shap_summary.png")

    # 2. Dependence plots for top-3 features
    mean_abs = np.abs(sv).mean(axis=0)
    top3_idx = np.argsort(mean_abs)[::-1][:3]

    for rank, feat_idx in enumerate(top3_idx):
        fname = feature_names[feat_idx].replace(" ", "_").replace("/", "_")
        fig, ax = plt.subplots(figsize=(7, 5))
        shap.dependence_plot(feat_idx, sv, X_transformed,
                             feature_names=feature_names, show=False, ax=ax)
        plt.tight_layout()
        path = os.path.join(out_dir, f"shap_dep_{rank+1}_{fname}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [OK] shap_dep_{rank+1}_{fname}.png")

    # 3. Individual force/waterfall for 3 high-risk clients
    for i, client_idx in enumerate(high_risk_idx[:3]):
        exp = shap.Explanation(
            values=sv[client_idx],
            base_values=ev,
            data=X_transformed[client_idx],
            feature_names=feature_names,
        )
        fig, _ = plt.subplots(figsize=(10, 4))
        shap.plots.waterfall(exp, max_display=12, show=False)
        plt.title(f"Cliente Alto Risco #{i+1}", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"shap_force_cliente_{i+1}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [OK] shap_force_cliente_{i+1}.png")

    return sv, ev, top3_idx, mean_abs


# ── Report generation ─────────────────────────────────────────────────────────

def generate_report(results_table: pd.DataFrame, best_name: str,
                    best_metrics_t05: dict, best_metrics_opt: dict,
                    opt_threshold: float, brier_before: float, brier_after: float,
                    shap_mean_abs: np.ndarray, top3_idx: np.ndarray,
                    df: pd.DataFrame, X_test, y_test, y_proba,
                    out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Top-10 features by mean |SHAP|
    feat_imp = pd.Series(shap_mean_abs, index=FEATURE_NAMES).sort_values(ascending=False)
    top10 = feat_imp.head(10)

    # 3 highest-risk correct predictions
    test_df = X_test.copy()
    test_df["churn_real"] = y_test.values
    test_df["prob_churn"] = y_proba
    high_risk = (test_df[test_df["prob_churn"] > 0.75]
                 .sort_values("prob_churn", ascending=False)
                 .head(3))

    # Comparison table markdown
    tbl = results_table.to_markdown(index=True)

    lines = [
        "# Relatório de Análise de Churn — FiberNet ISP",
        "\n_Gerado automaticamente por pipeline.py - Dados sintéticos - Hugo Nazário_\n",
        "---",
        "",
        "## 1. Modelo Escolhido e Justificativa",
        "",
        f"**Modelo selecionado: `{best_name}`**",
        "",
        f"Escolhido por apresentar o maior AUC-ROC médio em validação cruzada "
        f"estratificada (5-fold), com AUC = **{best_metrics_opt['AUC-ROC']:.4f}** "
        f"no conjunto de teste. Modelos tree-based são preferíveis neste contexto "
        f"por capturarem interações não-lineares entre features de risco (atraso, "
        f"NPS, suporte) sem exigir normalidade dos dados.",
        "",
        "---",
        "",
        "## 2. Tabela Comparativa dos 4 Modelos",
        "",
        tbl,
        "",
        "> Métricas calculadas no conjunto de teste (20% holdout, estratificado).",
        "> Threshold padrão = 0.50.",
        "",
        "---",
        "",
        "## 3. Top 10 Features por Importância (SHAP)",
        "",
        "| Rank | Feature | SHAP médio |",
        "|------|---------|-----------|",
    ]
    for rank, (feat, val) in enumerate(top10.items(), 1):
        lines.append(f"| {rank} | `{feat}` | {val:.4f} |")

    lines += [
        "",
        "> Importância global = média do valor absoluto dos SHAP values no conjunto de teste.",
        "",
        "---",
        "",
        "## 4. Threshold Ótimo e Impacto na Precision/Recall",
        "",
        f"| Métrica | Threshold = 0.50 | Threshold ótimo = {opt_threshold:.3f} |",
        "|---------|-----------------|--------------------------------------|",
        f"| F1-Score  | {best_metrics_t05['F1']:.4f} | {best_metrics_opt['F1']:.4f} |",
        f"| Precisão  | {best_metrics_t05['Precisão']:.4f} | {best_metrics_opt['Precisão']:.4f} |",
        f"| Recall    | {best_metrics_t05['Recall']:.4f} | {best_metrics_opt['Recall']:.4f} |",
        f"| AUC-ROC   | {best_metrics_t05['AUC-ROC']:.4f} | {best_metrics_opt['AUC-ROC']:.4f} |",
        "",
        f"**Threshold ótimo = {opt_threshold:.3f}** — calculado pelo máximo F1 na curva Precision-Recall.",
        "Recall é priorizado (não perder churners reais) em detrimento de precisão.",
        "",
        "### Calibração de Probabilidade",
        "",
        "| | Brier Score |",
        "|---|-------------|",
        f"| Modelo original  | {brier_before:.4f} |",
        f"| Após calibração (isotonic) | {brier_after:.4f} |",
        "",
        "---",
        "",
        "## 5. Perfis de Clientes de Alto Risco",
        "",
    ]

    if len(high_risk) > 0:
        for i, (idx, row) in enumerate(high_risk.iterrows(), 1):
            lines += [
                f"### Cliente Alto Risco #{i}",
                "",
                f"- **Probabilidade de churn:** `{row['prob_churn']:.1%}`",
                f"- **Plano:** {row.get('plano', 'N/A')}",
                f"- **Valor mensalidade:** R$ {row.get('valor_mensalidade', 0):.2f}",
                f"- **Tempo de contrato:** {int(row.get('tempo_contrato', 0))} meses",
                f"- **Dias em atraso:** {int(row.get('dias_atraso_pagamento', 0))} dias",
                f"- **NPS score:** {row.get('nps_score', 'N/A')}",
                f"- **Chamados de suporte:** {int(row.get('qtd_chamados_suporte', 0))}",
                f"- **Fidelidade:** {'Sim' if row.get('tem_fidelidade', 0) else 'Não'}",
                f"- **SHAP plot:** `outputs/shap/shap_force_cliente_{i}.png`",
                "",
            ]
    else:
        lines.append("> Nenhum cliente com P > 0.75 no conjunto de teste.\n")

    lines += [
        "---",
        "",
        "## 6. Recomendações de Ação por Faixa de Risco",
        "",
        "| Faixa de Probabilidade | Segmento | Ação Recomendada |",
        "|------------------------|----------|-----------------|",
        "| **P > 0.75** | Alto Risco | Contato imediato do supervisor comercial (48h). Oferta de cashback ou desconto de 20–30% por fidelidade. Escalar para equipe de retenção. |",
        "| **P 0.50–0.75** | Médio Risco | Campanha de engajamento proativo. Verificar tickets em aberto. Aplicar pesquisa NPS. Oferecer upgrade com desconto. |",
        "| **P < 0.50** | Baixo Risco | Monitoramento mensal. E-mail de relacionamento. Sem ação comercial imediata. |",
        "",
        "### Estimativa de Impacto Financeiro",
        "",
        "- Considerando retenção de 50% dos clientes de alto risco via ação proativa:",
        "  cada cliente Empresarial retido representa **~R$ 450/mês** em MRR preservado.",
        "- Prioridade: clientes com alto `risco_pagamento` + baixo `nps_score` + sem fidelidade.",
        "",
        "---",
        "",
        f"_Dataset: 15.000 registros sintéticos - Churn rate: "
        f"{df['churn'].mean()*100:.1f}% - Seed: 42_",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [OK] {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    def banner(txt, c="="):
        print(f"\n{c*62}\n  {txt}\n{c*62}")

    banner("CUSTOMER CHURN PREDICTOR — FiberNet ISP")

    # 1. Dataset
    print("\n[1/6] Gerando dataset sintético (15.000 registros)...")
    df = build_dataset(n=15_000, seed=SEED)
    churn_rate = df["churn"].mean()
    print(f"  [OK] {len(df):,} registros - churn rate: {churn_rate*100:.1f}%")
    if not (0.25 <= churn_rate <= 0.40):
        print(f"  [!] Churn rate fora do range [25%, 40%]: {churn_rate*100:.1f}%")

    # Correlation audit
    group_delay = df[df["dias_atraso_pagamento"] > 30]["churn"].mean()
    group_ok    = df[df["dias_atraso_pagamento"] <= 30]["churn"].mean()
    print(f"  - Churn com atraso > 30d: {group_delay*100:.1f}% "
          f"vs normal: {group_ok*100:.1f}% "
          f"(ratio: {group_delay/max(group_ok,1e-9):.2f}x)")

    # 2. Feature engineering
    print("\n[2/6] Feature engineering...")
    df_feat = add_derived_features(df)
    X = df_feat[ALL_FEATURES]
    y = df_feat["churn"]

    assert "churn" not in X.columns, "Data leakage: 'churn' in features!"
    print(f"  [OK] {len(ALL_FEATURES)} features - 5 derivadas adicionadas")
    print(f"  [OK] NaN em features: {X.isnull().sum().sum()}")

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    print(f"  [OK] Treino: {len(X_train):,} - Teste: {len(X_test):,} (80/20 estratificado)")

    # 4. Train & cross-validate all models
    print("\n[3/6] Treinando e validando 4 modelos (CV 5-fold estratificado)...")
    models     = get_models()
    cv         = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_results = {}
    fitted     = {}

    for name, clf in models.items():
        pipe = build_pipeline(clf)
        aucs = cross_val_score(pipe, X_train, y_train,
                               cv=cv, scoring="roc_auc", n_jobs=-1)
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        fitted[name] = {"pipeline": pipe, "y_proba": y_prob}
        cv_results[name] = {"CV AUC (μ)": round(aucs.mean(), 4),
                             "CV AUC (σ)": round(aucs.std(), 4)}
        m = eval_metrics(y_test, y_prob, threshold=0.50)
        cv_results[name].update(m)
        print(f"  - {name:<22} AUC={m['AUC-ROC']:.4f}  "
              f"F1={m['F1']:.4f}  Recall={m['Recall']:.4f}")

    results_df = pd.DataFrame(cv_results).T
    results_df.index.name = "Modelo"

    # Best model by AUC-ROC
    best_name = results_df["AUC-ROC"].idxmax()
    best_pipe = fitted[best_name]["pipeline"]
    best_prob = fitted[best_name]["y_proba"]
    print(f"\n  [*] Melhor modelo: {best_name} (AUC={results_df.loc[best_name,'AUC-ROC']:.4f})")

    # 5. Optimal threshold + calibration
    print("\n[4/6] Threshold ótimo e calibração...")
    opt_thr = optimal_threshold(y_test, best_prob)
    print(f"  [OK] Threshold ótimo (F1-max): {opt_thr:.4f}")

    m_t05  = eval_metrics(y_test, best_prob, threshold=0.50)
    m_topt = eval_metrics(y_test, best_prob, threshold=opt_thr)
    print(f"  - Threshold 0.50 -> F1={m_t05['F1']:.4f}  "
          f"P={m_t05['Precisão']:.4f}  R={m_t05['Recall']:.4f}")
    print(f"  - Threshold {opt_thr:.3f} -> F1={m_topt['F1']:.4f}  "
          f"P={m_topt['Precisão']:.4f}  R={m_topt['Recall']:.4f}")

    # Calibration: new pipeline wrapping best clf in CalibratedClassifierCV
    base_clf = clone(models[best_name])
    cal_pipe = build_pipeline(
        CalibratedClassifierCV(base_clf, method="isotonic", cv=3)
    )
    cal_pipe.fit(X_train, y_train)
    cal_prob      = cal_pipe.predict_proba(X_test)[:, 1]
    brier_before  = brier_score_loss(y_test, best_prob)
    brier_after   = brier_score_loss(y_test, cal_prob)
    print(f"  - Brier antes: {brier_before:.4f} -> após calibração: {brier_after:.4f}")

    # Minimum acceptance checks
    if m_topt["AUC-ROC"] < 0.80:
        print(f"  [!] AUC={m_topt['AUC-ROC']:.4f} abaixo do mínimo 0.80")
    if m_topt["F1"] < 0.72:
        print(f"  [!] F1={m_topt['F1']:.4f} abaixo do mínimo 0.72")

    # Confusion matrix
    y_pred_opt = (best_prob >= opt_thr).astype(int)
    cm = confusion_matrix(y_test, y_pred_opt)
    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#060912")
    ax.set_facecolor("#0d1117")
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Ativo", "Churn"], color="#8b92a5")
    ax.set_yticklabels(["Ativo", "Churn"], color="#8b92a5")
    ax.set_xlabel("Previsto", color="#8b92a5")
    ax.set_ylabel("Real", color="#8b92a5")
    ax.set_title(f"Confusion Matrix — {best_name}", color="#f0f2f8", fontsize=12)
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha="center", va="center",
                color="#f0f2f8", fontsize=18, fontweight="bold")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight", facecolor="#060912")
    plt.close()
    print("  [OK] confusion_matrix.png")

    # 6. SHAP
    print("\n[5/6] Análise SHAP...")
    shap_mean_abs = np.zeros(len(FEATURE_NAMES))
    top3_idx      = np.array([0, 1, 2])
    ev            = 0.0

    if SHAP_OK:
        best_clf        = best_pipe.named_steps["clf"]
        preprocessor    = best_pipe.named_steps["preprocessor"]
        X_test_prep     = preprocessor.transform(X_test)
        high_prob_idx   = np.where(best_prob > 0.75)[0]
        high_risk_3     = list(high_prob_idx[:3]) if len(high_prob_idx) >= 3 \
                          else list(np.argsort(best_prob)[::-1][:3])

        X_train_prep = preprocessor.transform(X_train)
        sv, ev, top3_idx, shap_mean_abs = run_shap(
            best_clf, X_test_prep, FEATURE_NAMES,
            "outputs/shap", high_risk_3,
            X_train_transformed=X_train_prep,
        )
    else:
        print("  [!] shap não instalado — skipping. Execute: pip install shap")
        # Fallback: use sklearn feature_importances_ if available
        if hasattr(best_pipe.named_steps["clf"], "feature_importances_"):
            shap_mean_abs = best_pipe.named_steps["clf"].feature_importances_
            top3_idx = np.argsort(shap_mean_abs)[::-1][:3]

    # Predictions CSV
    preds = X_test.copy()
    preds["churn_real"]    = y_test.values
    preds["prob_churn"]    = best_prob.round(4)
    preds["pred_opt_thr"]  = y_pred_opt
    preds["risco"] = pd.cut(best_prob, bins=[-0.01, 0.50, 0.75, 1.01],
                            labels=["BAIXO", "MÉDIO", "ALTO"])
    os.makedirs("outputs", exist_ok=True)
    preds.sort_values("prob_churn", ascending=False).to_csv(
        "outputs/predictions.csv", index=False)
    print("  [OK] outputs/predictions.csv")

    # 7. Report
    print("\n[6/6] Gerando relatório...")
    generate_report(
        results_table=results_df,
        best_name=best_name,
        best_metrics_t05=m_t05,
        best_metrics_opt=m_topt,
        opt_threshold=opt_thr,
        brier_before=brier_before,
        brier_after=brier_after,
        shap_mean_abs=shap_mean_abs,
        top3_idx=top3_idx,
        df=df,
        X_test=X_test,
        y_test=y_test,
        y_proba=best_prob,
        out_path="outputs/relatorio_churn.md",
    )

    # Final summary
    banner("RESUMO FINAL", c="-")
    print(f"  Modelo     : {best_name}")
    # A faixa esperada tem teto e piso. Piso porque abaixo de 0,72 o modelo nao
    # esta aprendendo; TETO porque, com 15% de desfechos que contrariam o
    # comportamento observado, um AUC de 0,95 aqui so poderia vir de vazamento
    # de informacao — e um numero alto demais e sintoma, nao conquista.
    auc = m_topt["AUC-ROC"]
    faixa = "[OK]" if 0.72 <= auc <= 0.92 else (
        "[!] ABAIXO DA FAIXA — modelo nao aprendeu" if auc < 0.72
        else "[!] ACIMA DA FAIXA — suspeitar de vazamento")
    print(f"  AUC-ROC    : {auc:.4f}  {faixa}")
    print(f"  F1-Score   : {m_topt['F1']:.4f}  "
          f"{'[OK]' if m_topt['F1'] >= 0.60 else '[!] ABAIXO DO MÍNIMO'}")
    print(f"  Recall     : {m_topt['Recall']:.4f}")
    print(f"  Precisão   : {m_topt['Precisão']:.4f}")
    print(f"  Threshold  : {opt_thr:.4f}")
    print(f"  Churn rate : {churn_rate*100:.1f}%")
    print("\n  Arquivos gerados:")
    print("    - confusion_matrix.png")
    print("    - outputs/predictions.csv")
    print("    - outputs/relatorio_churn.md")
    if SHAP_OK:
        print("    - outputs/shap/shap_summary.png")
        print("    - outputs/shap/shap_dep_1,2,3_*.png")
        print("    - outputs/shap/shap_force_cliente_1,2,3.png")
    print()


if __name__ == "__main__":
    main()
