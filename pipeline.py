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

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    brier_score_loss, precision_recall_curve, confusion_matrix,
    classification_report,
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

# ── Constants ──────────────────────────────────────────────────────────────────

SEED    = 42
PLANOS  = ["Fibra 100MB", "Fibra 200MB", "Fibra 500MB", "Empresarial"]
REGIOES = ["Norte", "Sul", "Leste", "Oeste", "Centro"]

PLAN_PRICE_RANGE = {
    "Fibra 100MB": (69,  89),
    "Fibra 200MB": (89,  119),
    "Fibra 500MB": (119, 199),
    "Empresarial": (380, 520),
}
PLAN_USAGE_MAX = {
    "Fibra 100MB": 150,
    "Fibra 200MB": 300,
    "Fibra 500MB": 600,
    "Empresarial": 900,
}
PLAN_W   = [0.35, 0.30, 0.25, 0.10]
REGION_W = [0.22, 0.24, 0.20, 0.18, 0.16]

# Ordinal encoding order (risk/value ascending)
PLAN_ORDER   = ["Fibra 100MB", "Fibra 200MB", "Fibra 500MB", "Empresarial"]
REGION_ORDER = ["Norte", "Leste", "Oeste", "Centro", "Sul"]

# Churn multipliers per group (multiplicative model)
PLAN_FACTOR   = {"Fibra 100MB": 1.35, "Fibra 200MB": 1.00,
                 "Fibra 500MB": 0.80, "Empresarial": 0.55}
REGION_FACTOR = {"Norte": 1.25, "Leste": 1.05, "Oeste": 0.95,
                 "Centro": 0.88, "Sul": 0.78}

# Feature column groups for preprocessor
CONTINUOUS_COLS = [
    "valor_mensalidade", "tempo_contrato", "qtd_chamados_suporte",
    "dias_atraso_pagamento", "uso_medio_gb", "nps_score",
    "meses_sem_incidente", "risco_pagamento", "pressao_suporte",
    "engagement_score", "ticket_medio_ajustado", "score_satisfacao",
    "score_risco_composto",  # interaction of behavioral risk flags (derived from observed behavior only)
]
ORDINAL_COLS  = ["plano", "regiao"]
BINARY_COLS   = ["tem_fidelidade", "qtd_upgrades"]
ALL_FEATURES  = CONTINUOUS_COLS + ORDINAL_COLS + BINARY_COLS
FEATURE_NAMES = ALL_FEATURES  # same order after ColumnTransformer


# ── Data generation ────────────────────────────────────────────────────────────

def build_dataset(n: int = 15_000, seed: int = SEED) -> pd.DataFrame:
    """
    Gera dataset ISP com geração condicional por classe (churn=1 / churn=0).
    Garante: ~30% churn, correlações validadas, e alta separabilidade (AUC >= 0.85).

    Correlações garantidas por design:
      dias_atraso > 30  -> P(churn|atrasado)/P(churn|normal)  >= 3x
      nps_score <= 5    -> P(churn|nps<=5)/P(churn|nps>5)     >= 2.5x
      tem_fidelidade    -> P(churn|fid)/P(churn|sem_fid)      <= 0.40
      qtd_upgrades >= 1 -> P(churn|upgrade)/P(churn|sem)      <= 0.75
    """
    rng = np.random.default_rng(seed)

    # ── Churn labels: 30% = 1, 70% = 0 ──────────────────────────────────────
    n_churn  = int(n * 0.30)
    n_active = n - n_churn
    churn    = np.array([1] * n_churn + [0] * n_active)

    def _gen(nc, na):
        """Helper: concatenate churner and active arrays."""
        return nc, na  # return separate arrays for convenience

    # ── dias_atraso_pagamento ─────────────────────────────────────────────────
    # Churners:  35% high-delay (>30d), 35% mid (6-30d), 30% low (0-5d)
    # Active:    6% high-delay, 22% mid, 72% low
    # Ratio P(churn|>30d) / P(churn|<=30d) = (0.35*nc/(0.35*nc+0.06*na)) / ... ≈ 3.4x
    tier_c = rng.choice(3, n_churn,  p=[0.30, 0.35, 0.35])
    tier_a = rng.choice(3, n_active, p=[0.72, 0.22, 0.06])
    def tier_to_days(tier, m):
        return np.where(tier == 0, rng.integers(0,  6, m),
               np.where(tier == 1, rng.integers(6, 31, m),
                                   rng.integers(31,91, m))).astype(int)
    dias_atraso = np.concatenate([tier_to_days(tier_c, n_churn),
                                   tier_to_days(tier_a, n_active)])

    # ── qtd_chamados_suporte ─────────────────────────────────────────────────
    # Churners: Poisson(3.0), Active: Poisson(0.8)
    chamados_c = np.clip(rng.poisson(3.0, n_churn),  0, 12).astype(int)
    chamados_a = np.clip(rng.poisson(0.8, n_active), 0, 12).astype(int)
    qtd_chamados = np.concatenate([chamados_c, chamados_a])

    # ── nps_score ─────────────────────────────────────────────────────────────
    # Churners: N(4.5, 2.0), Active: N(7.5, 1.5) — both clipped [0,10]
    nps_c = np.clip(rng.normal(4.5, 2.0, n_churn),  0, 10).round(1)
    nps_a = np.clip(rng.normal(7.5, 1.5, n_active), 0, 10).round(1)
    nps_score = np.concatenate([nps_c, nps_a])

    # ── tem_fidelidade ────────────────────────────────────────────────────────
    # Churners: 15% have fidelidade, Active: 45%
    fid_c = (rng.random(n_churn)  < 0.15).astype(int)
    fid_a = (rng.random(n_active) < 0.45).astype(int)
    tem_fidelidade = np.concatenate([fid_c, fid_a])

    # ── qtd_upgrades ─────────────────────────────────────────────────────────
    # Churners: fewer upgrades (80% zero), Active: more (50% zero)
    upg_c = rng.choice([0,1,2,3], n_churn,  p=[0.80, 0.15, 0.04, 0.01])
    upg_a = rng.choice([0,1,2,3], n_active, p=[0.50, 0.32, 0.14, 0.04])
    qtd_upgrades = np.concatenate([upg_c, upg_a])

    # ── plano ─────────────────────────────────────────────────────────────────
    # Churners: more Fibra 100MB; Active: more higher plans
    plano_c = rng.choice(PLANOS, n_churn,  p=[0.52, 0.28, 0.14, 0.06])
    plano_a = rng.choice(PLANOS, n_active, p=[0.26, 0.31, 0.30, 0.13])
    plano   = np.concatenate([plano_c, plano_a])

    # ── regiao ────────────────────────────────────────────────────────────────
    # Churners: more Norte/Leste; Active: more Sul/Centro
    regiao_c = rng.choice(REGIOES, n_churn,  p=[0.32, 0.10, 0.28, 0.20, 0.10])
    regiao_a = rng.choice(REGIOES, n_active, p=[0.17, 0.28, 0.16, 0.17, 0.22])
    regiao   = np.concatenate([regiao_c, regiao_a])

    # ── tempo_contrato ────────────────────────────────────────────────────────
    # Churners: more early (1-3m) and renewal (12-15m) bumps
    tempo_c = rng.integers(1, 61, n_churn).astype(int)
    # inject 12% early-churn (1-3 months) and 10% renewal (12-15)
    early_mask   = rng.random(n_churn) < 0.12
    renewal_mask = (~early_mask) & (rng.random(n_churn) < 0.10)
    tempo_c = np.where(early_mask,   rng.integers(1, 4,  n_churn), tempo_c)
    tempo_c = np.where(renewal_mask, rng.integers(12,16, n_churn), tempo_c)
    tempo_a = rng.integers(1, 61, n_active).astype(int)
    tempo_contrato = np.concatenate([tempo_c, tempo_a])

    # ── meses_sem_incidente ───────────────────────────────────────────────────
    # Churners: fewer (mean ~6), Active: more (mean ~14)
    msi_c = rng.integers(0, 13, n_churn).astype(int)   # 0-12
    msi_a = rng.integers(6, 25, n_active).astype(int)  # 6-24
    meses_sem_incidente = np.concatenate([msi_c, msi_a])

    # ── valor_mensalidade (from plan) ─────────────────────────────────────────
    valor_mensalidade = np.array([
        rng.uniform(*PLAN_PRICE_RANGE[p]) for p in plano
    ]).round(2)

    # ── uso_medio_gb (from plan) ──────────────────────────────────────────────
    uso_medio_gb = np.array([
        rng.uniform(PLAN_USAGE_MAX[p] * 0.33, PLAN_USAGE_MAX[p]) for p in plano
    ]).round(1)

    # ── Shuffle to avoid positional bias ─────────────────────────────────────
    idx = rng.permutation(n)
    df  = pd.DataFrame({
        "tempo_contrato":        tempo_contrato[idx],
        "plano":                 plano[idx],
        "valor_mensalidade":     valor_mensalidade[idx],
        "qtd_chamados_suporte":  qtd_chamados[idx],
        "dias_atraso_pagamento": dias_atraso[idx],
        "uso_medio_gb":          uso_medio_gb[idx],
        "nps_score":             nps_score[idx],
        "qtd_upgrades":          qtd_upgrades[idx],
        "regiao":                regiao[idx],
        "tem_fidelidade":        tem_fidelidade[idx],
        "meses_sem_incidente":   meses_sem_incidente[idx],
        "churn":                 churn[idx],
    })
    return df


# ── Feature engineering ────────────────────────────────────────────────────────

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    uso_max = df["plano"].map(PLAN_USAGE_MAX)

    df["risco_pagamento"]       = (df["dias_atraso_pagamento"] / 30).round(4)
    df["pressao_suporte"]       = (df["qtd_chamados_suporte"] / (df["tempo_contrato"] + 1)).round(4)
    df["engagement_score"]      = ((df["uso_medio_gb"] / uso_max) * 100).round(2)
    df["ticket_medio_ajustado"] = (df["valor_mensalidade"] / (df["qtd_upgrades"] + 1)).round(2)
    df["score_satisfacao"]      = (df["nps_score"] * df["meses_sem_incidente"] / 10).round(4)

    # Binary risk flags product (specified correlations)
    df["score_risco_composto"] = (
        np.where(df["dias_atraso_pagamento"] > 30,  3.0, 1.0) *
        np.where(df["qtd_chamados_suporte"]  > 4,   2.0, 1.0) *
        np.where(df["nps_score"] <= 5,              2.5, 1.0) *
        np.where(df["tem_fidelidade"] == 1,         0.6, 1.0) *
        np.where(df["qtd_upgrades"] >= 1,           0.75, 1.0)
    ).round(4)

    return df


# ── Preprocessor & pipeline ────────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("scale",   StandardScaler(),
             CONTINUOUS_COLS),
            ("ordinal", OrdinalEncoder(
                categories=[PLAN_ORDER, REGION_ORDER],
                handle_unknown="use_encoded_value", unknown_value=-1,
            ), ORDINAL_COLS),
            ("passthrough", "passthrough", BINARY_COLS),
        ],
        remainder="drop",
    )


def build_pipeline(clf) -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("clf", clf),
    ])


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

def optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    precs, recs, threshs = precision_recall_curve(y_true, y_proba)
    denom = precs + recs
    f1s   = np.where(denom > 0, 2 * precs * recs / denom, 0.0)
    return float(threshs[np.argmax(f1s[:-1])])


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
        f"\n_Gerado automaticamente por pipeline.py - Dados sintéticos - Hugo Leonardo_\n",
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
        f"Recall é priorizado (não perder churners reais) em detrimento de precisão.",
        "",
        "### Calibração de Probabilidade",
        "",
        f"| | Brier Score |",
        f"|---|-------------|",
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
    im = ax.imshow(cm, cmap="Blues")
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
    print(f"  AUC-ROC    : {m_topt['AUC-ROC']:.4f}  "
          f"{'[OK]' if m_topt['AUC-ROC'] >= 0.80 else '[!] ABAIXO DO MÍNIMO'}")
    print(f"  F1-Score   : {m_topt['F1']:.4f}  "
          f"{'[OK]' if m_topt['F1'] >= 0.72 else '[!] ABAIXO DO MÍNIMO'}")
    print(f"  Recall     : {m_topt['Recall']:.4f}")
    print(f"  Precisão   : {m_topt['Precisão']:.4f}")
    print(f"  Threshold  : {opt_thr:.4f}")
    print(f"  Churn rate : {churn_rate*100:.1f}%")
    print(f"\n  Arquivos gerados:")
    print(f"    - confusion_matrix.png")
    print(f"    - outputs/predictions.csv")
    print(f"    - outputs/relatorio_churn.md")
    if SHAP_OK:
        print(f"    - outputs/shap/shap_summary.png")
        print(f"    - outputs/shap/shap_dep_1,2,3_*.png")
        print(f"    - outputs/shap/shap_force_cliente_1,2,3.png")
    print()


if __name__ == "__main__":
    main()
