"""
Dados e pré-processamento do Churn Predictor — FiberNet ISP.

Fonte única do dataset sintético e das features. `pipeline.py` (treino,
comparação de modelos, SHAP, relatório) e `app.py` (demo Streamlit) importam
daqui — nenhum dos dois tem gerador próprio.

Por que o módulo existe: o app tinha o seu próprio gerador, com as variáveis
sorteadas condicionadas ao rótulo `churn`. O pipeline já tinha sido corrigido
disso, mas a cópia do app não — e era a cópia do app que ia para o ar. O demo
mostrava AUC 0,92 enquanto o repositório documentava 0,785 e o teste unitário
reprovava exatamente acima de 0,92. Registro paralelo mantido à mão: o remédio
é derivar, não copiar.
"""

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import precision_recall_curve

# ── Constants ──────────────────────────────────────────────────────────────────

SEED    = 42

# Parcela de clientes cujo desfecho contraria o comportamento observado.
# É o teto de separabilidade do dataset — ver a docstring de build_dataset.
RUIDO_COMPORTAMENTAL = 0.15
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
    Gera dataset ISP com geração condicional ao PERFIL de comportamento.

    Correlações garantidas por design (sobre o perfil):
      dias_atraso > 30  -> P(churn|atrasado)/P(churn|normal)  >= 3x
      nps_score <= 5    -> P(churn|nps<=5)/P(churn|nps>5)     >= 2.5x
      tem_fidelidade    -> P(churn|fid)/P(churn|sem_fid)      <= 0.40
      qtd_upgrades >= 1 -> P(churn|upgrade)/P(churn|sem)      <= 0.75

    O rótulo NÃO é o perfil. Uma versão anterior gerava cada variável
    condicionada ao próprio `churn`, o que produzia AUC 0,996 — e a regressão
    logística fazia 0,9946. Quando o modelo mais simples empata com o mais
    sofisticado em 0,99, o que está fácil é o problema, não o algoritmo: o
    dataset media o gerador, não o modelo, e o SHAP só redescobria as regras
    que o próprio script tinha escrito.

    `RUIDO_COMPORTAMENTAL` é a parcela de clientes cujo desfecho contraria o
    comportamento observado — quem cancela sem nenhum sinal prévio e quem
    acumula todos os sinais e fica. Isso existe em base real: as variáveis
    disponíveis não explicam tudo, e um teto de acerto é o que separa um
    exercício de modelagem de um exercício de decoração. O valor está calibrado
    para AUC de teste na faixa de 0,80–0,86, que é onde vive um bom modelo de
    churn com dado de verdade.
    """
    rng = np.random.default_rng(seed)

    # A proporção de perfis é resolvida de trás para frente para o churn final
    # continuar em ~30%: p*(1-e) + (1-p)*e = 0.30.
    e = RUIDO_COMPORTAMENTAL
    p_perfil = (0.30 - e) / (1 - 2 * e)
    n_churn  = int(n * p_perfil)      # bloco gerado com comportamento de churn
    n_active = n - n_churn
    perfil   = np.array([1] * n_churn + [0] * n_active)
    troca    = rng.random(n) < e
    churn    = np.where(troca, 1 - perfil, perfil)

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



# ── Threshold ──────────────────────────────────────────────────────────────────

def optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Corte que maximiza F1. Usado pelo pipeline e pelo app — 0,5 é arbitrário
    numa base com 30% de churn e joga o recall para baixo."""
    precs, recs, threshs = precision_recall_curve(y_true, y_proba)
    denom = precs + recs
    f1s   = np.where(denom > 0, 2 * precs * recs / denom, 0.0)
    return float(threshs[np.argmax(f1s[:-1])])
