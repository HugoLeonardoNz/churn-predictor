"""
Testes de sanidade do pipeline de churn — FiberNet ISP
Execute com: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.utils.validation import check_is_fitted

from pipeline import (
    build_dataset,
    add_derived_features,
    build_pipeline,
    get_models,
    optimal_threshold,
    ALL_FEATURES,
    CONTINUOUS_COLS, ORDINAL_COLS, BINARY_COLS,
    SEED,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def df():
    return build_dataset(n=15_000, seed=SEED)


@pytest.fixture(scope="module")
def df_feat(df):
    return add_derived_features(df)


@pytest.fixture(scope="module")
def trained(df_feat):
    X = df_feat[ALL_FEATURES]
    y = df_feat["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    # Use RandomForest for speed in CI
    models = get_models()
    pipe = build_pipeline(models["RandomForest"])
    pipe.fit(X_train, y_train)

    y_proba = pipe.predict_proba(X_test)[:, 1]
    auc_roc = roc_auc_score(y_test, y_proba)

    return {
        "pipeline": pipe,
        "X_train":  X_train,
        "X_test":   X_test,
        "y_train":  y_train,
        "y_test":   y_test,
        "y_proba":  y_proba,
        "auc_roc":  auc_roc,
    }


# ── Testes de qualidade de dados ──────────────────────────────────────────────

def test_n_registros(df):
    assert len(df) >= 15_000, f"Dataset tem apenas {len(df)} registros (mínimo: 15.000)"


def test_taxa_churn_realista(df):
    taxa = df["churn"].mean()
    assert 0.25 <= taxa <= 0.40, (
        f"Churn rate {taxa:.3f} fora do benchmark [0.25, 0.40]. "
        "Verificar lógica de geração de dados."
    )


def test_features_obrigatorias(df):
    required = [
        "tempo_contrato", "plano", "valor_mensalidade",
        "qtd_chamados_suporte", "dias_atraso_pagamento", "uso_medio_gb",
        "nps_score", "qtd_upgrades", "regiao", "tem_fidelidade",
        "meses_sem_incidente",
    ]
    for feat in required:
        assert feat in df.columns, f"Feature obrigatória ausente: {feat}"


def test_features_sem_nan(df_feat):
    nulls = df_feat[ALL_FEATURES].isnull().sum().sum()
    assert nulls == 0, f"{nulls} valores nulos encontrados nas features"


def test_features_derivadas_existem(df_feat):
    derived = [
        "risco_pagamento", "pressao_suporte", "engagement_score",
        "ticket_medio_ajustado", "score_satisfacao",
    ]
    for feat in derived:
        assert feat in df_feat.columns, f"Feature derivada ausente: {feat}"


def test_engagement_score_range(df_feat):
    assert df_feat["engagement_score"].between(0, 100.01).all(), \
        "engagement_score fora do range [0, 100]"


def test_risco_pagamento_positivo(df_feat):
    assert (df_feat["risco_pagamento"] >= 0).all(), \
        "risco_pagamento contém valores negativos"


# ── Testes de correlações especificadas ───────────────────────────────────────

def test_correlacao_atraso_churn(df):
    grupo_atrasado = df[df["dias_atraso_pagamento"] > 30]["churn"].mean()
    grupo_normal   = df[df["dias_atraso_pagamento"] <= 30]["churn"].mean()
    assert grupo_atrasado > grupo_normal * 2.0, (
        f"Correlação atraso→churn insuficiente: "
        f"atrasado={grupo_atrasado:.3f} vs 2×normal={grupo_normal*2:.3f}"
    )


def test_correlacao_nps_churn(df):
    churn_baixo_nps = df[df["nps_score"] <= 5]["churn"].mean()
    churn_alto_nps  = df[df["nps_score"] > 5]["churn"].mean()
    assert churn_baixo_nps > churn_alto_nps * 1.5, (
        f"Correlação NPS→churn insuficiente: "
        f"NPS≤5={churn_baixo_nps:.3f} vs 1.5×NPS>5={churn_alto_nps*1.5:.3f}"
    )


def test_fidelidade_reduz_churn(df):
    churn_fidelidade = df[df["tem_fidelidade"] == 1]["churn"].mean()
    churn_sem        = df[df["tem_fidelidade"] == 0]["churn"].mean()
    assert churn_fidelidade < churn_sem * 0.80, (
        f"Fidelidade não reduz churn suficientemente: "
        f"com={churn_fidelidade:.3f} vs 0.8×sem={churn_sem*0.80:.3f}"
    )


def test_upgrades_reduz_churn(df):
    churn_upgrade  = df[df["qtd_upgrades"] >= 1]["churn"].mean()
    churn_no_upgrade = df[df["qtd_upgrades"] == 0]["churn"].mean()
    assert churn_upgrade < churn_no_upgrade, (
        "Upgrades não reduzem churn como esperado"
    )


# ── Testes de modelo e pipeline ───────────────────────────────────────────────

def test_auc_na_faixa_esperada(trained):
    """AUC tem piso E teto.

    O piso protege contra o modelo nao aprender. O teto protege contra o erro
    que este projeto ja cometeu: gerar as variaveis condicionadas ao proprio
    rotulo, chegar a 0,996 e achar que isso era resultado. Com 15% de desfechos
    que contrariam o comportamento observado, AUC acima de 0,92 aqui significa
    vazamento de informacao, nao um modelo melhor.
    """
    auc_roc = trained["auc_roc"]
    assert auc_roc >= 0.72, f"AUC-ROC abaixo do piso 0.72: {auc_roc:.4f}"
    assert auc_roc <= 0.92, (
        f"AUC-ROC acima do teto 0.92 ({auc_roc:.4f}) — investigar vazamento "
        f"antes de comemorar"
    )


def test_sem_data_leakage(trained):
    X_train = trained["X_train"]
    assert "churn" not in X_train.columns, \
        "Data leakage detectado: coluna 'churn' está em X_train"


def test_pipeline_sem_fit_no_test(trained):
    pipe = trained["pipeline"]
    # check_is_fitted raises NotFittedError if not fitted
    check_is_fitted(pipe.named_steps["clf"])
    check_is_fitted(pipe.named_steps["preprocessor"])


def test_proba_range(trained):
    y_proba = trained["y_proba"]
    assert (y_proba >= 0).all() and (y_proba <= 1).all(), \
        "Probabilidades fora do range [0, 1]"


def test_optimal_threshold_range(trained):
    thr = optimal_threshold(trained["y_test"], trained["y_proba"])
    assert 0.0 < thr < 1.0, f"Threshold ótimo inválido: {thr}"


def test_feature_count_in_pipeline(trained):
    pipe  = trained["pipeline"]
    X_tr  = trained["X_train"].head(5)
    proba = pipe.predict_proba(X_tr)
    assert proba.shape == (5, 2), \
        f"Pipeline retornou shape inesperado: {proba.shape}"


def test_features_match_definition(df_feat):
    all_defined = CONTINUOUS_COLS + ORDINAL_COLS + BINARY_COLS
    for feat in all_defined:
        assert feat in df_feat.columns, f"Feature definida mas ausente no DataFrame: {feat}"
    assert len(all_defined) == len(set(all_defined)), \
        "Colunas duplicadas em ALL_FEATURES"
