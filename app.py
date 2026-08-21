import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
import plotly.graph_objects as go

# Dataset, features e pre-processamento vem do mesmo modulo que o pipeline de
# treino usa. Nao importar `pipeline` aqui de proposito: ele carrega xgboost,
# lightgbm e shap, que o app nao usa e que o deploy nao precisa instalar.
from churn_data import (
    SEED, PLANOS, REGIOES, REGION_ORDER, PLAN_PRICE_RANGE, PLAN_USAGE_MAX,
    ALL_FEATURES, build_dataset, add_derived_features, build_pipeline,
    optimal_threshold,
)

st.set_page_config(
    page_title="Churn Predictor · FiberNet",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp { background-color: #060912 !important; font-family: 'Inter', sans-serif !important; }
.block-container { padding: 2rem 2.5rem !important; max-width: 1280px !important; }
[data-testid="stSidebar"] { background: #0a0e1a !important; border-right: 1px solid rgba(99,102,241,0.15) !important; }

/* Escala de arredondamento — um degrau por nivel de superficie, e nada fora
   dela. E a mesma escala dos relatorios Power BI do portfolio: paleta e
   tipografia separam as pecas, o acabamento as une. Antes eram seis raios
   escolhidos um a um (14, 12, 10, 8, 3, 999). */
:root {
    --r-chip:  10px;   /* selo, tag, barra de progresso */
    --r-ctrl:  14px;   /* campo, botao, aba */
    --r-panel: 20px;   /* cartao e painel */
}
h1 { color: #f0f2f8 !important; font-size: 26px !important; letter-spacing: -0.5px !important; font-weight: 700 !important; }
h2 { color: #cbd5e1 !important; font-size: 16px !important; font-weight: 600 !important; }
h3 { color: #94a3b8 !important; font-size: 13px !important; font-weight: 500 !important; }
p, li { color: #8b92a5 !important; font-size: 13px !important; }
hr { border-color: rgba(99,102,241,0.1) !important; margin: 12px 0 !important; }

.stat-card {
    background: rgba(13,17,30,0.9);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: var(--r-panel);
    padding: 18px 22px;
}
.stat-label { font-size: 10px; color: #4b5468; letter-spacing: 0.12em; text-transform: uppercase; font-family: 'JetBrains Mono', monospace; margin-bottom: 6px; }
.stat-value { font-size: 28px; font-weight: 700; font-family: 'JetBrains Mono', monospace; line-height: 1.1; }
.stat-sub   { font-size: 11px; color: #4b5468; margin-top: 5px; }

.risk-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: var(--r-chip);
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.08em;
}

.action-box {
    background: rgba(13,17,30,0.9);
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: var(--r-panel);
    padding: 16px 20px;
    margin-top: 16px;
}
.action-title { font-size: 10px; color: #4b5468; letter-spacing: 0.2em; text-transform: uppercase; font-family: 'JetBrains Mono', monospace; margin-bottom: 10px; }
.action-item { display: flex; align-items: flex-start; gap: 10px; margin-bottom: 8px; font-size: 12px; color: #8b92a5; line-height: 1.5; }

.section-label { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #6366f1; letter-spacing: 0.3em; text-transform: uppercase; margin-bottom: 12px; }

.stTabs [data-baseweb="tab-list"] {
    background: rgba(13,17,30,0.8) !important;
    border: 1px solid rgba(99,102,241,0.12) !important;
    border-radius: var(--r-panel) !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] { color: #8b92a5 !important; border-radius: var(--r-ctrl) !important; font-size: 13px !important; padding: 0.5rem 1.1rem !important; }
.stTabs [aria-selected="true"] { color: #a5b4fc !important; background: rgba(99,102,241,0.15) !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 20px !important; }

.stSlider label { color: #8b92a5 !important; font-size: 12px !important; }
.stSelectbox label { color: #8b92a5 !important; font-size: 12px !important; }
[data-testid="stDataFrame"] { border: 1px solid rgba(99,102,241,0.15) !important; border-radius: var(--r-panel) !important; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: var(--r-chip); }

.stDownloadButton button {
    background: rgba(99,102,241,0.12) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    color: #a5b4fc !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    border-radius: var(--r-ctrl) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────

COLORS = {
    "indigo": "#6366f1", "cyan": "#22d3ee", "green": "#10b981",
    "amber": "#f59e0b", "red": "#ef4444", "muted": "#8b92a5",
}

# Cor fixa por plano, usada em todo grafico que quebra por plano. Antes, a lista
# de cores era aplicada por POSICAO: o mesmo plano saia ambar num grafico e roxo
# no outro, ao lado. Cor amarrada ao valor resolve isso e ainda carrega sentido —
# do plano mais barato (vermelho, mais churn) ao mais caro (verde).
# PLANOS ja vem em ordem de preco, nao alfabetica: ordenado por texto, a escada
# que a leitura promete — quanto maior o plano, menor o churn — quebra no eixo.
PLAN_COLORS = {
    "Fibra 100MB": COLORS["red"],
    "Fibra 200MB": COLORS["amber"],
    "Fibra 500MB": COLORS["cyan"],
    "Empresarial": COLORS["green"],
}

# Nome de exibicao das features do modelo. A ordem de ALL_FEATURES e a que sai do
# ColumnTransformer, entao o mapa e por nome, nao por posicao.
FEATURE_LABELS = {
    "valor_mensalidade":     "Mensalidade (R$)",
    "tempo_contrato":        "Tempo de contrato",
    "qtd_chamados_suporte":  "Chamados de suporte",
    "dias_atraso_pagamento": "Dias de atraso",
    "uso_medio_gb":          "Uso médio (GB)",
    "nps_score":             "NPS",
    "meses_sem_incidente":   "Meses sem incidente",
    "risco_pagamento":       "Risco de pagamento",
    "pressao_suporte":       "Pressão de suporte",
    "engagement_score":      "Engajamento (% do plano)",
    "ticket_medio_ajustado": "Ticket médio ajustado",
    "score_satisfacao":      "Score de satisfação",
    "score_risco_composto":  "Score de risco composto",
    "plano":                 "Plano",
    "regiao":                "Região",
    "tem_fidelidade":        "Tem fidelidade",
    "qtd_upgrades":          "Upgrades",
}

# Colunas cruas do gerador — as que o simulador expoe. As derivadas saem de
# add_derived_features, nunca digitadas a mao.
RAW_COLS = [
    "tempo_contrato", "plano", "valor_mensalidade", "qtd_chamados_suporte",
    "dias_atraso_pagamento", "uso_medio_gb", "nps_score", "qtd_upgrades",
    "regiao", "tem_fidelidade", "meses_sem_incidente",
]

N_BASE = 15_000   # mesma base do pipeline.py e dos testes

ACTIONS = {
    "BAIXO": [
        ("✅", "Cliente saudável. Manter ciclo padrão de NPS e check-in semestral."),
        ("📩", "Enviar e-mail de relacionamento com novidades de planos e vantagens."),
        ("📊", "Monitorar indicadores mensalmente — sem ação comercial imediata."),
    ],
    "MÉDIO": [
        ("📞", "Acionar time comercial para contato proativo nos próximos 15 dias."),
        ("🎁", "Oferecer upgrade de plano com desconto de adesão ou mês cortesia."),
        ("🔧", "Verificar chamados abertos e garantir resolução dentro do SLA."),
        ("💬", "Aplicar NPS + pesquisa de satisfação — identificar ponto de atrito."),
    ],
    "ALTO": [
        ("🚨", "PRIORIDADE CRÍTICA: contato imediato do supervisor comercial (48h)."),
        ("💰", "Oferecer cashback ou desconto de 20-30% por fidelidade de 12 meses."),
        ("🔄", "Renegociar plano com custo menor — reter com ticket reduzido > perder."),
        ("📋", "Escalar para equipe de retenção com histórico completo do contrato."),
        ("⏰", "Definir prazo de resposta do cliente: máximo 72h antes de marcar como saída."),
    ],
}

# ── Data & model ──────────────────────────────────────────────────────────────

def _rf():
    return RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_split=5,
        class_weight="balanced", random_state=SEED, n_jobs=-1,
    )


def _typical_row(df):
    """Cliente mediano da base — referencia do simulador para dizer o quanto cada
    sinal deste contrato afasta ele do normal."""
    return {
        c: (df[c].mode().iloc[0] if df[c].dtype == object else df[c].median())
        for c in RAW_COLS
    }


@st.cache_resource(show_spinner="Gerando a base e treinando o modelo…")
def train():
    """Treina o mesmo modelo do pipeline.py, sobre o mesmo gerador.

    O score que aparece nas abas 1 e 3 e out-of-fold: a probabilidade de cada
    contrato vem do fold em que ele estava FORA do treino. Probabilidade tirada
    do modelo que ja viu a linha e otimista, e era ela que alimentava a lista de
    prioridade — o painel prometia um acerto que so existia dentro da amostra.
    """
    df = add_derived_features(build_dataset(n=N_BASE, seed=SEED))
    X, y = df[ALL_FEATURES], df["churn"]

    oof   = np.zeros(len(df))
    folds = []
    for tr_idx, te_idx in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        pipe = build_pipeline(_rf()).fit(X.iloc[tr_idx], y.iloc[tr_idx])
        p = pipe.predict_proba(X.iloc[te_idx])[:, 1]
        oof[te_idx] = p
        folds.append(roc_auc_score(y.iloc[te_idx], p))

    auc  = roc_auc_score(y, oof)
    thr  = optimal_threshold(y.values, oof)
    pred = (oof >= thr).astype(int)

    # Modelo servido ao simulador: treinado na base inteira, ja que aqui nao ha
    # avaliacao — e a avaliacao acima que nao pode ver o proprio dado.
    model = build_pipeline(_rf()).fit(X, y)

    imps = pd.Series(
        model.named_steps["clf"].feature_importances_,
        index=[FEATURE_LABELS[c] for c in ALL_FEATURES],
    ).sort_values(ascending=True)

    return {
        "model": model, "df": df, "X": X, "y": y,
        "oof": oof, "pred": pred, "thr": thr, "auc": auc,
        "folds": np.array(folds),
        "cm": confusion_matrix(y, pred),
        "report": classification_report(y, pred, target_names=["Ativo", "Churn"], output_dict=True),
        "importances": imps,
        "base_row": _typical_row(df),
    }


def predict_raw(model, raw: dict) -> float:
    df = add_derived_features(pd.DataFrame([raw]))
    return float(model.predict_proba(df[ALL_FEATURES])[0, 1])


def local_drivers(model, raw: dict, base: dict, prob: float, top=3):
    """Quanto da probabilidade deste cliente vem de cada sinal dele.

    Para cada variavel, troca o valor pelo do cliente mediano e mede o quanto a
    probabilidade cai. E uma explicacao deste contrato, nao a importancia global
    do modelo. A versao anterior multiplicava importancia global pelo valor bruto
    da feature — somava reais com contagem de chamados e exibia o resultado como
    porcentagem.
    """
    out = []
    for c in RAW_COLS:
        if raw[c] == base[c]:
            continue
        alt = dict(raw)
        alt[c] = base[c]
        out.append((FEATURE_LABELS[c], prob - predict_raw(model, alt)))
    out.sort(key=lambda t: t[1], reverse=True)
    return [t for t in out if t[1] > 0][:top]


def tier_of(p):
    return "BAIXO" if p < 0.30 else "MÉDIO" if p < 0.60 else "ALTO"

# ── Chart helpers ─────────────────────────────────────────────────────────────

def dark(fig, title="", height=320):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["muted"], family="Inter, sans-serif", size=11),
        title=dict(text=title, font=dict(color="#f0f2f8", size=13)),
        margin=dict(l=12, r=12, t=44 if title else 12, b=12),
        height=height,
        xaxis=dict(gridcolor="rgba(99,102,241,0.08)", zerolinecolor="rgba(99,102,241,0.12)",
                   tickfont=dict(color=COLORS["muted"])),
        yaxis=dict(gridcolor="rgba(99,102,241,0.08)", zerolinecolor="rgba(99,102,241,0.12)",
                   tickfont=dict(color=COLORS["muted"])),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["muted"])),
    )
    return fig


def stat(label, value, sub, color):
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">{label}</div>
        <div class="stat-value" style="color:{color}">{value}</div>
        <div class="stat-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


def br(n, dec=0):
    """Numero no formato brasileiro — ponto de milhar."""
    return f"{n:,.{dec}f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def brl(v):
    return "R$ " + br(v)


def gauge(prob):
    color = COLORS["green"] if prob < 0.3 else COLORS["amber"] if prob < 0.6 else COLORS["red"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"color": "#f0f2f8", "size": 42, "family": "JetBrains Mono"}},
        title={"text": "Probabilidade de Churn", "font": {"color": "#8b92a5", "size": 13}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"color": "#4b5468"}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": [
                {"range": [0,  30], "color": "rgba(16,185,129,0.12)"},
                {"range": [30, 60], "color": "rgba(245,158,11,0.12)"},
                {"range": [60, 100], "color": "rgba(239,68,68,0.12)"},
            ],
        },
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=220, margin=dict(l=20, r=20, t=40, b=0))
    return fig, color

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    md = train()

    st.markdown("# Customer Churn Predictor")
    st.markdown(
        "<p style='margin-top:-8px;margin-bottom:0'>FiberNet ISP · RandomForest + Feature Engineering · "
        "<span style='color:#6366f1;font-family:monospace'>portfólio Hugo Leonardo</span></p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='margin-top:8px;font-size:12px;color:#6b7280;line-height:1.6'>Base sintética de "
        f"{br(N_BASE)} contratos, gerada por <code>churn_data.py</code> — o mesmo módulo que alimenta "
        "o pipeline de treino e os testes. 15% dos clientes têm desfecho que contraria o comportamento "
        "observado, o que põe um teto na separabilidade.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Análise de Risco",
        "Simulador de Contratos",
        "Plano de Retenção",
        "Desempenho do Modelo",
    ])

    full_df = md["df"].copy()
    full_df["prob"] = md["oof"]
    full_df["tier"] = pd.Categorical(
        [tier_of(p) for p in md["oof"]], categories=["BAIXO", "MÉDIO", "ALTO"]
    )

    alto  = full_df[full_df["tier"] == "ALTO"]
    medio = full_df[full_df["tier"] == "MÉDIO"]
    baixo = full_df[full_df["tier"] == "BAIXO"]
    mrr_alto  = alto["valor_mensalidade"].sum()
    mrr_medio = medio["valor_mensalidade"].sum()
    mrr_total = full_df["valor_mensalidade"].sum()

    # ── Tab 1: Análise de Risco ───────────────────────────────────────────────
    with tab1:
        st.markdown(
            f'<p class="section-label">Exposição de Receita · Base de {br(N_BASE)} Contratos</p>',
            unsafe_allow_html=True)
        b1, b2, b3, b4 = st.columns(4)
        with b1: stat("MRR em Risco Crítico",  brl(mrr_alto),  f"{br(len(alto))} contratos — ação imediata", COLORS["red"])
        with b2: stat("MRR em Risco Médio",    brl(mrr_medio), f"{br(len(medio))} contratos — monitoramento ativo", COLORS["amber"])
        with b3: stat("Base Saudável",         br(len(baixo)), f"contratos · {brl(baixo['valor_mensalidade'].sum())}/mês estável", COLORS["green"])
        with b4: stat("Potencial de Retenção", brl(mrr_alto * 0.5), "projetando 50% retenção dos críticos", COLORS["indigo"])

        # ── Insight callout ──
        st.markdown("<br>", unsafe_allow_html=True)
        pct_mrr_risco   = (mrr_alto + mrr_medio) / mrr_total * 100
        churn_por_plano = full_df.groupby("plano")["churn"].mean() * 100
        pior, melhor    = PLANOS[0], PLANOS[-1]
        razao           = churn_por_plano[pior] / churn_por_plano[melhor]
        st.markdown(f"""
        <div style='background:rgba(239,68,68,0.06);border:1px solid rgba(239,68,68,0.2);border-radius:var(--r-panel);padding:16px 20px;margin-bottom:24px'>
          <div style='font-size:10px;color:#ef4444;letter-spacing:0.2em;font-family:monospace;text-transform:uppercase;margin-bottom:8px'>⚠ Alerta de Risco · Base Atual</div>
          <div style='font-size:13px;color:#c8d0db;line-height:1.7'>
            <b style='color:#f8fafc'>{pct_mrr_risco:.0f}% do MRR total</b> está em zona de risco (ALTO + MÉDIO).
            O plano <b style='color:#ef4444'>{pior} concentra {churn_por_plano[pior]:.0f}% de churn</b> —
            {razao:.1f}x o do {melhor} ({churn_por_plano[melhor]:.0f}%). Atraso acima de 30 dias e NPS até 5
            são os dois sinais que mais deslocam a probabilidade individual.
          </div>
        </div>""", unsafe_allow_html=True)

        # ── Churn por plano + por região ──
        st.markdown('<p class="section-label">Padrão de Cancelamento por Segmento</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            v = churn_por_plano.reindex(PLANOS)
            fig = go.Figure(go.Bar(
                x=list(v.index), y=v.values,
                marker=dict(color=[PLAN_COLORS[p] for p in v.index]),
                text=[f"{x:.1f}%" for x in v.values],
                textposition="outside", textfont=dict(color=COLORS["muted"]),
            ))
            dark(fig, "Taxa de Churn por Plano (%)", height=280)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Correlação inversa entre valor do plano e churn — clientes de menor ticket concentram maior risco de evasão.")

        with col2:
            r = full_df.groupby("regiao")["churn"].mean().reindex(REGION_ORDER) * 100
            fig2 = go.Figure(go.Bar(
                x=list(r.index), y=r.values,
                marker=dict(color=COLORS["indigo"]),
                text=[f"{x:.1f}%" for x in r.values],
                textposition="outside", textfont=dict(color=COLORS["muted"]),
            ))
            dark(fig2, "Taxa de Churn por Região (%)", height=280)
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("Regiões em ordem de risco, não alfabética — a ordem do eixo é parte da leitura.")

        # ── Atraso x NPS ──
        # Era um scatter de 2.500 pontos: com dias de atraso inteiro e NPS em
        # passo de 0,5, os pontos caem todos nas mesmas coordenadas e viram uma
        # mancha — nao da para ler densidade em cima de sobreposicao. A taxa de
        # churn por faixa responde a mesma pergunta e cabe em 20 celulas.
        st.markdown('<p class="section-label">Onde o Churn Acontece · Atraso de Pagamento vs NPS</p>', unsafe_allow_html=True)
        faixa_atraso = pd.cut(full_df["dias_atraso_pagamento"],
                              bins=[-1, 5, 15, 30, 60, 90],
                              labels=["0–5 dias", "6–15", "16–30", "31–60", "61–90"])
        faixa_nps    = pd.cut(full_df["nps_score"],
                              bins=[-0.1, 2, 5, 8, 10],
                              labels=["NPS 0–2", "NPS 3–5", "NPS 6–8", "NPS 9–10"])
        matriz = (full_df.assign(fa=faixa_atraso, fn=faixa_nps)
                  .pivot_table(index="fn", columns="fa", values="churn",
                               aggfunc="mean", observed=False) * 100)
        contagem = (full_df.assign(fa=faixa_atraso, fn=faixa_nps)
                    .pivot_table(index="fn", columns="fa", values="churn",
                                 aggfunc="size", observed=False))
        fig3 = go.Figure(go.Heatmap(
            z=matriz.values, x=list(matriz.columns), y=list(matriz.index),
            colorscale=[[0, "#0d1117"], [0.5, "rgba(239,68,68,0.45)"], [1, "#ef4444"]],
            text=[[f"{v:.0f}%" for v in row] for row in matriz.values],
            texttemplate="%{text}",
            textfont=dict(color="#f0f2f8", size=13),
            customdata=contagem.values,
            hovertemplate="%{y} · atraso %{x}<br>Churn: %{z:.1f}%<br>%{customdata} contratos<extra></extra>",
            colorbar=dict(title=dict(text="% churn", font=dict(color=COLORS["muted"], size=10)),
                          tickfont=dict(color=COLORS["muted"], size=10), thickness=12),
        ))
        dark(fig3, "Taxa de churn por faixa — base completa", height=300)
        st.plotly_chart(fig3, use_container_width=True)
        pior_nps = matriz.loc["NPS 0–2", "0–5 dias"]
        pior_atr = matriz.loc["NPS 9–10", "61–90"]
        st.caption(
            f"Cada célula é a taxa de churn dos contratos daquela faixa, sobre os {br(N_BASE)} da base. "
            f"Os dois sinais não pesam igual: NPS até 2 sozinho, com pagamento em dia, já dá "
            f"{pior_nps:.0f}% de churn, enquanto atraso de 61–90 dias com NPS 9–10 fica em {pior_atr:.0f}%. "
            "Insatisfação é o sinal que manda; o atraso empilha em cima dele."
        )

        # ── Desempenho do modelo (rodapé técnico) ──
        st.markdown("<hr style='border-color:rgba(139,92,246,0.1);margin:28px 0'>", unsafe_allow_html=True)
        st.markdown('<p class="section-label">Precisão do Modelo · Validação Técnica</p>', unsafe_allow_html=True)
        st.caption("Métricas out-of-fold: cada contrato é avaliado pelo modelo que não o viu no treino. Detalhe na aba Desempenho do Modelo.")
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1: stat("AUC-ROC",        f"{md['auc']:.3f}",          "out-of-fold, base completa",          COLORS["indigo"])
        with c2: stat("AUC por fold",   f"{md['folds'].mean():.3f}", f"± {md['folds'].std():.3f} · 5-fold", COLORS["cyan"])
        with c3: stat("Precisão Churn", f"{md['report']['Churn']['precision']:.0%}", f"dos alertas acertam · corte {md['thr']:.2f}", COLORS["amber"])
        with c4: stat("Recall Churn",   f"{md['report']['Churn']['recall']:.0%}",    "dos churns são detectados", COLORS["green"])

    # ── Tab 2: Simulador de Contratos ─────────────────────────────────────────
    with tab2:
        st.markdown("#### Simule o perfil de um cliente e veja a probabilidade de churn em tempo real.")
        st.caption(
            "Só as variáveis observadas são digitadas. As derivadas — risco de pagamento, pressão de "
            "suporte, engajamento, score composto — saem de `add_derived_features`, a mesma conta do treino."
        )
        st.markdown("<br>", unsafe_allow_html=True)

        col_form, col_result = st.columns([1, 1])

        with col_form:
            st.markdown('<p class="section-label">Perfil do Contrato</p>', unsafe_allow_html=True)
            f1, f2 = st.columns(2)
            with f1:
                p_plano  = st.selectbox("Plano", PLANOS)
                p_tempo  = st.slider("Tempo de contrato (meses)", 1, 60, 18)
                p_atraso = st.slider("Dias de atraso no pagamento", 0, 90, 0)
                p_cham   = st.slider("Chamados de suporte", 0, 12, 0)
                p_nps    = st.slider("NPS", 0.0, 10.0, 8.0, step=0.5)
            with f2:
                p_regiao = st.selectbox("Região", REGIOES)
                lo, hi   = PLAN_PRICE_RANGE[p_plano]
                p_valor  = st.slider("Mensalidade (R$)", float(lo), float(hi), float(round((lo + hi) / 2)), step=1.0)
                uso_max  = PLAN_USAGE_MAX[p_plano]
                p_uso    = st.slider("Uso médio (GB)", 0.0, float(uso_max), float(round(uso_max * 0.6)), step=5.0)
                p_msi    = st.slider("Meses sem incidente", 0, 24, 12)
                p_upg    = st.slider("Upgrades no contrato", 0, 3, 0)
            p_fid = st.checkbox("Tem fidelidade vigente", value=False)

        raw = {
            "tempo_contrato": p_tempo, "plano": p_plano, "valor_mensalidade": p_valor,
            "qtd_chamados_suporte": p_cham, "dias_atraso_pagamento": p_atraso,
            "uso_medio_gb": p_uso, "nps_score": p_nps, "qtd_upgrades": p_upg,
            "regiao": p_regiao, "tem_fidelidade": int(p_fid), "meses_sem_incidente": p_msi,
        }

        with col_result:
            prob  = predict_raw(md["model"], raw)
            level = tier_of(prob)
            color = COLORS["green"] if level == "BAIXO" else COLORS["amber"] if level == "MÉDIO" else COLORS["red"]

            fig_g, col_g = gauge(prob)
            st.plotly_chart(fig_g, use_container_width=True)

            rgb = tuple(int(col_g.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
            st.markdown(
                f"<div style='text-align:center;margin-top:-8px'>"
                f"<span class='risk-badge' style='background:rgba({rgb[0]},{rgb[1]},{rgb[2]},0.15);"
                f"color:{color};border:1px solid {color}44'>RISCO {level}</span></div>",
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="stat-card" style="text-align:center">
                <div class="stat-label">MRR em risco se churn</div>
                <div class="stat-value" style="color:{COLORS['amber']}">R$ {p_valor:.2f}/mês</div>
                <div class="stat-sub">{p_plano} · {p_regiao}</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p class="section-label" style="text-align:center">O que puxa este contrato para cima</p>', unsafe_allow_html=True)

            drivers = local_drivers(md["model"], raw, md["base_row"], prob)
            if not drivers:
                st.caption("Nenhum sinal deste contrato o afasta do cliente mediano para cima — o risco que aparece vem da base, não do perfil.")
            else:
                maior = max(d for _, d in drivers)
                for feat, delta in drivers:
                    pct = delta / maior * 100
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
                        <span style="color:#8b92a5;font-size:11px;width:150px;flex-shrink:0">{feat}</span>
                        <div style="flex:1;height:6px;background:rgba(255,255,255,0.07);border-radius:var(--r-chip)">
                            <div style="width:{pct:.0f}%;height:6px;background:{COLORS['indigo']};border-radius:var(--r-chip);opacity:0.85"></div>
                        </div>
                        <span style="color:{COLORS['indigo']};font-family:monospace;font-size:10px">+{delta*100:.1f}pp</span>
                    </div>""", unsafe_allow_html=True)
                st.caption("Quanto a probabilidade cairia se este cliente tivesse, naquele item, o valor do cliente mediano da base.")

        # Plano de Ação
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="action-box" style="border-color:{color}33">
            <div class="action-title" style="color:{color}">⚡ Plano de Ação · Risco {level}</div>
            {''.join(f'<div class="action-item"><span style="font-size:16px">{icon}</span><span>{text}</span></div>' for icon, text in ACTIONS[level])}
        </div>""", unsafe_allow_html=True)

    # ── Tab 3: Plano de Retenção ──────────────────────────────────────────────
    with tab3:
        st.markdown(
            f'<p class="section-label">Exposição por Nível de Risco · Base de {br(N_BASE)} Contratos</p>',
            unsafe_allow_html=True)
        r1, r2, r3, r4 = st.columns(4)
        with r1: stat("Risco ALTO",    br(len(alto)),  f"{brl(mrr_alto)}/mês em risco",  COLORS["red"])
        with r2: stat("Risco MÉDIO",   br(len(medio)), f"{brl(mrr_medio)}/mês em risco", COLORS["amber"])
        with r3: stat("Risco BAIXO",   br(len(baixo)), "base saudável · sem ação necessária", COLORS["green"])
        with r4: stat("Recovery Est.", brl(mrr_alto * 0.5), "projetando 50% retenção ALTO", COLORS["indigo"])

        st.markdown("<br>", unsafe_allow_html=True)

        col_a, col_b = st.columns([2, 3])
        with col_a:
            tier_counts = full_df["tier"].value_counts().reindex(["ALTO", "MÉDIO", "BAIXO"])
            fig_pie = go.Figure(go.Pie(
                labels=["ALTO", "MÉDIO", "BAIXO"],
                values=tier_counts.values,
                hole=0.5,
                marker=dict(colors=[COLORS["red"], COLORS["amber"], COLORS["green"]]),
                textfont=dict(color="#f0f2f8", size=12),
            ))
            fig_pie.update_traces(textinfo="label+percent")
            dark(fig_pie, "Distribuição de Risco · Base Completa", height=280)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_b:
            fig_bar = go.Figure()
            for tier, cor in [("ALTO", COLORS["red"]), ("MÉDIO", COLORS["amber"]), ("BAIXO", COLORS["green"])]:
                grp = (full_df[full_df["tier"] == tier]
                       .groupby("plano")["valor_mensalidade"].sum()
                       .reindex(PLANOS).fillna(0))
                fig_bar.add_trace(go.Bar(name=tier, x=list(grp.index), y=grp.values,
                                         marker_color=cor, opacity=0.85))
            fig_bar.update_layout(barmode="stack")
            dark(fig_bar, "MRR por Plano e Nível de Risco (R$/mês)", height=280)
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown('<p class="section-label">Lista de Prioridade · Contratos ALTO & MÉDIO Risco</p>', unsafe_allow_html=True)

        priority = (
            full_df[full_df["tier"].isin(["ALTO", "MÉDIO"])]
            .sort_values("prob", ascending=False)
            .reset_index(drop=True)
        )
        priority.index = range(1, len(priority) + 1)

        cols = ["regiao", "plano", "valor_mensalidade", "tempo_contrato",
                "dias_atraso_pagamento", "qtd_chamados_suporte", "nps_score", "prob", "tier"]
        display = priority[cols].head(300).copy()
        display.columns = ["Região", "Plano", "R$/mês", "Meses", "Dias Atraso",
                           "Chamados", "NPS", "Prob. Churn", "Risco"]
        display["Prob. Churn"] = display["Prob. Churn"].apply(lambda x: f"{x:.1%}")
        display["Ação"] = display["Risco"].map({
            "ALTO":  "🚨 Supervisor — 48h",
            "MÉDIO": "📞 Comercial — 15d",
        })

        st.dataframe(display, use_container_width=True, height=400)
        st.caption(
            f"{br(len(priority))} contratos em zona de risco · a tabela mostra os 300 de maior "
            "probabilidade; o CSV traz a lista inteira."
        )

        st.markdown("<br>", unsafe_allow_html=True)
        col_dl, _ = st.columns([2, 5])
        with col_dl:
            csv = priority[cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Exportar Lista de Retenção (CSV)",
                data=csv,
                file_name="fibernet_plano_retencao.csv",
                mime="text/csv",
            )

    # ── Tab 4: Desempenho do Modelo ───────────────────────────────────────────
    with tab4:
        st.caption(
            "Validação out-of-fold: as métricas vêm de 5 modelos, cada contrato avaliado por aquele "
            "que não o treinou. Mesma base e mesmo modelo do `pipeline.py` — e o teste unitário do "
            "repositório reprova AUC acima de 0,92: número alto demais aqui é sintoma de vazamento, "
            "não conquista."
        )
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            imp = md["importances"]
            fig = go.Figure(go.Bar(
                x=imp.values, y=imp.index, orientation="h",
                marker=dict(color=COLORS["indigo"]),
                text=[f"{v:.3f}" for v in imp.values],
                textposition="outside", textfont=dict(color=COLORS["muted"], size=10),
            ))
            dark(fig, "Variáveis mais relevantes para o modelo", height=460)
            # Folga a direita: sem isso o rotulo da barra mais longa e cortado
            # pela borda do grafico, que e' justamente a variavel que mais importa.
            fig.update_xaxes(range=[0, float(imp.values.max()) * 1.18])
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Importância global do RandomForest. Para o peso de cada sinal num contrato específico, veja o Simulador.")

        with col2:
            cm = md["cm"]
            fig2 = go.Figure(go.Heatmap(
                z=cm, x=["Ativo", "Churn"], y=["Ativo", "Churn"],
                colorscale=[[0, "#0d1117"], [0.5, "rgba(139,92,246,0.4)"], [1, "#8b5cf6"]],
                text=[[br(v) for v in row] for row in cm],
                texttemplate="%{text}",
                textfont=dict(color="#f0f2f8", size=20),
                showscale=False,
            ))
            fig2.update_xaxes(title="Previsto", title_font=dict(color="#8b92a5"))
            # Eixo Y invertido para "Ativo" ficar em cima, na mesma ordem do X.
            # Com a ordem padrao do heatmap, a diagonal de acerto vai do canto
            # inferior esquerdo ao superior direito e a matriz se le ao contrario.
            fig2.update_yaxes(title="Real", title_font=dict(color="#8b92a5"), autorange="reversed")
            dark(fig2, f"Matriz de Confusão — out-of-fold, corte {md['thr']:.2f}", height=460)
            st.plotly_chart(fig2, use_container_width=True)

        fpr, tpr, _ = roc_curve(md["y"], md["oof"])
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={md['auc']:.3f})",
            line=dict(color=COLORS["indigo"], width=2.5),
            fill="tozeroy", fillcolor="rgba(139,92,246,0.08)",
        ))
        fig3.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Baseline aleatório",
            line=dict(color=COLORS["muted"], width=1, dash="dash"),
        ))
        dark(fig3, "Curva ROC · Capacidade de separação do modelo", height=300)
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown('<p class="section-label">Estabilidade · AUC por Fold</p>', unsafe_allow_html=True)
        cv_vals = md["folds"]
        fig5 = go.Figure(go.Bar(
            x=[f"Fold {i+1}" for i in range(len(cv_vals))], y=cv_vals,
            marker=dict(color=COLORS["cyan"]),
            text=[f"{v:.4f}" for v in cv_vals],
            textposition="outside", textfont=dict(color=COLORS["muted"]),
        ))
        fig5.add_hline(y=cv_vals.mean(), line_dash="dash", line_color=COLORS["amber"],
                       annotation_text=f"Média: {cv_vals.mean():.4f}", annotation_font_color=COLORS["amber"])
        fig5.update_yaxes(range=[0, 1])
        dark(fig5, "AUC-ROC por Fold — baixa variância indica modelo estável", height=260)
        st.plotly_chart(fig5, use_container_width=True)
        st.caption(f"Variância de {cv_vals.std():.4f} entre folds confirma que o modelo generaliza para contratos que não viu.")


main()
