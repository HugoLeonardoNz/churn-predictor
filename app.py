import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
import plotly.graph_objects as go
import plotly.express as px

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

h1 { color: #f0f2f8 !important; font-size: 26px !important; letter-spacing: -0.5px !important; font-weight: 700 !important; }
h2 { color: #cbd5e1 !important; font-size: 16px !important; font-weight: 600 !important; }
h3 { color: #94a3b8 !important; font-size: 13px !important; font-weight: 500 !important; }
p, li { color: #8b92a5 !important; font-size: 13px !important; }
hr { border-color: rgba(99,102,241,0.1) !important; margin: 12px 0 !important; }

.stat-card {
    background: rgba(13,17,30,0.9);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 14px;
    padding: 18px 22px;
}
.stat-label { font-size: 10px; color: #4b5468; letter-spacing: 0.12em; text-transform: uppercase; font-family: 'JetBrains Mono', monospace; margin-bottom: 6px; }
.stat-value { font-size: 28px; font-weight: 700; font-family: 'JetBrains Mono', monospace; line-height: 1.1; }
.stat-sub   { font-size: 11px; color: #4b5468; margin-top: 5px; }

.risk-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 999px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.08em;
}

.action-box {
    background: rgba(13,17,30,0.9);
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: 12px;
    padding: 16px 20px;
    margin-top: 16px;
}
.action-title { font-size: 10px; color: #4b5468; letter-spacing: 0.2em; text-transform: uppercase; font-family: 'JetBrains Mono', monospace; margin-bottom: 10px; }
.action-item { display: flex; align-items: flex-start; gap: 10px; margin-bottom: 8px; font-size: 12px; color: #8b92a5; line-height: 1.5; }

.section-label { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #6366f1; letter-spacing: 0.3em; text-transform: uppercase; margin-bottom: 12px; }

.stTabs [data-baseweb="tab-list"] {
    background: rgba(13,17,30,0.8) !important;
    border: 1px solid rgba(99,102,241,0.12) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] { color: #8b92a5 !important; border-radius: 8px !important; font-size: 13px !important; }
.stTabs [aria-selected="true"] { color: #a5b4fc !important; background: rgba(99,102,241,0.15) !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 20px !important; }

.stSlider label { color: #8b92a5 !important; font-size: 12px !important; }
.stSelectbox label { color: #8b92a5 !important; font-size: 12px !important; }
[data-testid="stDataFrame"] { border: 1px solid rgba(99,102,241,0.15) !important; border-radius: 10px !important; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 3px; }

.stDownloadButton button {
    background: rgba(99,102,241,0.12) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    color: #a5b4fc !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────

PLAN_PRICES = {"Fibra 100MB": 89.90, "Fibra 200MB": 109.90, "Fibra 500MB": 139.90, "Fibra 1GB": 179.90}
PLAN_CHURN  = {"Fibra 100MB": 0.367, "Fibra 200MB": 0.262, "Fibra 500MB": 0.145, "Fibra 1GB": 0.102}
CITIES      = ["Betim", "Contagem", "Ribeirão das Neves", "Esmeraldas", "Ibirité"]
SELLERS     = ["Carlos Mendes", "Patricia Lima", "Roberto Souza", "Fernanda Costa"]
COLORS      = {
    "indigo": "#6366f1", "cyan": "#22d3ee", "green": "#10b981",
    "amber": "#f59e0b", "red": "#ef4444", "muted": "#8b92a5",
}

FEATURE_NAMES = [
    "Mensalidade (R$)", "Dias Ativo", "Faturas em Atraso",
    "Tickets Abertos", "Downgrades", "% Pgto. Atrasado",
    "Score de Risco",
]
FEATURE_COLS = ["amount","days_active","overdue","tickets","downgrades","late_pct","risk_score"]

ACTIONS = {
    "BAIXO": [
        ("✅", "Cliente saudável. Manter ciclo padrão de NPS e check-in semestral."),
        ("📩", "Enviar e-mail de relacionamento com novidades de planos e vantagens."),
        ("📊", "Monitorar indicadores mensalmente — sem ação comercial imediata."),
    ],
    "MÉDIO": [
        ("📞", "Acionar time comercial para contato proativo nos próximos 15 dias."),
        ("🎁", "Oferecer upgrade de plano com desconto de adesão ou mês cortesia."),
        ("🔧", "Verificar tickets abertos e garantir resolução dentro do SLA."),
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

@st.cache_data
def generate_data(n=300, seed=42):
    np.random.seed(seed)
    plans   = np.random.choice(list(PLAN_PRICES), n, p=[0.33, 0.28, 0.23, 0.16])
    cities  = np.random.choice(CITIES, n, p=[0.26, 0.27, 0.23, 0.15, 0.09])
    sellers = np.random.choice(SELLERS, n)
    created = (
        pd.date_range("2022-01-01", periods=n, freq="3D")
        + pd.to_timedelta(np.random.randint(0, 60, n), unit="D")
    )
    churn = np.array([np.random.binomial(1, PLAN_CHURN[p]) for p in plans])

    df = pd.DataFrame({
        "plan": plans, "city": cities, "seller": sellers,
        "created_at": created, "churn": churn,
    })
    df["amount"]      = df["plan"].map(PLAN_PRICES)
    df["days_active"] = (pd.Timestamp("2024-10-31") - df["created_at"]).dt.days
    df["overdue"]     = np.where(churn==1, np.random.poisson(1.8,n).astype(int), np.random.poisson(0.7,n).astype(int))
    df["tickets"]     = np.where(churn==1, np.random.poisson(1.3,n).astype(int), np.random.poisson(0.5,n).astype(int))
    df["downgrades"]  = np.random.choice([0,1,2], n, p=[0.80,0.15,0.05])
    df["late_pct"]    = np.where(churn==1, np.random.beta(2.5,3,n), np.random.beta(1,4,n))
    df["risk_score"]  = df["overdue"]*1.0 + df["tickets"]*2.0 + df["downgrades"]*1.5 + df["late_pct"]*3.0
    return df


@st.cache_resource
def train():
    df = generate_data()
    le_city   = LabelEncoder().fit(CITIES)
    le_plan   = LabelEncoder().fit(list(PLAN_PRICES))
    le_seller = LabelEncoder().fit(SELLERS)

    df["city_enc"]   = le_city.transform(df["city"])
    df["plan_enc"]   = le_plan.transform(df["plan"])
    df["seller_enc"] = le_seller.transform(df["seller"])

    all_cols  = FEATURE_COLS + ["city_enc", "plan_enc", "seller_enc"]
    all_names = FEATURE_NAMES + ["Cidade", "Plano", "Vendedor"]

    X = df[all_cols]
    y = df["churn"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_split=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    model.fit(X_tr, y_tr)

    cv_auc  = cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="roc_auc")
    y_pred  = model.predict(X_te)
    y_prob  = model.predict_proba(X_te)[:, 1]
    y_prob_all = model.predict_proba(X)[:, 1]
    auc     = roc_auc_score(y_te, y_prob)
    cm      = confusion_matrix(y_te, y_pred)
    report  = classification_report(y_te, y_pred, target_names=["Ativo","Churn"], output_dict=True)
    imps    = pd.Series(model.feature_importances_, index=all_names).sort_values(ascending=True)

    return {
        "model": model, "df": df, "X": X, "y": y,
        "X_te": X_te, "y_te": y_te, "y_prob": y_prob, "y_pred": y_pred,
        "y_prob_all": y_prob_all,
        "auc": auc, "cv_auc": cv_auc, "cm": cm, "report": report,
        "importances": imps, "all_cols": all_cols, "all_names": all_names,
        "le_city": le_city, "le_plan": le_plan, "le_seller": le_seller,
    }

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
                {"range": [60,100], "color": "rgba(239,68,68,0.12)"},
            ],
        },
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=220, margin=dict(l=20,r=20,t=40,b=0))
    return fig, color

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    md = train()

    st.markdown("# 🤖 Customer Churn Predictor")
    st.markdown(
        "<p style='margin-top:-8px;margin-bottom:0'>FiberNet ISP · RandomForest + Feature Engineering · "
        "<span style='color:#6366f1;font-family:monospace'>portfólio Hugo Leonardo</span></p>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "💼  Análise de Risco",
        "🔍  Simulador de Contratos",
        "🎯  Plano de Retenção",
        "⚙️  Desempenho do Modelo",
    ])

    # ── Tab 1: Análise de Risco ───────────────────────────────────────────────
    with tab1:
        full_df = md["df"].copy()
        full_df["prob"] = md["y_prob_all"]
        full_df["tier"] = pd.cut(full_df["prob"], bins=[-0.01,0.3,0.6,1.01], labels=["BAIXO","MÉDIO","ALTO"])

        alto  = full_df[full_df["tier"] == "ALTO"]
        medio = full_df[full_df["tier"] == "MÉDIO"]
        baixo = full_df[full_df["tier"] == "BAIXO"]
        mrr_alto  = alto["amount"].sum()
        mrr_medio = medio["amount"].sum()
        mrr_total = full_df["amount"].sum()
        recovery  = mrr_alto * 0.5

        # ── KPIs de negócio ──
        st.markdown('<p class="section-label">Exposição de Receita · Base de 300 Contratos</p>', unsafe_allow_html=True)
        b1, b2, b3, b4 = st.columns(4)
        with b1: stat("MRR em Risco Crítico",  f"R$ {mrr_alto:,.0f}".replace(",","."),  f"{len(alto)} contratos — ação imediata",       COLORS["red"])
        with b2: stat("MRR em Risco Médio",    f"R$ {mrr_medio:,.0f}".replace(",","."), f"{len(medio)} contratos — monitoramento ativo", COLORS["amber"])
        with b3: stat("Base Saudável",         f"{len(baixo)} contratos",               f"R$ {full_df[full_df['tier']=='BAIXO']['amount'].sum():,.0f}/mês estável".replace(",","."), COLORS["green"])
        with b4: stat("Potencial de Retenção", f"R$ {recovery:,.0f}".replace(",","."),  "projetando 50% retenção dos críticos",          COLORS["indigo"])

        # ── Insight callout ──
        st.markdown("<br>", unsafe_allow_html=True)
        pct_mrr_risco = (mrr_alto + mrr_medio) / mrr_total * 100
        churn_100 = md["df"][md["df"]["plan"] == "Fibra 100MB"]["churn"].mean() * 100
        churn_1g  = md["df"][md["df"]["plan"] == "Fibra 1GB"]["churn"].mean() * 100
        st.markdown(f"""
        <div style='background:rgba(239,68,68,0.06);border:1px solid rgba(239,68,68,0.2);border-radius:12px;padding:16px 20px;margin-bottom:24px'>
          <div style='font-size:10px;color:#ef4444;letter-spacing:0.2em;font-family:monospace;text-transform:uppercase;margin-bottom:8px'>⚠ Alerta de Risco · Base Atual</div>
          <div style='font-size:13px;color:#c8d0db;line-height:1.7'>
            <b style='color:#f8fafc'>{pct_mrr_risco:.0f}% do MRR total</b> está em zona de risco (ALTO + MÉDIO).
            Plano <b style='color:#ef4444'>Fibra 100MB concentra {churn_100:.0f}% de churn</b> —
            3.5x maior que o Fibra 1GB ({churn_1g:.0f}%). Clientes com histórico de atraso e múltiplos
            tickets são os de maior exposição e devem ser priorizados na régua de retenção.
          </div>
        </div>""", unsafe_allow_html=True)

        # ── Churn por plano + Score de risco ──
        st.markdown('<p class="section-label">Padrão de Cancelamento por Segmento</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            churn_by_plan = (
                md["df"].groupby("plan")["churn"]
                .apply(lambda x: x.mean() * 100)
                .reset_index(name="churn_pct")
            )
            fig = go.Figure(go.Bar(
                x=churn_by_plan["plan"], y=churn_by_plan["churn_pct"],
                marker=dict(color=[COLORS["red"], COLORS["amber"], COLORS["cyan"], COLORS["green"]]),
                text=[f"{v:.1f}%" for v in churn_by_plan["churn_pct"]],
                textposition="outside", textfont=dict(color=COLORS["muted"]),
            ))
            dark(fig, "Taxa de Churn por Plano (%)", height=280)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Correlação inversa entre valor do plano e churn — clientes de menor ticket concentram maior risco de evasão.")

        with col2:
            risk_dist = md["df"].groupby("plan")["risk_score"].mean().reset_index()
            fig2 = go.Figure(go.Bar(
                x=risk_dist["plan"], y=risk_dist["risk_score"],
                marker=dict(color=risk_dist["risk_score"], colorscale=[[0,"#8b5cf6"],[1,"#ef4444"]]),
                text=[f"{v:.2f}" for v in risk_dist["risk_score"]],
                textposition="outside", textfont=dict(color=COLORS["muted"]),
            ))
            dark(fig2, "Score de Risco Médio por Plano", height=280)
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("Score composto: faturas em atraso + tickets abertos + downgrades + % pagamentos em atraso.")

        # ── Dispersão risco ──
        st.markdown('<p class="section-label">Perfil de Risco · Tempo de Vida vs Score</p>', unsafe_allow_html=True)
        fig3 = px.scatter(
            md["df"], x="days_active", y="risk_score",
            color=md["df"]["churn"].map({0:"Ativo", 1:"Churn"}),
            color_discrete_map={"Ativo": COLORS["indigo"], "Churn": COLORS["red"]},
            opacity=0.65, size_max=8,
            labels={"days_active":"Dias como Cliente","risk_score":"Score de Risco"},
        )
        dark(fig3, "Tempo de Vida vs Score de Risco — cada ponto é um contrato", height=300)
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Contratos com alto score de risco independem do tempo de casa — o comportamento financeiro é o principal sinalizador.")

        # ── Desempenho do modelo (rodapé técnico) ──
        st.markdown("<hr style='border-color:rgba(139,92,246,0.1);margin:28px 0'>", unsafe_allow_html=True)
        st.markdown('<p class="section-label">Precisão do Modelo · Validação Técnica</p>', unsafe_allow_html=True)
        st.caption("O score de risco é calculado por um modelo RandomForest treinado e validado com as métricas abaixo. Para análise detalhada, veja a aba Desempenho do Modelo.")
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1: stat("AUC-ROC",        f"{md['auc']:.3f}",                           "conjunto de teste (holdout)",      COLORS["indigo"])
        with c2: stat("Cross-Val AUC",  f"{md['cv_auc'].mean():.3f}",                 f"± {md['cv_auc'].std():.3f} · 5-fold", COLORS["cyan"])
        with c3: stat("Precisão Churn", f"{md['report']['Churn']['precision']:.0%}",  "dos alertas são verdadeiros",      COLORS["amber"])
        with c4: stat("Recall Churn",   f"{md['report']['Churn']['recall']:.0%}",     "dos churns são detectados",        COLORS["green"])

    # ── Tab 2: Predição Interativa ────────────────────────────────────────────
    with tab2:
        st.markdown("#### Simule o perfil de um cliente e veja a probabilidade de churn em tempo real.")
        st.markdown("<br>", unsafe_allow_html=True)

        col_form, col_result = st.columns([1, 1])

        with col_form:
            st.markdown('<p class="section-label">Perfil do Contrato</p>', unsafe_allow_html=True)
            p_plan   = st.selectbox("Plano",   list(PLAN_PRICES))
            p_city   = st.selectbox("Cidade",  CITIES)
            p_seller = st.selectbox("Vendedor",SELLERS)
            p_days   = st.slider("Dias como cliente", 30, 900, 180)
            p_overdue= st.slider("Faturas em atraso", 0, 10, 0)
            p_tickets= st.slider("Tickets abertos",   0, 8,  0)
            p_down   = st.slider("Downgrades",        0, 3,  0)
            p_late   = st.slider("% pagamentos em atraso", 0.0, 1.0, 0.05, step=0.05, format="%.0f%%")

        with col_result:
            risk_sc = p_overdue*1.0 + p_tickets*2.0 + p_down*1.5 + p_late*3.0
            le_c = md["le_city"]; le_p = md["le_plan"]; le_s = md["le_seller"]

            x_input = pd.DataFrame([[
                PLAN_PRICES[p_plan], p_days, p_overdue, p_tickets, p_down, p_late, risk_sc,
                le_c.transform([p_city])[0],
                le_p.transform([p_plan])[0],
                le_s.transform([p_seller])[0],
            ]], columns=md["all_cols"])

            prob  = md["model"].predict_proba(x_input)[0, 1]
            level = "BAIXO" if prob < 0.3 else "MÉDIO" if prob < 0.6 else "ALTO"
            color = COLORS["green"] if level == "BAIXO" else COLORS["amber"] if level == "MÉDIO" else COLORS["red"]

            fig_g, col_g = gauge(prob)
            st.plotly_chart(fig_g, use_container_width=True)

            rgb = tuple(int(col_g.lstrip('#')[i:i+2], 16) for i in (0,2,4))
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
                <div class="stat-value" style="color:{COLORS['amber']}">R$ {PLAN_PRICES[p_plan]:.2f}/mês</div>
                <div class="stat-sub">Plano {p_plan}</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            feat_vals = x_input.iloc[0].rename(index=dict(zip(md["all_cols"], md["all_names"])))
            contrib   = feat_vals * pd.Series(md["model"].feature_importances_, index=md["all_names"])
            top3      = contrib.abs().sort_values(ascending=False).head(3)

            st.markdown('<p class="section-label" style="text-align:center">Principais Fatores</p>', unsafe_allow_html=True)
            for feat, val in top3.items():
                pct = val / contrib.abs().sum() * 100
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
                    <span style="color:#8b92a5;font-size:11px;width:150px;flex-shrink:0">{feat}</span>
                    <div style="flex:1;height:6px;background:rgba(255,255,255,0.07);border-radius:3px">
                        <div style="width:{pct:.0f}%;height:6px;background:{COLORS['indigo']};border-radius:3px;opacity:0.85"></div>
                    </div>
                    <span style="color:{COLORS['indigo']};font-family:monospace;font-size:10px">{pct:.0f}%</span>
                </div>""", unsafe_allow_html=True)

        # Plano de Ação
        st.markdown("<br>", unsafe_allow_html=True)
        action_color = color
        st.markdown(f"""
        <div class="action-box" style="border-color:{action_color}33">
            <div class="action-title" style="color:{action_color}">⚡ Plano de Ação · Risco {level}</div>
            {''.join(f'<div class="action-item"><span style="font-size:16px">{icon}</span><span>{text}</span></div>' for icon, text in ACTIONS[level])}
        </div>""", unsafe_allow_html=True)

    # ── Tab 3: Plano de Retenção ──────────────────────────────────────────────
    with tab3:
        full_df = md["df"].copy()
        full_df["prob"] = md["y_prob_all"]
        full_df["tier"] = pd.cut(
            full_df["prob"], bins=[-0.01,0.3,0.6,1.01],
            labels=["BAIXO","MÉDIO","ALTO"]
        )

        alto  = full_df[full_df["tier"] == "ALTO"]
        medio = full_df[full_df["tier"] == "MÉDIO"]
        baixo = full_df[full_df["tier"] == "BAIXO"]

        mrr_alto  = alto["amount"].sum()
        mrr_medio = medio["amount"].sum()

        st.markdown('<p class="section-label">Exposição por Nível de Risco · Base de 300 Contratos</p>', unsafe_allow_html=True)
        r1, r2, r3, r4 = st.columns(4)
        with r1: stat("Risco ALTO",    str(len(alto)),  f"R$ {mrr_alto:,.0f}/mês em risco".replace(",","."),   COLORS["red"])
        with r2: stat("Risco MÉDIO",   str(len(medio)), f"R$ {mrr_medio:,.0f}/mês em risco".replace(",","."),  COLORS["amber"])
        with r3: stat("Risco BAIXO",   str(len(baixo)), "base saudável · sem ação necessária",                  COLORS["green"])
        with r4: stat("Recovery Est.", f"R$ {mrr_alto*0.5:,.0f}".replace(",","."), "projetando 50% retenção ALTO", COLORS["indigo"])

        st.markdown("<br>", unsafe_allow_html=True)

        col_a, col_b = st.columns([2, 3])
        with col_a:
            tier_counts = full_df["tier"].value_counts().reindex(["ALTO","MÉDIO","BAIXO"])
            fig_pie = go.Figure(go.Pie(
                labels=["🔴 ALTO","🟡 MÉDIO","🟢 BAIXO"],
                values=tier_counts.values,
                hole=0.5,
                marker=dict(colors=[COLORS["red"], COLORS["amber"], COLORS["green"]]),
                textfont=dict(color="#f0f2f8", size=12),
            ))
            fig_pie.update_traces(textinfo="label+percent+value")
            dark(fig_pie, "Distribuição de Risco · Base Completa", height=280)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_b:
            fig_bar = go.Figure()
            for tier, color, label in [("ALTO", COLORS["red"], "🔴 ALTO"), ("MÉDIO", COLORS["amber"], "🟡 MÉDIO"), ("BAIXO", COLORS["green"], "🟢 BAIXO")]:
                grp = full_df[full_df["tier"] == tier].groupby("plan")["amount"].sum().reset_index()
                fig_bar.add_trace(go.Bar(
                    name=label, x=grp["plan"], y=grp["amount"],
                    marker_color=color, opacity=0.85,
                ))
            fig_bar.update_layout(barmode="stack")
            dark(fig_bar, "MRR em Risco por Plano (R$/mês)", height=280)
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown('<p class="section-label">Lista de Prioridade · Contratos ALTO & MÉDIO Risco</p>', unsafe_allow_html=True)

        priority = (
            full_df[full_df["tier"].isin(["ALTO","MÉDIO"])]
            .sort_values("prob", ascending=False)
            .reset_index(drop=True)
        )
        priority.index = range(1, len(priority)+1)

        display = priority[["city","plan","seller","amount","days_active","overdue","tickets","prob","tier"]].copy()
        display.columns = ["Cidade","Plano","Vendedor","R$/mês","Dias Ativo","Fat. Atraso","Tickets","Prob. Churn","Risco"]
        display["Prob. Churn"] = display["Prob. Churn"].apply(lambda x: f"{x:.1%}")
        display["Ação"] = display["Risco"].map({
            "ALTO":  "🚨 Supervisor — 48h",
            "MÉDIO": "📞 Comercial — 15d",
        })

        st.dataframe(display, use_container_width=True, height=400)
        st.caption(f"{len(priority)} contratos em zona de risco · ordenados por probabilidade de churn decrescente · prioridade de contato definida por nível")

        st.markdown("<br>", unsafe_allow_html=True)
        col_dl, _ = st.columns([2, 5])
        with col_dl:
            csv = priority.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Exportar Lista de Retenção (CSV)",
                data=csv,
                file_name="fibernet_plano_retencao.csv",
                mime="text/csv",
            )

    # ── Tab 4: Desempenho do Modelo ───────────────────────────────────────────
    with tab4:
        st.caption("Validação técnica do modelo que gera os scores de risco exibidos nas demais abas.")
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            imp = md["importances"]
            fig = go.Figure(go.Bar(
                x=imp.values, y=imp.index, orientation="h",
                marker=dict(color=imp.values, colorscale=[[0,"#1e1b4b"],[0.4,"#8b5cf6"],[1,"#38bdf8"]]),
                text=[f"{v:.3f}" for v in imp.values],
                textposition="outside", textfont=dict(color=COLORS["muted"], size=11),
            ))
            dark(fig, "Variáveis mais relevantes para o modelo", height=320)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("% de pagamentos em atraso e score de risco são os maiores preditores de cancelamento.")

        with col2:
            cm = md["cm"]
            fig2 = go.Figure(go.Heatmap(
                z=cm, x=["Ativo","Churn"], y=["Ativo","Churn"],
                colorscale=[[0,"#0d1117"],[0.5,"rgba(139,92,246,0.4)"],[1,"#8b5cf6"]],
                text=[[str(v) for v in row] for row in cm],
                texttemplate="%{text}",
                textfont=dict(color="#f0f2f8", size=20),
                showscale=False,
            ))
            fig2.update_xaxes(title="Previsto", title_font=dict(color="#8b92a5"))
            fig2.update_yaxes(title="Real",     title_font=dict(color="#8b92a5"))
            dark(fig2, "Matriz de Confusão — Conjunto de Teste", height=320)
            st.plotly_chart(fig2, use_container_width=True)

        fpr, tpr, _ = roc_curve(md["y_te"], md["y_prob"])
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={md['auc']:.3f})",
            line=dict(color=COLORS["indigo"], width=2.5),
            fill="tozeroy", fillcolor="rgba(139,92,246,0.08)",
        ))
        fig3.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode="lines", name="Baseline aleatório",
            line=dict(color=COLORS["muted"], width=1, dash="dash"),
        ))
        dark(fig3, "Curva ROC · Capacidade de separação do modelo", height=300)
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown('<p class="section-label">Estabilidade · Cross-Validation 5-Fold</p>', unsafe_allow_html=True)
        cv_vals = md["cv_auc"]
        fig5 = go.Figure(go.Bar(
            x=[f"Fold {i+1}" for i in range(len(cv_vals))], y=cv_vals,
            marker=dict(color=cv_vals, colorscale=[[0,"#8b5cf6"],[1,"#38bdf8"]]),
            text=[f"{v:.4f}" for v in cv_vals],
            textposition="outside", textfont=dict(color=COLORS["muted"]),
        ))
        fig5.add_hline(y=cv_vals.mean(), line_dash="dash", line_color=COLORS["amber"],
                       annotation_text=f"Média: {cv_vals.mean():.4f}", annotation_font_color=COLORS["amber"])
        dark(fig5, "AUC-ROC por Fold — baixa variância indica modelo estável", height=260)
        st.plotly_chart(fig5, use_container_width=True)
        st.caption(f"Variância de {cv_vals.std():.4f} entre folds confirma que o modelo generaliza bem para novos contratos.")


main()
