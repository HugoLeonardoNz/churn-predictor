# Audit Report — Customer Churn Predictor

**Data:** 2026-04-27  
**Auditor:** Hugo Leonardo  
**Versão:** v2.0

---

## Resumo do Projeto

Pipeline completo de Machine Learning para predição de churn em ISP de fibra (universo FiberNet). Compara quatro modelos (LogisticRegression, RandomForest, XGBoost, LightGBM) sobre um gerador com 15% de desfechos que contrariam o comportamento observado — o que coloca a AUC na faixa de 0,78 e transforma a escolha de modelo num trade-off entre calibração, precisão e legibilidade, em vez de uma corrida por decimal. Entrega interface Streamlit interativa com explicabilidade SHAP.

Projeto 3 de 3 da série FiberNet Analytics: SQL Pack → KPI Dashboard → **Churn Predictor**.

---

## Tecnologias

- **Python 3.10+** — linguagem principal
- **Streamlit** — interface interativa (`app.py`)
- **XGBoost / scikit-learn** — modelagem preditiva (`pipeline.py`)
- **SHAP** — explicabilidade de features
- **Plotly** — visualizações de métricas e features
- **pytest** — testes de sanidade (`tests/test_sanity.py`)
- **Pandas / NumPy** — manipulação de dados

---

## Estrutura

```
churn-predictor/
├── app.py              — Dashboard Streamlit interativo
├── pipeline.py         — Pipeline técnico de validação e treinamento
├── requirements.txt    — Dependências pinadas
├── tests/
│   └── test_sanity.py  — Testes pytest de sanidade do modelo
├── outputs/
│   ├── predictions.csv
│   ├── relatorio_churn.md
│   └── shap/           — Visualizações SHAP (summary, dependência, force plots)
├── .gitignore
└── AUDIT_REPORT.md     — Este arquivo
```

---

## Status da Estrutura

| Item | Status |
|---|---|
| README.md real | ✅ |
| requirements.txt pinado | ✅ |
| Testes automatizados | ✅ (`tests/test_sanity.py`) |
| Outputs documentados | ✅ |
| .gitignore Python | ✅ (adicionado 2026-04-27) |
| AUDIT_REPORT.md | ✅ (criado 2026-04-27) |
| Live demo | ✅ [Streamlit Cloud](https://hugoleonardonz-churn-predictor.streamlit.app) |

---

## Pontos Fortes

- Dois artefatos complementares: `app.py` (experiência do usuário) + `pipeline.py` (validação técnica rigorosa)
- Comparação explícita de quatro algoritmos com cross-validation 5-fold
- SHAP: summary plot + dependência top-3 features + force plots de 3 clientes individuais
- Dados sintéticos FiberNet gerados por `churn_data.py`: 15.000 contratos, fonte única do app e do pipeline. **Não** é a mesma tabela do SQL Pack nem do KPI Dashboard — cada projeto gera a sua; o que se repete é o padrão de negócio, não o número
- Lista priorizada de contratos críticos exportável com MRR em risco

---

## Melhorias Aplicadas (2026-04-27)

- Adicionado `.gitignore` Python padrão (remove `__pycache__/`, `.pyc`, etc.)
- Criado `AUDIT_REPORT.md` para rastreabilidade da evolução do projeto
