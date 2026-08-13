# Customer Churn Predictor — FiberNet ISP

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Domain](https://img.shields.io/badge/Domain-Telecom%20%2F%20ISP-0ea5e9?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Live-10b981?style=for-the-badge)

**Projeto 3 de 3 da série FiberNet Analytics.**  
Pipeline completo de ML para identificação e priorização de contratos em risco de cancelamento — do feature engineering ao plano de ação operacional.

[🔗 Ver Demo Live](https://hugoleonardonz-churn-predictor.streamlit.app) · [Ver Pipeline Técnico](#-pipeline-técnico-pipelinepy)

</div>

---

## Universo FiberNet — Escala Canônica

Os 3 projetos desta série representam a **mesma empresa fictícia** em granularidades complementares:

| Granularidade | Projetos | Escala | Abrangência |
|---|---|---|---|
| **Análise Regional** | SQL Analytics Pack · Churn Predictor | 300 contratos | Região Centro-MG: Betim, Contagem, Ribeirão das Neves, Esmeraldas, Ibirité |
| **Visão Operacional Nacional** | Telecom KPI Dashboard | ~82.500 clientes | 5 regiões nacionais (Norte, Sul, Leste, Oeste, Centro) |

A divergência de escala é **intencional**: o SQL Pack e o Churn Predictor mergulham numa amostra regional de alta granularidade para análise SQL profunda e modelagem preditiva. O KPI Dashboard consolida a operação completa para monitoramento em tempo real — os mesmos padrões de negócio (churn por plano, inadimplência, MRR), em escala nacional.

---

## O Problema de Negócio

Em ISPs, churn não controlado corrói o MRR silenciosamente. O desafio não é saber que clientes cancelam — é saber **quais vão cancelar antes que cancelem**, para acionar retenção com tempo hábil.

Este projeto entrega exatamente isso: uma lista priorizada de contratos por nível de risco, com MRR em exposição calculado e plano de ação definido por faixa.

---

## O Que o Analista Entrega

| Entrega | Descrição |
|---------|-----------|
| Score de risco por contrato | Composto por atraso, tickets, downgrades e % pgto. em atraso |
| Segmentação ALTO / MÉDIO / BAIXO | Prioridade de ação para o time comercial e de retenção |
| MRR em risco calculado | Exposição financeira real por segmento |
| Plano de ação por faixa | Do contato proativo ao cashback de emergência |
| Lista exportável (CSV) | Pronta para uso no CRM ou régua de atendimento |

---

## Principais Achados (Base Sintética FiberNet)

- **Plano Fibra 100MB concentra ~37% de churn** — 3,6× maior que o Fibra 1GB (10%)
- Correlação inversa clara entre valor do plano e taxa de cancelamento
- Clientes com ≥ 2 faturas em atraso + score de risco > 4 têm probabilidade de churn > 70%
- **Tempo de casa por si só não é protetor** — o comportamento financeiro é o principal sinalizador

---

## Estrutura do Projeto

```
churn-predictor/
├── app.py              ← Dashboard interativo (Streamlit) — RandomForest
├── pipeline.py         ← Validação técnica: 4 modelos comparados, SHAP, threshold ótimo
├── predictions.csv     ← Output rápido do pipeline
├── outputs/
│   ├── predictions.csv
│   ├── relatorio_churn.md   ← Relatório completo gerado pelo pipeline
│   └── shap/                ← SHAP summary, dependência e force plots
├── tests/
│   └── test_sanity.py  ← Testes de sanidade do pipeline
└── requirements.txt
```

---

## Como Rodar

### Dashboard interativo (Streamlit)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Acessa em `http://localhost:8501` — sem instalação: [ver demo live](https://hugoleonardonz-churn-predictor.streamlit.app)

### Pipeline de Validação Técnica

```bash
python pipeline.py
# Gera: outputs/relatorio_churn.md · outputs/predictions.csv · outputs/shap/*.png
```

---

## Dois Componentes, um Objetivo

Este projeto tem dois artefatos intencionalmente separados:

### `app.py` — Dashboard Interativo
- Modelo: **RandomForest** (scikit-learn)
- Foco: usabilidade, simulador interativo, plano de ação em tempo real
- Recall churn: ~95% no dado sintético — ver a ressalva sobre as métricas mais abaixo

### `pipeline.py` — Validação Técnica Completa
- Compara 4 modelos: Logistic Regression, RandomForest, **XGBoost** (selecionado), LightGBM
- Modelo final: XGBoost — AUC 0.9960, F1 0.9544
- Threshold ótimo calibrado via curva Precision-Recall (0.522)
- SHAP values calculados para interpretabilidade de feature importance
- Calibração de probabilidade com isotonic regression

> O `pipeline.py` representa a profundidade técnica do projeto. O `app.py` representa a entrega para o usuário final. Ambos são parte do mesmo fluxo de trabalho analítico.

---

## Desempenho dos Modelos (pipeline.py)

| Modelo | CV AUC (μ) | AUC-ROC | F1 | Recall |
|--------|-----------|---------|-----|--------|
| LogisticRegression | 0.9934 | 0.9946 | 0.9409 | 0.9556 |
| RandomForest | 0.9942 | 0.9951 | 0.9520 | 0.9467 |
| **XGBoost** ✓ | **0.9949** | **0.9960** | **0.9544** | **0.9544** |
| LightGBM | 0.9948 | 0.9960 | 0.9530 | 0.9567 |

### Sobre esses números: eles medem o gerador, não o modelo

AUC 0,996 em churn não acontece com dado real. Aqui acontece porque **os dados são
sintéticos e gerados condicionalmente ao rótulo**: cada variável é sorteada de uma
distribuição diferente conforme o cliente ser churn ou não (`build_dataset`, em
`pipeline.py`, declara isso na própria docstring — "garante alta separabilidade").

O sinal de que isso é artefato do dado e não mérito do modelo está na própria tabela:
**uma regressão logística simples faz 0,9946.** Quando o modelo mais simples empata com o
mais sofisticado em 0,99, o que está fácil é o problema, não o algoritmo.

O que este projeto demonstra de verdade: montagem de pipeline com validação cruzada
estratificada, comparação de modelos, calibração de threshold pela curva
Precision-Recall, calibração de probabilidade e interpretabilidade com SHAP. **A métrica
não é evidência de performance em produção** — com dado real, esperar algo entre 0,75 e
0,85 seria realista.

---

## Interpretabilidade — o que o modelo está olhando

![Importância das features por SHAP](outputs/shap/shap_summary.png)

Cada ponto é um cliente. À direita do eixo, a variável empurra a previsão para churn; à
esquerda, segura. A cor é o valor da variável — vermelho alto, azul baixo.

![Matriz de confusão](outputs/confusion_matrix.png)

Explicação individual — por que **este** cliente foi classificado como risco alto:

![Explicação individual de um cliente](outputs/shap/shap_force_cliente_1.png)

---

## Top Features por Importância (SHAP)

| Rank | Feature | Interpretação |
|------|---------|---------------|
| 1 | `meses_sem_incidente` | Clientes que não abrem chamados há muito tempo têm menos engajamento |
| 2 | `score_satisfacao` | NPS baixo é sinal precoce de saída |
| 3 | `score_risco_total` | Agregado de inadimplência + suporte |
| 4 | `qtd_chamados_suporte` | Volume de tickets é preditor forte |
| 5 | `dias_atraso_pagamento` | Atraso financeiro confirma tendência de saída |

---

## Série FiberNet Analytics

Este é o **Projeto 3 de 3** de uma série coesa sobre inteligência de dados em ISP:

| # | Projeto | Foco | Link |
|---|---------|------|------|
| 1 | [SQL Analytics Pack](https://github.com/HugoLeonardoNz/SQL-Analytics-Pack) | SQL analítico · 10 queries · insights brutos | GitHub |
| 2 | [Telecom KPI Dashboard](https://github.com/HugoLeonardoNz/telecom-kpi-dashboard) | BI operacional · visualização em tempo real | GitHub |
| 3 | **Customer Churn Predictor** | ML · predição e priorização de risco | **Este repo** |

---

## Stack

`Python` · `scikit-learn` · `XGBoost` · `LightGBM` · `Streamlit` · `Plotly` · `Pandas` · `NumPy` · `SHAP`

---

## Autor

**Hugo Leonardo**  
Analista de Dados Pleno — SQL · Python · Power BI  
Speed Fibra · Santa Luzia, MG

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Hugo%20Leonardo-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/hugo-leonardo-data-analyst/)
[![GitHub](https://img.shields.io/badge/GitHub-HugoLeonardoNz-181717?style=flat&logo=github)](https://github.com/HugoLeonardoNz)

---

<div align="center">
<sub>Dados 100% sintéticos gerados para fins de portfólio. Nenhuma informação real de clientes foi utilizada.</sub>
</div>
