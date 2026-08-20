# Customer Churn Predictor — FiberNet ISP

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Domain](https://img.shields.io/badge/Domain-Telecom%20%2F%20ISP-0ea5e9?style=for-the-badge)
![Status](https://img.shields.io/badge/Rodar-local%20em%202%20comandos-10b981?style=for-the-badge)

**Projeto 3 de 3 da série FiberNet Analytics.**  
Pipeline completo de ML para identificação e priorização de contratos em risco de cancelamento — do feature engineering ao plano de ação operacional.

[Ver pipeline técnico](#desempenho-dos-modelos-pipelinepy) · [Como rodar](#como-rodar)

</div>

---

![Dashboard de análise de risco](docs/img/app.png)

*Aba de Análise de Risco: exposição de MRR por faixa, alerta de concentração e padrão de cancelamento por plano.*

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

> **Leia isto antes dos números.** A base é gerada, e as relações abaixo foram
> **escolhidas** no gerador — o churn por plano sai de `PLAN_FACTOR`, os pesos SHAP
> refletem a ordem em que as variáveis entram na simulação. Não são descobertas
> sobre o mercado de ISP, e citá-las como se fossem seria o mesmo erro que a seção
> "Por que 0,78 e não 0,99" descreve, uma camada acima.
>
> O que É demonstrável aqui: o pipeline recupera a estrutura que existe no dado,
> mede o quanto dela é recuperável (AUC 0,785 contra um teto imposto por 15% de
> ruído), e traduz isso em priorização com MRR em exposição. A leitura de negócio
> é o exercício; o achado de mercado teria de vir de base real.

- **Plano Fibra 100MB concentra 36,7% de churn** — 3,6× o do Fibra 1GB (10,2%). É o mesmo
  número que o [SQL Analytics Pack](https://github.com/HugoLeonardoNz/sql-analytics-pack)
  devolve rodando `queries/02_churn_por_plano.sql` contra o seed: os dois projetos
  compartilham a base FiberNet, e conferir isso é o teste de que a série é coerente
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

Acessa em `http://localhost:8501`.

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
- Base própria de 300 contratos, separada da do `pipeline.py` — as métricas do modelo
  do app são calculadas e exibidas na aba **Desempenho do Modelo**, não fixadas aqui

### `pipeline.py` — Validação Técnica Completa
- Compara 4 modelos: Logistic Regression, RandomForest, XGBoost, LightGBM
- Modelo final: LogisticRegression — AUC 0.785, F1 0.673
- Threshold ótimo calibrado via curva Precision-Recall (0.603)
- SHAP values calculados para interpretabilidade de feature importance
- Calibração de probabilidade com isotonic regression

> O `pipeline.py` representa a profundidade técnica do projeto. O `app.py` representa a entrega para o usuário final. Ambos são parte do mesmo fluxo de trabalho analítico.

---

## Desempenho dos Modelos (pipeline.py)

| Modelo | CV AUC (μ) | AUC-ROC | F1 | Precisão | Recall | Brier |
|--------|-----------|---------|-----|---------|--------|-------|
| **LogisticRegression** ✓ | **0.7738** | **0.7854** | 0.6517 | 0.6542 | 0.6491 | 0.1695 |
| RandomForest | 0.7800 | 0.7787 | 0.6710 | **0.8009** | 0.5774 | **0.1418** |
| XGBoost | 0.7791 | 0.7769 | **0.6733** | 0.7549 | 0.6076 | 0.1530 |
| LightGBM | 0.7741 | 0.7846 | 0.6667 | 0.7418 | 0.6054 | 0.1550 |

### Por que 0,78 e não 0,99

Uma versão anterior deste projeto reportava **AUC 0,996** — e a regressão logística fazia
0,9946. Quando o modelo mais simples empata com o mais sofisticado na terceira casa, o que
está fácil é o problema, não o algoritmo.

A causa estava no gerador: cada variável era sorteada de uma distribuição diferente
conforme o cliente ser churn ou não. O rótulo era o próprio parâmetro da simulação, o
modelo só precisava desfazer a conta, e o SHAP redescobria as regras que o script tinha
acabado de escrever. O número media o gerador, não o modelo.

Hoje o gerador tem `RUIDO_COMPORTAMENTAL = 0.15`: **15% dos clientes têm desfecho que
contraria o próprio comportamento** — gente que cancela sem nenhum sinal prévio e gente
que acumula atraso, chamado e NPS baixo e fica. Isso existe em base real, onde as
variáveis disponíveis nunca explicam tudo, e cria um **teto de acerto**. A suíte de testes
verifica esse teto: `test_auc_na_faixa_esperada` falha tanto abaixo de 0,72 quanto **acima
de 0,92** — porque, com esse ruído, AUC alto demais só pode vir de vazamento.

O resultado é mais interessante que o anterior. Os quatro modelos empatam em AUC (0,777 a
0,785), então a escolha deixa de ser "qual tem o maior número" e passa a ser um
trade-off explícito:

- **RandomForest** tem a melhor calibração (Brier 0,142) e a maior precisão (0,80) — erra
  menos ao dizer "vai cancelar", e é o que faz sentido quando cada acionamento custa
  desconto ou visita técnica;
- **XGBoost** tem o melhor F1, o equilíbrio entre os dois erros;
- **LogisticRegression** ganha no AUC médio da validação cruzada e é o modelo do relatório
  — com coeficientes legíveis, o que num painel de retenção vale mais do que 0,002 de AUC.

Essa conversa é a entrega. Uma tabela em que todo mundo faz 0,99 não tem conversa nenhuma.

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

| Rank | Feature | SHAP médio | Interpretação |
|------|---------|-----------:|---------------|
| 1 | `meses_sem_incidente` | 0,961 | Tempo desde o último incidente — o sinal mais forte da base |
| 2 | `score_satisfacao` | 0,749 | Derivada de NPS e chamados; concentra a leitura de satisfação |
| 3 | `nps_score` | 0,582 | NPS baixo é sinal precoce de saída |
| 4 | `qtd_chamados_suporte` | 0,298 | Volume de tickets |
| 5 | `plano` | 0,177 | Plano de menor ticket concentra mais cancelamento |

`meses_sem_incidente` e `score_satisfacao` sozinhos carregam mais peso que todo o resto
somado, e as variáveis financeiras (`risco_pagamento`, `dias_atraso_pagamento`) aparecem
só no 8º e 9º lugar.

**Esse ranking é o do gerador, não do mercado.** Ele diz que o SHAP recuperou
corretamente a estrutura que `build_dataset` escreveu — que é o que se pode pedir a uma
validação de pipeline, e é exatamente por isso que ele está aqui. A frase que essa
figura *sugere* ("quando o atraso de pagamento aparece, a decisão já foi tomada") é uma
hipótese plausível de retenção em ISP, e continua sendo só isso: uma hipótese, até
alguém rodar este mesmo pipeline contra uma base real.

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
