# Relatório de Análise de Churn — FiberNet ISP

_Gerado automaticamente por pipeline.py - Dados sintéticos - Hugo Leonardo_

---

## 1. Modelo Escolhido e Justificativa

**Modelo selecionado: `LogisticRegression`**

Escolhido por apresentar o maior AUC-ROC médio em validação cruzada estratificada (5-fold), com AUC = **0.7854** no conjunto de teste. Modelos tree-based são preferíveis neste contexto por capturarem interações não-lineares entre features de risco (atraso, NPS, suporte) sem exigir normalidade dos dados.

---

## 2. Tabela Comparativa dos 4 Modelos

| Modelo             |   CV AUC (μ) |   CV AUC (σ) |   AUC-ROC |     F1 |   Precisão |   Recall |   Brier |
|:-------------------|-------------:|-------------:|----------:|-------:|-----------:|---------:|--------:|
| LogisticRegression |       0.7738 |       0.008  |    0.7854 | 0.6517 |     0.6542 |   0.6491 |  0.1695 |
| RandomForest       |       0.78   |       0.0096 |    0.7787 | 0.671  |     0.8009 |   0.5774 |  0.1418 |
| XGBoost            |       0.7791 |       0.0088 |    0.7769 | 0.6733 |     0.7549 |   0.6076 |  0.153  |
| LightGBM           |       0.7741 |       0.0118 |    0.7846 | 0.6667 |     0.7418 |   0.6054 |  0.155  |

> Métricas calculadas no conjunto de teste (20% holdout, estratificado).
> Threshold padrão = 0.50.

---

## 3. Top 10 Features por Importância (SHAP)

| Rank | Feature | SHAP médio |
|------|---------|-----------|
| 1 | `meses_sem_incidente` | 0.9609 |
| 2 | `score_satisfacao` | 0.7491 |
| 3 | `nps_score` | 0.5815 |
| 4 | `qtd_chamados_suporte` | 0.2984 |
| 5 | `plano` | 0.1772 |
| 6 | `tem_fidelidade` | 0.1450 |
| 7 | `regiao` | 0.0937 |
| 8 | `risco_pagamento` | 0.0730 |
| 9 | `dias_atraso_pagamento` | 0.0729 |
| 10 | `uso_medio_gb` | 0.0710 |

> Importância global = média do valor absoluto dos SHAP values no conjunto de teste.

---

## 4. Threshold Ótimo e Impacto na Precision/Recall

| Métrica | Threshold = 0.50 | Threshold ótimo = 0.603 |
|---------|-----------------|--------------------------------------|
| F1-Score  | 0.6517 | 0.6730 |
| Precisão  | 0.6542 | 0.7702 |
| Recall    | 0.6491 | 0.5975 |
| AUC-ROC   | 0.7854 | 0.7854 |

**Threshold ótimo = 0.603** — calculado pelo máximo F1 na curva Precision-Recall.
Recall é priorizado (não perder churners reais) em detrimento de precisão.

### Calibração de Probabilidade

| | Brier Score |
|---|-------------|
| Modelo original  | 0.1695 |
| Após calibração (isotonic) | 0.1388 |

---

## 5. Perfis de Clientes de Alto Risco

### Cliente Alto Risco #1

- **Probabilidade de churn:** `99.8%`
- **Plano:** Fibra 200MB
- **Valor mensalidade:** R$ 112.88
- **Tempo de contrato:** 33 meses
- **Dias em atraso:** 71 dias
- **NPS score:** 0.0
- **Chamados de suporte:** 8
- **Fidelidade:** Não
- **SHAP plot:** `outputs/shap/shap_force_cliente_1.png`

### Cliente Alto Risco #2

- **Probabilidade de churn:** `99.8%`
- **Plano:** Fibra 100MB
- **Valor mensalidade:** R$ 75.40
- **Tempo de contrato:** 3 meses
- **Dias em atraso:** 86 dias
- **NPS score:** 0.0
- **Chamados de suporte:** 9
- **Fidelidade:** Não
- **SHAP plot:** `outputs/shap/shap_force_cliente_2.png`

### Cliente Alto Risco #3

- **Probabilidade de churn:** `99.8%`
- **Plano:** Fibra 100MB
- **Valor mensalidade:** R$ 74.33
- **Tempo de contrato:** 1 meses
- **Dias em atraso:** 70 dias
- **NPS score:** 3.0
- **Chamados de suporte:** 9
- **Fidelidade:** Não
- **SHAP plot:** `outputs/shap/shap_force_cliente_3.png`

---

## 6. Recomendações de Ação por Faixa de Risco

| Faixa de Probabilidade | Segmento | Ação Recomendada |
|------------------------|----------|-----------------|
| **P > 0.75** | Alto Risco | Contato imediato do supervisor comercial (48h). Oferta de cashback ou desconto de 20–30% por fidelidade. Escalar para equipe de retenção. |
| **P 0.50–0.75** | Médio Risco | Campanha de engajamento proativo. Verificar tickets em aberto. Aplicar pesquisa NPS. Oferecer upgrade com desconto. |
| **P < 0.50** | Baixo Risco | Monitoramento mensal. E-mail de relacionamento. Sem ação comercial imediata. |

### Estimativa de Impacto Financeiro

- Considerando retenção de 50% dos clientes de alto risco via ação proativa:
  cada cliente Empresarial retido representa **~R$ 450/mês** em MRR preservado.
- Prioridade: clientes com alto `risco_pagamento` + baixo `nps_score` + sem fidelidade.

---

_Dataset: 15.000 registros sintéticos - Churn rate: 29.7% - Seed: 42_