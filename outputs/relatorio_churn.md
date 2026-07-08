# Relatório de Análise de Churn — FiberNet ISP

_Gerado automaticamente por pipeline.py - Dados sintéticos - Hugo Leonardo_

---

## 1. Modelo Escolhido e Justificativa

**Modelo selecionado: `XGBoost`**

Escolhido por apresentar o maior AUC-ROC médio em validação cruzada estratificada (5-fold), com AUC = **0.9960** no conjunto de teste. Modelos tree-based são preferíveis neste contexto por capturarem interações não-lineares entre features de risco (atraso, NPS, suporte) sem exigir normalidade dos dados.

---

## 2. Tabela Comparativa dos 4 Modelos

| Modelo             |   CV AUC (μ) |   CV AUC (σ) |   AUC-ROC |     F1 |   Precisão |   Recall |   Brier |
|:-------------------|-------------:|-------------:|----------:|-------:|-----------:|---------:|--------:|
| LogisticRegression |       0.9934 |       0.0012 |    0.9946 | 0.9409 |     0.9267 |   0.9556 |  0.0268 |
| RandomForest       |       0.9942 |       0.001  |    0.9951 | 0.952  |     0.9573 |   0.9467 |  0.023  |
| XGBoost            |       0.9949 |       0.001  |    0.996  | 0.9544 |     0.9544 |   0.9544 |  0.0217 |
| LightGBM           |       0.9948 |       0.001  |    0.996  | 0.953  |     0.9493 |   0.9567 |  0.0214 |

> Métricas calculadas no conjunto de teste (20% holdout, estratificado).
> Threshold padrão = 0.50.

---

## 3. Top 10 Features por Importância (SHAP)

| Rank | Feature | SHAP médio |
|------|---------|-----------|
| 1 | `meses_sem_incidente` | 2.5761 |
| 2 | `score_satisfacao` | 1.5148 |
| 3 | `score_risco_total` | 1.2588 |
| 4 | `qtd_chamados_suporte` | 0.6966 |
| 5 | `pressao_suporte` | 0.5328 |
| 6 | `nps_score` | 0.4139 |
| 7 | `dias_atraso_pagamento` | 0.3200 |
| 8 | `log_score_risco` | 0.3122 |
| 9 | `tempo_contrato` | 0.2285 |
| 10 | `valor_mensalidade` | 0.1870 |

> Importância global = média do valor absoluto dos SHAP values no conjunto de teste.

---

## 4. Threshold Ótimo e Impacto na Precision/Recall

| Métrica | Threshold = 0.50 | Threshold ótimo = 0.522 |
|---------|-----------------|--------------------------------------|
| F1-Score  | 0.9544 | 0.9549 |
| Precisão  | 0.9544 | 0.9575 |
| Recall    | 0.9544 | 0.9522 |
| AUC-ROC   | 0.9960 | 0.9960 |

**Threshold ótimo = 0.522** — calculado pelo máximo F1 na curva Precision-Recall.
Recall é priorizado (não perder churners reais) em detrimento de precisão.

### Calibração de Probabilidade

| | Brier Score |
|---|-------------|
| Modelo original  | 0.0217 |
| Após calibração (isotonic) | 0.0212 |

---

## 5. Perfis de Clientes de Alto Risco

### Cliente Alto Risco #1

- **Probabilidade de churn:** `100.0%`
- **Plano:** Fibra 200MB
- **Valor mensalidade:** R$ 97.59
- **Tempo de contrato:** 1 meses
- **Dias em atraso:** 15 dias
- **NPS score:** 3.7
- **Chamados de suporte:** 9
- **Fidelidade:** Não
- **SHAP plot:** `outputs/shap/shap_force_cliente_1.png`

### Cliente Alto Risco #2

- **Probabilidade de churn:** `100.0%`
- **Plano:** Fibra 100MB
- **Valor mensalidade:** R$ 83.51
- **Tempo de contrato:** 1 meses
- **Dias em atraso:** 30 dias
- **NPS score:** 3.6
- **Chamados de suporte:** 4
- **Fidelidade:** Não
- **SHAP plot:** `outputs/shap/shap_force_cliente_2.png`

### Cliente Alto Risco #3

- **Probabilidade de churn:** `100.0%`
- **Plano:** Fibra 100MB
- **Valor mensalidade:** R$ 84.40
- **Tempo de contrato:** 2 meses
- **Dias em atraso:** 47 dias
- **NPS score:** 4.2
- **Chamados de suporte:** 4
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

_Dataset: 15.000 registros sintéticos - Churn rate: 30.0% - Seed: 42_