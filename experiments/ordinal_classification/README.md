# Classificação Ordinal vs. Nominal — Wine Quality (Red)

> **Área:** Classificação ordinal / Tabular
> **Tarefa:** Predizer qualidade do vinho (notas 3–8, 6 classes ordenadas) como problema ordinal vs. nominal
> **Métricas:** Acurácia, MAE (distância ordinal), Kappa de Cohen, Acurácia ±1
> **Status:** Concluído
> **Dataset:** Wine Quality Red (UCI) — 1.599 linhas × 12 colunas; split 1.199 treino / 400 teste (seed 42). Fallback sintético ordinal se o download falhar.

## 1. Resumo

Compara 4 abordagens no mesmo split: LogReg nominal, RF nominal, LogisticAT ordinal (`mord`) e LogisticIT ordinal (`mord`). O **RF nominal vence em acurácia (0.6600) e MAE (0.3600)**, mas os modelos ordinais empatam em **acurácia ±1 (~0.9775)** — erram "perto". A lição: para rótulos ordenados, MAE/Kappa/acc±1 contam mais que acurácia exata; modelo ordinal raramente vence o RF em acc pura, porém produz erros menos graves.

## 2. Contexto e Objetivos

Classificação nominal ignora a ordem (errar 5→8 custa o mesmo que 5→6). Classificação ordinal penaliza a distância. Questões: (RQ1) o modelo ordinal supera o nominal em MAE/Kappa? (RQ2) acc±1 revela equivalência prática?

## 3. Fundamentação Teórica (curta)

- **Nominal (LogReg OvR, RF):** fronteiras por classe, sem noção de vizinhança ordinal.
- **Ordinal (LogisticAT/IT, `mord`):** limiares cumulativos sobre escore latente; AT (all-threshold) vs IT (immediate-threshold).
- **OrdinalRandomForest (notebook):** decomposição em K−1 binários `P(y ≥ k)`, classe final = soma das predições.
- **Métricas ordinais:** MAE = média |ŷ−y|; Kappa pondera concordância além do acaso; acc±1 = fração com erro ≤ 1 nível.

## 4. Metodologia

### 4.1 Dados

Wine Quality Red: 1.599 amostras; distribuição por qualidade: 3:10, 4:53, 5:681, 6:638, 7:199, 8:18 (forte desbalanceamento, classes extremas raras). `y −= y.min()` para 0..5 (`mord` exige 0..K−1).

### 4.2 Métodos comparados

| Modelo | Tipo | Config |
|---|---|---|
| LogReg (Nominal) | nominal | `StandardScaler + LogisticRegression(max_iter=2000)` |
| RF (Nominal) | nominal | `RandomForestClassifier(n_estimators=200, random_state=42)` |
| LogisticAT (Ordinal) | ordinal | `StandardScaler + mord.LogisticAT(alpha=1.0)` |
| LogisticIT (Ordinal) | ordinal | `StandardScaler + mord.LogisticIT(alpha=1.0)` |

### 4.3 Avaliação

Holdout único 75/25 estratificado (1.199/400). Métricas: acurácia, MAE, Kappa, acc±1. Figuras: barras por métrica, matrizes de confusão, distribuição de erros |ŷ−y|.

### 4.4 Reprodução

```bash
jupyter nbconvert --to notebook --execute experiments/ordinal_classification/ordinal_classification.ipynb --inplace
pip install mord scikit-learn pandas matplotlib seaborn
```

## 5. Resultados

| Modelo | Tipo | Acurácia | MAE | Kappa | Acc ±1 |
|---|---|---|---|---|---|
| **RF (Nominal)** | nominal | **0.6600** | **0.3600** | **0.4451** | **0.9800** |
| LogisticIT (Ordinal) | ordinal | 0.5975 | 0.4275 | 0.3285 | 0.9775 |
| LogReg (Nominal) | nominal | 0.5950 | 0.4375 | 0.3286 | 0.9700 |
| LogisticAT (Ordinal) | ordinal | 0.5850 | 0.4400 | 0.3082 | 0.9775 |

## 6. Discussão

- **RF domina tudo** (acc +0.06, MAE −0.07 vs 2º): árvores capturam não-linearidades químicas que limiares lineares não capturam.
- **Ordinais não superam o nominal correspondente** (LogReg 0.5950 vs LogisticIT 0.5975 — empate; MAE 0.4375 vs 0.4275 — ganho marginal). O ganho ordinal aparece em acc±1 (0.9775 vs 0.9700): erros "de 1 nível".
- **Classes raras (3, 8) são quase nunca acertadas** — Kappa 0.31–0.45 reflete isso; sem rebalanceamento ou loss ordinal ponderada, o modelo colapsa para 5/6.
- **Limitações:** holdout único (sem CV); 1 seed; `mord` linear (sem kernel); sem calibração de limiares por classe.

## 7. Conclusões e Recomendações

- Se a métrica de negócio tolera erro de ±1 nível (ex.: faixa de qualidade), **qualquer modelo serve** (≥0.97) — escolha o mais simples.
- Se erro grave custa caro, **RF nominal + monitoramento de MAE** é o melhor custo-benefício aqui; ordinal linear só vale com restrição de interpretabilidade monotônica.
- Próximos: CV estratificado ×5, `class_weight=balanced`, threshold tuning por classe, RF ordinal (K−1 binários) como meio-termo.

## 8. Referências e Arquivos

- Notebook: `./ordinal_classification.ipynb` (executado, com figuras).
- Referências: Pedregosa et al. (`mord`); UCI Wine Quality (Cortez et al., 2009); Cohen (1960) Kappa; ver `docs/modelo-academico-readme.md`.
