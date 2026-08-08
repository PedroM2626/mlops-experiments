# Modelo de README Acadêmico por Experimento

> Modelo de referência para documentação de experimentos. Cada experimento
> deve ter o seu próprio `README.md` imediatamente dentro da sua pasta em
> `experiments/<experimento>/`, seguindo a estrutura abaixo (artigo compacto,
> em português).

---

## Estrutura obrigatória

```markdown
# <Título do Experimento>

> **Área:** <NLP | Séries Temporais | Computer Vision | Regressão | RecSys | ...>
> **Tarefa:** <Classificação | Regressão | ...>
> **Métrica principal:** <F1-macro, R², MAE, AUC-ROC, ...>
> **Status:** <Concluído | Em andamento>
> **Datasets:** <origem e tamanho>

## 1. Resumo
<3-5 linhas: problema investigado, método proposto, resultado principal e
conclusão em uma frase. Sem tabelas.>

## 2. Contexto e Objetivos

<O que motivou o estudo; questões de pesquisa; hipóteses (se houver);
referência a trabalhos/problemas anteriores que motivaram.>

## 3. Fundamentação Teórica (curta)

<conceitos-chave para entender o experimento: representações, algoritmos,
métricas. Sem extensões; apenas ligar teoria à decisão do estudo.>

## 4. Metodologia

### 4.1 Dados
- Fonte, tamanho, nº de classes/features, split validação.

### 4.2 Pré-processamento
- Limpeza, transformações, engenharia de features, tratamento de outliers.

### 4.3 Métodos comparados
- Tabela com modelo/paradigma, estratégia e configuração relevantes.

### 4.4 Avaliação
- Métricas, protocolo de validação (holdout/CV/temporal), seeds, hardware.

### 4.5 Reprodução
- Comando(s) e/ou caminho dos notebooks.
- Padrão de saída: `experiments/artifacts/<experimento>_<timestamp>_<sha>/`.

## 5. Resultados

<tabelas comparativas e figuras com valores REAIS obtidos; mencione seed e
data da execução. Nunca inventar valores.>

## 6. Discussão

<interpretação dos resultados, comparação entre métodos, limitações e
possíveis fontes de viés.>

## 7. Conclusões e Recomendações

<bullet points práticos + recomendação de escolha para cenários de uso.>

## 8. Referências e Arquivos

- Link para notebook/scripts/artefatos (caminhos relativos).
- Referências bibliográficas quando aplicável (APIBT/APA curto).
```

---

## Regras de composição

1. **Veracidade**: números/figuras devem refletir execuções reais presentes
   nos notebooks/outputs. Se um valor não estiver disponível, escreva "a
   medir/TBD" e **nunca** invente.
2. **Idioma**: português do Brasil.
3. **Caminhos**: sempre relativos ao diretório do README (ex. `por-ramal: ./feature_selection_ea.py`).
4. **Tabelas**: use tabelas Markdown para resultados comparativos.
5. **Reproducibilidade**: cuide da seção "Reproduzir" — deve permitir que
   qualquer pessoa rode o experimento localmente.
6. **Links do índice**: o README da raiz lista aponta para
   `experiments/<experimento>/README.md`.