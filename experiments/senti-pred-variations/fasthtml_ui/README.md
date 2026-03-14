# FastHTML UI - Senti Pred Variations

Interface web para:

- mostrar status dos experimentos
- rodar pipelines reais (scripts existentes)
- analisar metricas e visualizacoes geradas

## Funcionalidades

- Home com cartoes de cada experimento e botao de execucao
- Dois modos de execucao: `full` (pipeline completo) e `smoke` (validacao rapida)
- Pagina Runs com status em tempo real (polling)
- Pagina de detalhes por run com logs em streaming
- Pagina Analysis com metricas encontradas e galeria de artefatos

## Como executar

No root do repositorio:

```bash
c:/python314/python.exe -m pip install -r experiments/senti-pred-variations/fasthtml_ui/requirements.txt
c:/python314/python.exe experiments/senti-pred-variations/fasthtml_ui/app.py
```

Abra:

- http://127.0.0.1:8502

## Observacoes

- A interface executa scripts em suas pastas originais, sem copiar arquivos.
- Os logs de execucao ficam em memoria durante a sessao da aplicacao.
- As metricas sao carregadas de JSON/CSV existentes nas pastas de `reports`.
- A rota `/artifact/...` serve imagens e arquivos para visualizacao no browser.
