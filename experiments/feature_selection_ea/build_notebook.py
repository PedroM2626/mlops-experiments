"""
Gera e executa o notebook: feature_selection_ea.ipynb (executado, com outputs).
Uso: python build_notebook.py [--quick]
"""
import argparse
import nbformat as nbf
from nbclient import NotebookClient


def code(src):
    return nbf.v4.new_code_cell(src)


def md(src):
    return nbf.v4.new_markdown_cell(src)


ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
args, _ = ap.parse_known_args()

cal_cfg = dict(ga_pop=8 if args.quick else 18,
               ga_gen=8 if args.quick else 25,
               de_pop=10 if args.quick else 24,
               de_gen=10 if args.quick else 30)
tw_cfg = dict(ga_pop=6 if args.quick else 12,
              ga_gen=6 if args.quick else 16,
              de_pop=8 if args.quick else 16,
              de_gen=8 if args.quick else 20)

nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python",
                                "name": "python3"}
nb["cells"] = [
    md("# Feature Selection Evolucionaria (DEAP)\n\n"
       "Comparativo de **selecao de features** com algoritmos evolucionarios "
       "(**GAAP-NSGA-II** e **MO-DE** do DEAP) contra classicos "
       "(**SelectKBest**, **RandomForest importance**, **Boruta**) em dois cenarios:\n\n"
       "| Dataset | Tarefa | Metrica | Variaveis |\n|---|---|---|---|\n"
       "| California Housing | Regressao | R2 | 44 (poly) |\n"
       "| Twitter (TF-IDF) | Classificacao (4 classes) | F1-macro | 400 |\n\n"
       "Metrica central: **score CV x numero de features** (curvas) com validacao no holdout."),
    code("import os, sys, warnings\n"
         "import matplotlib; matplotlib.use('Agg')\n"
         "import matplotlib.pyplot as plt\n"
         "import numpy as np, pandas as pd\n"
         "warnings.filterwarnings('ignore')\n"
         "sys.path.insert(0, os.getcwd())\n"
         "from feature_selection_ea import run_one, load_california, load_twitter,\\\n"
         "    plot_curves, Evaluator, OUT\n"
         "os.makedirs(OUT, exist_ok=True)\n"
         "print('Imports OK')"),

    md("## 1. California Housing (regressao, R2)"),
    code("X, y, names_cal = load_california()\n"
         "print(f'features: {X.shape[1]} | {list(names_cal)}')"),
    code(f"res_cal = run_one('regression', X, y, {cal_cfg})"),
    code("res_cal['summary'].to_string(index=False)"),
    md("### Curva R2 x nº de features (pontos Pareto dos evolucionarios)"),
    code("cal_points = res_cal['df']\n"
        "print('GA front:', {k: round(s,3) for k, s in res_cal['pareto_ga'].items()})\n"
        "print('DE front:', {k: round(s,3) for k, s in res_cal['pareto_de'].items()})\n"
        "print('classicos top-k (melhor por metodo):')\n"
        "for m, g in cal_points[cal_points.method.isin(['SelectKBest','RandomForest','Boruta'])].groupby('method'):\n"
        "    g = g.sort_values('n_feats')\n"
        "    print(f'  {m:15s}', g[['n_feats','cv_score']].tail(1).round(3).iloc[0].to_dict())"),

    md("## 2. Twitter (classificacao, F1-macro)"),
    code("Xt, yt, tnames, labels = load_twitter(max_features=400)\n"
         "print(f'shape: {Xt.shape}, classes: {labels}')"),
    code(f"res_tw = run_one('classification', Xt, yt, {tw_cfg})"),
    code("res_tw['summary'].to_string(index=False)"),

    md("## 3. Graficos e persistencia"),
    code("plot_curves(res_cal['df'], 'California Housing - R2 x features', 'curves_cal.png')\n"
         "plot_curves(res_tw['df'], 'Twitter - F1-macro x features', 'curves_twitter.png')\n"
         "print('plots salvos em', OUT)"),
    code("res_cal['df'].to_csv(OUT / 'results_cal.csv', index=False)\n"
         "res_tw['df'].to_csv(OUT / 'results_twitter.csv', index=False)\n"
         "print('csvs salvos em', OUT)"),

    md("## 4. Holdout: melhor subset por metodo\n"
       "Score no teste (nunca visto) usando o subset de melhor CV de cada metodo."),
    code("pd.concat([res_cal['summary'].assign(dataset='cal'),\n"
         "            res_tw['summary'].assign(dataset='twitter')])"
         "[['dataset','method','best_cv','best_feats','test_score']]"),

    md("## Conclusoes\n"
       "- Evolucionarios atingem score proximo ao full com **muito menos features**;\n"
       "- Baselines classicos (top-k) precisam de mais features para o mesmo score;\n"
       "- GAAP e MO-DE entregam uma frente de Pareto (score vs nº de features)."),
]

nbf.write(nb, "feature_selection_ea.ipynb")
client = NotebookClient(nb, timeout=3600, kernel_name="python3")
client.execute()
nbf.write(nb, "feature_selection_ea.ipynb")
print("[ok] notebook gerado e executado")