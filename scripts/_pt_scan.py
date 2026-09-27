#!/usr/bin/env python3
"""Scan tracked text files for Portuguese prose. Diagnostic only (scratch tool)."""
import json
import os
import re
import subprocess
import sys

# Unambiguous Portuguese function words / forms. English collisions avoided.
PT_WORDS = {
    "não", "nas", "nos", "são", "está", "estão", "este", "esta", "esses", "essas",
    "aqui", "ali", "lá", "porque", "pois", "também", "tão", "mesmo", "próprio",
    "resultados", "conclusão", "análise", "objetivo", "hipótese", "variável",
    "métrica", "gráfico", "tabela", "média", "desvio", "padrão", "amostra",
    "quando", "enquanto", "antes", "depois", "apenas", "assim", "logo", "porém",
    "entretanto", "visto", "dado", "dados", "base", "onde", "cada", "todos",
    "todas", "muito", "muitos", "pouco", "melhor", "pior", "vamos", "fizemos",
    "observa-se", "note-se", "percebe", "vemos", "mostra", "apresenta",
    "utilizamos", "usamos", "treinamento", "treinar", "previsão", "prever",
    "classificação", "regressão", "agrupamento", "segmentação", "predição",
    "predizer", "avaliar", "avaliação", "validação", "cruzada", "recorte",
    "janela", "série", "temporais", "temporal", "hierárquico", "hierárquica",
    "flat", "comparação", "comparar", "escolhemos", "decidimos", "verificamos",
    "validamos", "rodamos", "executamos", "obtemos", "chegamos", "ficou",
    "ficaram", "deu", "deram", "seria", "seriam", "haver", "existe", "existem",
    "fiz", "fez", "deixamos", "mantivemos", "removemos", "adicionamos",
    "incluindo", "considerando", "levando", "seguindo", "usando", "criando",
    "gerando", "testando", "ajustando", "selecionando", "nossa", "nosso",
    "desta", "desse", "dessa", "daquela", "daquele", "nele", "nela", "sobre",
    "entre", "até", "após", "durante", "mediante", "través", "graças",
    "acima", "abaixo", "junto", "separado", "separada", "inteiro", "inteira",
    "quebrado", "quebrada", "errado", "errada", "certo", "certa", "livro",
    "distribuição", "probabilidade", "amostral", "populacional", "residual",
    "ajustado", "ajustada", "ponderado", "ponderada", "normalizado", "cap",
    "cauda", "pico", "vale", "linha", "linhas", "coluna", "colunas", "chave",
    "nível", "níveis", "faixa", "faixas", "corte", "cortes", "lista", "listas",
    "arquivo", "arquivos", "pasta", "pastas", "código", "códigos", "trecho",
    "texto", "textos", "palavra", "palavras", "frase", "frases", "modelo",
    "modelos", "experimento", "experimentos", "hipóteses", "suposição",
}
ACCENT_RE = re.compile(r"[ãõçáéíóúâêôàü]", re.I)
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'\-_]*")

# phrases that strongly indicate PT prose
PHRASES = [
    "de dados", "para dados", "a base", "em seguida", "dessa forma", "desta forma",
    "a seguir", "no gráfico", "na tabela", "nos dados", "não há", "não é",
    "são os", "vamos a", "a fim de", "com o objetivo", "de modo que",
    "é possível", "é importante", "observe que", "note que", "vale ressaltar",
    "em resumo", "por fim", "a partir", "partir de", "na prática", "de fato",
]


def score_line(line):
    low = line.lower()
    tokens = TOKEN_RE.findall(line)
    if not tokens:
        return 0, []
    hits = [t for t in tokens if t.lower() in PT_WORDS]
    phrase_hits = [p for p in PHRASES if p in low]
    accent_hits = 0
    if ACCENT_RE.search(line):
        for t in tokens:
            if ACCENT_RE.search(t) and t.lower() not in PT_WORDS:
                accent_hits += 1
    s = len(hits) * 2 + len(phrase_hits) * 3 + accent_hits
    return s, hits + phrase_hits


def main():
    repo = sys.argv[1] if len(sys.argv) > 1 else "."
    files = subprocess.run(
        ["git", "ls-files", "-z"], cwd=repo, capture_output=True
    ).stdout.split(b"\0")
    report = {}
    for raw in files:
        try:
            rel = raw.decode("utf-8")
        except UnicodeDecodeError:
            rel = raw.decode("unicode_escape")
        if not rel:
            continue
        path = os.path.join(repo, rel.replace("\\", "/"))
        ext = os.path.splitext(rel)[1].lower()
        if ext not in (".md", ".py", ".ipynb", ".txt", ".html", ".js", ".yml",
                       ".yaml", ".tex", ".json", ".sh", ".cfg", ".ini",
                       ".example", ""):
            continue
        if not os.path.isfile(path):
            continue
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        if size > 8_000_000:
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                content = fh.read()
        except (UnicodeDecodeError, OSError):
            continue
        marked = []
        if ext == ".ipynb":
            marked = nb_lines(content)
        else:
            for i, line in enumerate(content.splitlines(), 1):
                s, why = score_line(line)
                if s >= 4:
                    marked.append((i, s, line.strip()[:160]))
        if marked:
            report[rel] = marked
    total = 0
    for rel in sorted(report, key=lambda r: -len(report[r])):
        print(f"### {rel}  ({len(report[rel])} lines)")
        total += len(report[rel])
    print(f"\nTOTAL flagged lines: {total}  across {len(report)} files")
    with open(os.path.join(repo, "scripts/_pt_report.json"), "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in report.items()}, fh, ensure_ascii=False, indent=1)


def nb_lines(content):
    out = []
    try:
        nb = json.loads(content)
    except Exception:
        return out
    for ci, cell in enumerate(nb.get("cells", [])):
        src = "".join(cell.get("source", []))
        base = 0
        for j, line in enumerate(src.splitlines(), 1):
            s, _ = score_line(line)
            if s >= 4:
                out.append((ci * 1000 + j, s, line.strip()[:160]))
        for oi, o in enumerate(cell.get("outputs", []) or []):
            txt = ""
            if isinstance(o.get("text"), list):
                txt = "".join(o["text"])
            elif isinstance(o.get("text"), str):
                txt = o["text"]
            data = o.get("data", {}) or {}
            plain = data.get("text/plain")
            if isinstance(plain, list):
                txt += "".join(plain)
            elif isinstance(plain, str):
                txt += plain
            for j, line in enumerate(txt.splitlines(), 1):
                s, _ = score_line(line)
                if s >= 6:
                    out.append((10_000 + oi, s, f"[out {cell.get('cell_type')} c{ci}] {line.strip()[:120]}"))
    return out


if __name__ == "__main__":
    main()
