import argparse
import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Cria e salva o gráfico de barras divergente para a análise de momentum.
def plot_momentum(df_momentum, output_path):
    if df_momentum.empty:
        logging.warning("DataFrame de momentum vazio. O gráfico não será gerado.")
        return
        
    colors = ['#2ca02c' if x > 0 else '#d62728' for x in df_momentum['momentum_percent']]
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.barplot(x='momentum_percent', y='produto', data=df_momentum, palette=colors, ax=ax)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title('Análise de Momentum: Produtos em Ascensão vs. em Declínio\n(Previsão Jan/2023 vs. Mês Anterior)', fontsize=16, weight='bold', pad=20)
    ax.set_xlabel('Variação Percentual na Média de Vendas Semanal (%)', fontsize=12)
    ax.set_ylabel('Produto (SKU)', fontsize=12)
    ax.xaxis.set_major_formatter(plt.FuncFormatter('{:.0f}%'.format))
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 0.5 * (1 if width > 0 else -1), p.get_y() + p.get_height() / 2,
                f'{width:+.1f}%', va='center', ha='left' if width > 0 else 'right', fontsize=9)
    plt.tight_layout()
    output_filename = os.path.join(output_path, '01_grafico_momentum_produtos.png')
    plt.savefig(output_filename, dpi=300)
    logging.info(f"Gráfico de momentum salvo em: {output_filename}")
    plt.close()

# Função para carregar e pré-processar os dados de vendas históricas.
def preprocess_historical_data(hist_path: str, products_path: str) -> pd.DataFrame:

    logging.info("Pré-processando dados históricos...")
    df_hist = pd.read_parquet(hist_path)
    df_produtos = pd.read_parquet(products_path)
    
    # Junta com produtos para ter o 'sku'
    df_merged = pd.merge(df_hist, df_produtos, left_on='internal_product_id', right_on='produto', how='inner')
    df_merged.rename(columns={'produto': 'sku'}, inplace=True)
    
    # Converte data e agrega por semana
    df_merged['transaction_date'] = pd.to_datetime(df_merged['transaction_date'])
    df_merged['ano'] = df_merged['transaction_date'].dt.isocalendar().year
    df_merged['semana'] = df_merged['transaction_date'].dt.isocalendar().week
    
    df_agg = df_merged.groupby(['ano', 'semana', 'sku']).agg(quantidade=('quantity', 'sum')).reset_index()
    return df_agg

def main(forecast_path: str, historical_path: str, products_path: str, output_path: str):
    logging.info("Iniciando a Análise de Momentum.")
    
    try:
        df_forecast = pd.read_parquet(forecast_path)
        df_hist_agg = preprocess_historical_data(historical_path, products_path)
    except FileNotFoundError as e:
        logging.error(f"Arquivo não encontrado. Verifique os caminhos. Erro: {e}")
        return

    # 1. Calcula a média de vendas PREVISTA por produto
    media_prevista = df_forecast.groupby('produto')['quantidade'].mean().reset_index()
    media_prevista.rename(columns={'quantidade': 'media_prevista'}, inplace=True)
    
    # 2. Calcula a média de vendas HISTÓRICA do último mês de 2022
    hist_recente = df_hist_agg[(df_hist_agg['ano'] == 2022) & (df_hist_agg['semana'] >= 49)]
    media_historica = hist_recente.groupby('sku')['quantidade'].mean().reset_index()
    media_historica.rename(columns={'sku': 'produto', 'quantidade': 'media_historica'}, inplace=True)

    # 3. Une as informações
    df_momentum = pd.merge(media_prevista, media_historica, on='produto', how='inner')
    
    # Filtra produtos com pouca venda histórica para evitar ruído
    df_momentum = df_momentum[df_momentum['media_historica'] > 5]
    
    # 4. Calcula a variação percentual (o "momentum")
    epsilon = 1e-6  # epsilon para evitar divisão por zero
    df_momentum['momentum_percent'] = ((df_momentum['media_prevista'] - df_momentum['media_historica']) / (df_momentum['media_historica'] + epsilon)) * 100
    
    # 5. Seleciona os Top 15 em ascensão e os Top 15 em declínio
    top_ascensao = df_momentum.nlargest(15, 'momentum_percent')
    top_declinio = df_momentum.nsmallest(15, 'momentum_percent')
    df_final = pd.concat([top_ascensao, top_declinio]).sort_values('momentum_percent', ascending=False)
    
    # Salva a tabela estratégica
    output_csv = os.path.join(output_path, '01_tabela_momentum.csv')
    df_final.to_csv(output_csv, index=False)
    logging.info(f"Tabela de momentum salva em: {output_csv}")
    
    # Gera o gráfico
    plot_momentum(df_final, output_path)

if __name__ == "__main__":
    # --- BLOCO DE ARGUMENTOS CORRIGIDO ---
    parser = argparse.ArgumentParser(description="Gera uma análise de momentum dos produtos.")
    parser.add_argument("--forecast_path", type=str, required=True, help="Caminho para o arquivo de previsão final (.parquet).")
    parser.add_argument("--historical_path", type=str, required=True, help="Caminho para o arquivo histórico de vendas (fato_vendas.parquet).")
    parser.add_argument("--products_path", type=str, required=True, help="Caminho para o arquivo de dimensão de produtos (dim_produtos.parquet).")
    parser.add_argument("--output_path", type=str, default="visualizacao_estrategica", help="Pasta para salvar os resultados.")
    
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    main(args.forecast_path, args.historical_path, args.products_path, args.output_path)
