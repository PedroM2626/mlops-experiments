import argparse
import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def main(forecast_path: str, products_path: str, output_path: str):
    logging.info("Iniciando a Análise Hierárquica por Categoria e Marca.")
    
    try:
        df_forecast = pd.read_parquet(forecast_path)
        df_produtos = pd.read_parquet(products_path)
    except FileNotFoundError as e:
        logging.error(f"Arquivo não encontrado. Verifique os caminhos. Erro: {e}")
        return

    # Junta as previsões com as informações de produto
    df_merged = pd.merge(df_forecast, df_produtos, left_on='produto', right_on='sku', how='inner')
    
    # Agrega as vendas por categoria e marca
    df_agg = df_merged.groupby(['categoria', 'marca'])['quantidade'].sum().reset_index()
    
    # Prepara dados para o treemap
    df_agg['label'] = df_agg['categoria'] + "\n(" + df_agg['marca'] + ")"
    
    # Cria o gráfico
    plt.style.use('default')
    plt.figure(figsize=(16, 9))
    
    # o squarify cria o treemap
    squarify.plot(
        sizes=df_agg['quantidade'],
        label=df_agg['label'],
        alpha=0.8,
        text_kwargs={'fontsize': 9},
        color=sns.color_palette("viridis", len(df_agg))
    )
    
    # Títulos e formatação
    plt.title('Análise Hierárquica: Previsão de Vendas por Categoria e Marca', fontsize=20, weight='bold')
    plt.axis('off')
    
    output_filename = os.path.join(output_path, '02_treemap_categorias_marcas.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    logging.info(f"Gráfico Treemap salvo em: {output_filename}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera um treemap de previsão por categoria e marca.")
    parser.add_argument("--forecast_path", type=str, required=True, help="Caminho para o arquivo de previsão final (.parquet).")
    parser.add_argument("--products_path", type=str, required=True, help="Caminho para o arquivo de dimensão de produtos (dim_produtos.parquet).")
    parser.add_argument("--output_path", type=str, default="visualizacao_estrategica", help="Pasta para salvar os resultados.")
    
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    main(args.forecast_path, args.products_path, args.output_path)