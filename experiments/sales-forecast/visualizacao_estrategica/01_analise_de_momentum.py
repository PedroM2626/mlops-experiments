import argparse
import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Create and save the diverging bar chart for the momentum analysis.
def plot_momentum(df_momentum, output_path):
    if df_momentum.empty:
        logging.warning("Empty momentum DataFrame. The chart will not be generated.")
        return
        
    colors = ['#2ca02c' if x > 0 else '#d62728' for x in df_momentum['momentum_percent']]
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.barplot(x='momentum_percent', y='produto', data=df_momentum, palette=colors, ax=ax)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title('Momentum Analysis: Rising vs. Declining Products\n(Forecast Jan/2023 vs. Previous Month)', fontsize=16, weight='bold', pad=20)
    ax.set_xlabel('Percent Change in the Weekly Sales Average (%)', fontsize=12)
    ax.set_ylabel('Product (SKU)', fontsize=12)
    ax.xaxis.set_major_formatter(plt.FuncFormatter('{:.0f}%'.format))
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 0.5 * (1 if width > 0 else -1), p.get_y() + p.get_height() / 2,
                f'{width:+.1f}%', va='center', ha='left' if width > 0 else 'right', fontsize=9)
    plt.tight_layout()
    output_filename = os.path.join(output_path, '01_grafico_momentum_produtos.png')
    plt.savefig(output_filename, dpi=300)
    logging.info(f"Momentum chart saved to: {output_filename}")
    plt.close()

# Function to load and preprocess the historical sales data.
def preprocess_historical_data(hist_path: str, products_path: str) -> pd.DataFrame:

    logging.info("Preprocessing historical data...")
    df_hist = pd.read_parquet(hist_path)
    df_produtos = pd.read_parquet(products_path)
    
    # Merge with products to get the 'sku'
    df_merged = pd.merge(df_hist, df_produtos, left_on='internal_product_id', right_on='produto', how='inner')
    df_merged.rename(columns={'produto': 'sku'}, inplace=True)
    
    # Convert date and aggregate by week
    df_merged['transaction_date'] = pd.to_datetime(df_merged['transaction_date'])
    df_merged['ano'] = df_merged['transaction_date'].dt.isocalendar().year
    df_merged['semana'] = df_merged['transaction_date'].dt.isocalendar().week
    
    df_agg = df_merged.groupby(['ano', 'semana', 'sku']).agg(quantidade=('quantity', 'sum')).reset_index()
    return df_agg

def main(forecast_path: str, historical_path: str, products_path: str, output_path: str):
    logging.info("Starting the Momentum Analysis.")
    
    try:
        df_forecast = pd.read_parquet(forecast_path)
        df_hist_agg = preprocess_historical_data(historical_path, products_path)
    except FileNotFoundError as e:
        logging.error(f"File not found. Check the paths. Error: {e}")
        return

    # 1. Compute the FORECAST sales average per product
    media_prevista = df_forecast.groupby('produto')['quantidade'].mean().reset_index()
    media_prevista.rename(columns={'quantidade': 'media_prevista'}, inplace=True)
    
    # 2. Compute the HISTORICAL sales average for the last month of 2022
    hist_recente = df_hist_agg[(df_hist_agg['ano'] == 2022) & (df_hist_agg['semana'] >= 49)]
    media_historica = hist_recente.groupby('sku')['quantidade'].mean().reset_index()
    media_historica.rename(columns={'sku': 'produto', 'quantidade': 'media_historica'}, inplace=True)

    # 3. Combine the information
    df_momentum = pd.merge(media_prevista, media_historica, on='produto', how='inner')
    
    # Filter out products with little historical sales to avoid noise
    df_momentum = df_momentum[df_momentum['media_historica'] > 5]
    
    # 4. Compute the percent change (the "momentum")
    epsilon = 1e-6  # epsilon to avoid division by zero
    df_momentum['momentum_percent'] = ((df_momentum['media_prevista'] - df_momentum['media_historica']) / (df_momentum['media_historica'] + epsilon)) * 100
    
    # 5. Select the Top 15 rising and the Top 15 declining
    top_ascensao = df_momentum.nlargest(15, 'momentum_percent')
    top_declinio = df_momentum.nsmallest(15, 'momentum_percent')
    df_final = pd.concat([top_ascensao, top_declinio]).sort_values('momentum_percent', ascending=False)
    
    # Save the strategic table
    output_csv = os.path.join(output_path, '01_tabela_momentum.csv')
    df_final.to_csv(output_csv, index=False)
    logging.info(f"Momentum table saved to: {output_csv}")
    
    # Generate the chart
    plot_momentum(df_final, output_path)

if __name__ == "__main__":
    # --- FIXED ARGUMENTS BLOCK ---
    parser = argparse.ArgumentParser(description="Generate a momentum analysis of the products.")
    parser.add_argument("--forecast_path", type=str, required=True, help="Path to the final forecast file (.parquet).")
    parser.add_argument("--historical_path", type=str, required=True, help="Path to the historical sales file (fato_vendas.parquet).")
    parser.add_argument("--products_path", type=str, required=True, help="Path to the products dimension file (dim_produtos.parquet).")
    parser.add_argument("--output_path", type=str, default="visualizacao_estrategica", help="Folder to save the results.")
    
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    main(args.forecast_path, args.historical_path, args.products_path, args.output_path)
