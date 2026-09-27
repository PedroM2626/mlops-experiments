import argparse
import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def main(forecast_path: str, products_path: str, output_path: str):
    logging.info("Starting the Hierarchical Analysis by Category and Brand.")
    
    try:
        df_forecast = pd.read_parquet(forecast_path)
        df_produtos = pd.read_parquet(products_path)
    except FileNotFoundError as e:
        logging.error(f"File not found. Check the paths. Error: {e}")
        return

    # Join the forecasts with the product information
    df_merged = pd.merge(df_forecast, df_produtos, left_on='produto', right_on='sku', how='inner')
    
    # Aggregate sales by category and brand
    df_agg = df_merged.groupby(['categoria', 'marca'])['quantidade'].sum().reset_index()
    
    # Prepare the data for the treemap
    df_agg['label'] = df_agg['categoria'] + "\n(" + df_agg['marca'] + ")"
    
    # Create the chart
    plt.style.use('default')
    plt.figure(figsize=(16, 9))
    
    # squarify builds the treemap
    squarify.plot(
        sizes=df_agg['quantidade'],
        label=df_agg['label'],
        alpha=0.8,
        text_kwargs={'fontsize': 9},
        color=sns.color_palette("viridis", len(df_agg))
    )
    
    # Titles and formatting
    plt.title('Hierarchical Analysis: Sales Forecast by Category and Brand', fontsize=20, weight='bold')
    plt.axis('off')
    
    output_filename = os.path.join(output_path, '02_treemap_categorias_marcas.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    logging.info(f"Treemap chart saved to: {output_filename}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a treemap of the forecast by category and brand.")
    parser.add_argument("--forecast_path", type=str, required=True, help="Path to the final forecast file (.parquet).")
    parser.add_argument("--products_path", type=str, required=True, help="Path to the products dimension file (dim_produtos.parquet).")
    parser.add_argument("--output_path", type=str, default="strategic_visualization", help="Folder to save the results.")
    
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    main(args.forecast_path, args.products_path, args.output_path)