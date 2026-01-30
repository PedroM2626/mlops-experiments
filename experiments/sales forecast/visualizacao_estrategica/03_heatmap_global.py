import argparse
import logging
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from tqdm import tqdm
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# precisa converter os zipcodes da tabela dimensão "pdvs" em coordenadas (latitude, longitude).

def geocode_pdvs(pdvs_path: str, output_path: str) -> pd.DataFrame:
    

    cache_file = os.path.join(output_path, 'geocoded_pdvs_cache.parquet')
    if os.path.exists(cache_file):
        logging.info(f"Carregando coordenadas do arquivo de cache: {cache_file}")
        return pd.read_parquet(cache_file)

    logging.info("Arquivo de cache não encontrado. Iniciando processo de geocodificação (pode levar tempo)...")
    df_pdvs = pd.read_parquet(pdvs_path)
    
    # Se você não tiver cidade/país, pode tentar usar apenas o zipcode, mas pode ser menos preciso.
    df_pdvs['full_address'] = df_pdvs['zipcode'].astype(str)

    geolocator = Nominatim(user_agent="hackathon_forecast_analyzer")
    
    latitudes, longitudes = [], []
    
    # tqdm cria uma barra de progresso visual
    for address in tqdm(df_pdvs['full_address'], desc="Geocodificando PDVs"):
        try:
            location = geolocator.geocode(address, timeout=10)
            if location:
                latitudes.append(location.latitude)
                longitudes.append(location.longitude)
            else:
                latitudes.append(None)
                longitudes.append(None)
        except Exception as e:
            logging.warning(f"Erro ao geocodificar '{address}': {e}")
            latitudes.append(None)
            longitudes.append(None)
        
        # Pausa de 1 segundo para respeitar a política de uso do serviço gratuito
        time.sleep(1)

    df_pdvs['latitude'] = latitudes
    df_pdvs['longitude'] = longitudes
    
    # Salva o resultado no cache para uso futuro
    df_pdvs.to_parquet(cache_file, index=False)
    logging.info(f"Geocodificação concluída. Resultados salvos em cache: {cache_file}")
    
    return df_pdvs


def main(forecast_path: str, pdvs_path: str, output_path: str):
    logging.info("Iniciando a Geração do Mapa de Bolhas Global por Zipcode.")
    
    # 1. Geocodifica ou carrega as coordenadas do cache
    df_pdvs_geocoded = geocode_pdvs(pdvs_path, output_path)
    
    # 2. Carrega e processa as previsões
    df_forecast = pd.read_parquet(forecast_path)
    vendas_por_pdv = df_forecast.groupby('pdv')['quantidade'].sum().reset_index()
    
    # 3. Une as previsões com os dados geocodificados
    df_plot = pd.merge(df_pdvs_geocoded, vendas_por_pdv, on='pdv', how='inner')
    df_plot.dropna(subset=['latitude', 'longitude'], inplace=True)
    
    # --- Criação do Mapa ---
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # Plota o mapa-mundi como base
    world.plot(ax=ax, color='#e0e0e0', edgecolor='white')

    # Plota as scatter plot sobre o mapa
    scatter = ax.scatter(
        df_plot['longitude'],
        df_plot['latitude'],
        s=df_plot['quantidade'] / 100,  # fator de escala para o tamanho da bolha
        c=df_plot['quantidade'],
        cmap='viridis',
        alpha=0.7,
        edgecolor='k',
        linewidth=0.5
    )

    # --- Formatação e Legendas ---
    ax.set_title('Mapa de Calor Global: Previsão de Vendas por Localidade (PDV)\nJaneiro 2023', fontsize=22, weight='bold', pad=20)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, linestyle='--', alpha=0.5)

    # Legenda da cor
    cbar = fig.colorbar(scatter, shrink=0.5, orientation='horizontal', pad=0.01)
    cbar.set_label('Quantidade Total Prevista', weight='bold')

    # Legenda do tamanho das bolhas
    for- sales in [df_plot['quantidade'].min(), df_plot['quantidade'].median(), df_plot['quantidade'].max()]:
        ax.scatter([], [], s=sales/100, c='k', alpha=0.5, label=f'{int(sales)} unidades')
    ax.legend(scatterpoints=1, frameon=False, labelspacing=1.5, title='Volume de Vendas', loc='lower left')

    output_filename = os.path.join(output_path, '03_mapa_bolhas_zipcode.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    logging.info(f"Mapa de bolhas salvo em: {output_filename}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera um mapa de bolhas global de previsão de vendas por PDV.")
    parser.add_argument("--forecast_path", type=str, required=True, help="Caminho para o arquivo de previsão final (.parquet).")
    parser.add_argument("--pdvs_path", type=str, required=True, help="Caminho para o arquivo de dimensão de PDVs (dim_pdvs.parquet).")
    parser.add_argument("--output_path", type=str, default="visualizacao_estrategica", help="Pasta para salvar os resultados.")
    
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    main(args.forecast_path, args.pdvs_path, args.output_path)