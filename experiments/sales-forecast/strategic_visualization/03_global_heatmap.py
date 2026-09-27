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

# need to convert the zipcodes of the "pdvs" dimension table into coordinates (latitude, longitude).

def geocode_pdvs(pdvs_path: str, output_path: str) -> pd.DataFrame:
    

    cache_file = os.path.join(output_path, 'geocoded_pdvs_cache.parquet')
    if os.path.exists(cache_file):
        logging.info(f"Loading coordinates from the cache file: {cache_file}")
        return pd.read_parquet(cache_file)

    logging.info("Cache file not found. Starting the geocoding process (this may take a while)...")
    df_pdvs = pd.read_parquet(pdvs_path)
    
    # If you don't have city/country, you can try using only the zipcode, but it may be less accurate.
    df_pdvs['full_address'] = df_pdvs['zipcode'].astype(str)

    geolocator = Nominatim(user_agent="hackathon_forecast_analyzer")
    
    latitudes, longitudes = [], []
    
    # tqdm creates a visual progress bar
    for address in tqdm(df_pdvs['full_address'], desc="Geocoding PDVs"):
        try:
            location = geolocator.geocode(address, timeout=10)
            if location:
                latitudes.append(location.latitude)
                longitudes.append(location.longitude)
            else:
                latitudes.append(None)
                longitudes.append(None)
        except Exception as e:
            logging.warning(f"Error geocoding '{address}': {e}")
            latitudes.append(None)
            longitudes.append(None)
        
        # Pause for 1 second to respect the free service's usage policy
        time.sleep(1)

    df_pdvs['latitude'] = latitudes
    df_pdvs['longitude'] = longitudes
    
    # Save the result to the cache for future use
    df_pdvs.to_parquet(cache_file, index=False)
    logging.info(f"Geocoding complete. Results saved to cache: {cache_file}")
    
    return df_pdvs


def main(forecast_path: str, pdvs_path: str, output_path: str):
    logging.info("Starting the Generation of the Global Bubble Map by Zipcode.")
    
    # 1. Geocode or load the coordinates from the cache
    df_pdvs_geocoded = geocode_pdvs(pdvs_path, output_path)
    
    # 2. Load and process the forecasts
    df_forecast = pd.read_parquet(forecast_path)
    vendas_por_pdv = df_forecast.groupby('pdv')['quantidade'].sum().reset_index()
    
    # 3. Combine the forecasts with the geocoded data
    df_plot = pd.merge(df_pdvs_geocoded, vendas_por_pdv, on='pdv', how='inner')
    df_plot.dropna(subset=['latitude', 'longitude'], inplace=True)
    
    # --- Map Creation ---
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # Plot the world map as the base
    world.plot(ax=ax, color='#e0e0e0', edgecolor='white')

    # Plot the scatter plot on top of the map
    scatter = ax.scatter(
        df_plot['longitude'],
        df_plot['latitude'],
        s=df_plot['quantidade'] / 100,  # scale factor for the bubble size
        c=df_plot['quantidade'],
        cmap='viridis',
        alpha=0.7,
        edgecolor='k',
        linewidth=0.5
    )

    # --- Formatting and Legends ---
    ax.set_title('Global Heatmap: Sales Forecast by Location (PDV)\nJanuary 2023', fontsize=22, weight='bold', pad=20)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, linestyle='--', alpha=0.5)

    # Color legend
    cbar = fig.colorbar(scatter, shrink=0.5, orientation='horizontal', pad=0.01)
    cbar.set_label('Total Forecast Quantity', weight='bold')

    # Bubble size legend
    for sales in [df_plot['quantidade'].min(), df_plot['quantidade'].median(), df_plot['quantidade'].max()]:
        ax.scatter([], [], s=sales/100, c='k', alpha=0.5, label=f'{int(sales)} units')
    ax.legend(scatterpoints=1, frameon=False, labelspacing=1.5, title='Sales Volume', loc='lower left')

    output_filename = os.path.join(output_path, '03_mapa_bolhas_zipcode.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    logging.info(f"Bubble map saved to: {output_filename}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a global bubble map of the sales forecast by PDV.")
    parser.add_argument("--forecast_path", type=str, required=True, help="Path to the final forecast file (.parquet).")
    parser.add_argument("--pdvs_path", type=str, required=True, help="Path to the PDVs dimension file (dim_pdvs.parquet).")
    parser.add_argument("--output_path", type=str, default="strategic_visualization", help="Folder to save the results.")
    
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    main(args.forecast_path, args.pdvs_path, args.output_path)