import numpy as np, pandas as pd, warnings, time
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import xgboost as xgb

print("=" * 95)
print("TIME SERIES FORECASTING: SUPERVISIONADO vs NÃO SUP supervisionado")
print("=" * 95)

# ============================================================
# 1. Dataset: Air Passengers + Mauna Loa CO2
# ============================================================
print("\n>> Carregando datasets...")

# Air Passengers
air = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv",
                   parse_dates=["Month"])
air.columns = ["ds", "y"]
air["ds"] = pd.to_datetime(air["ds"])
print(f"  Air Passengers: {len(air)} meses, {air['y'].min():.0f}-{air['y'].max():.0f}")

# Sunspots (clássico para ciclos)
sun = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv",
                   parse_dates=["Month"])
sun.columns = ["ds", "y"]
sun["ds"] = pd.to_datetime(sun["ds"])
print(f"  Sunspots: {len(sun)} meses, {sun['y'].min():.1f}-{sun['y'].max():.1f}")

# ============================================================
# 2. Funções auxiliares
# ============================================================
def create_lag_features(series, lags=12):
    """Transforma série temporal em features supervisionadas (lags)."""
    df = pd.DataFrame({"y": series})
    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = series.shift(lag)
    # Rolling stats
    for w in [3, 6, 12]:
        df[f"roll_mean_{w}"] = series.rolling(w, min_periods=1).mean().shift(1)
        df[f"roll_std_{w}"] = series.rolling(w, min_periods=1).std().fillna(0).shift(1)
    return df.dropna().reset_index(drop=True)

def forecast_supervised(model, train_y, test_y, lags=12):
    """Treina modelo supervisionado com lags e faz previsão recursiva."""
    df_train = create_lag_features(train_y, lags)
    df_test = create_lag_features(pd.concat([train_y.iloc[-lags:], test_y]), lags)

    feat_cols = [c for c in df_train.columns if c != "y"]
    X_tr, y_tr = df_train[feat_cols].values, df_train["y"].values
    X_te = df_test[feat_cols].values[:len(test_y)]

    t0 = time.time()
    model.fit(X_tr, y_tr)
    t1 = time.time()
    y_pred = model.predict(X_te)

    return y_pred, y_tr, t1 - t0

def evaluate(name, y_true, y_pred, y_train=None):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    return {"nome": name, "MAE": mae, "RMSE": rmse, "MAPE": mape}

def run_comparison(data, name, test_size=12, lags=12):
    print(f"\n{'='*95}")
    print(f"{name}")
    print(f"{'='*95}")

    train = data.iloc[:-test_size].copy()
    test = data.iloc[-test_size:].copy()
    y_train, y_test = train["y"].values, test["y"].values

    results = []

    # ----- UNSUPERVISED / DECOMPOSITION -----
    print("\n>> Métodos de Decomposição (Não Supervisionados)")

    # Holt-Winters
    try:
        t0 = time.time()
        hw = ExponentialSmoothing(train["y"], seasonal_periods=12, trend="add", seasonal="add",
                                  initialization_method="estimated").fit()
        t1 = time.time()
        hw_pred = hw.forecast(test_size).values
        r = evaluate("Holt-Winters", y_test, hw_pred)
        r["tempo"] = t1 - t0
        results.append(r)
        print(f"  Holt-Winters   | MAE={r['MAE']:.2f} | RMSE={r['RMSE']:.2f} | MAPE={r['MAPE']:.2f}% | {r['tempo']:.2f}s")
    except Exception as e:
        print(f"  Holt-Winters   | ERRO: {e}")

    # SARIMA
    try:
        t0 = time.time()
        sarima = SARIMAX(train["y"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                         enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        t1 = time.time()
        sarima_pred = sarima.forecast(test_size).values
        r = evaluate("SARIMA", y_test, sarima_pred)
        r["tempo"] = t1 - t0
        results.append(r)
        print(f"  SARIMA         | MAE={r['MAE']:.2f} | RMSE={r['RMSE']:.2f} | MAPE={r['MAPE']:.2f}% | {r['tempo']:.2f}s")
    except Exception as e:
        print(f"  SARIMA         | ERRO: {e}")

    # Prophet
    try:
        t0 = time.time()
        freq = pd.infer_freq(train["ds"])
        if freq is None:
            freq = "MS" if len(train) > 100 else "YS"
        prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        prophet.fit(train.rename(columns={"y": "y"}))
        future = prophet.make_future_dataframe(periods=test_size, freq=freq)
        forecast = prophet.predict(future)
        t1 = time.time()
        prophet_pred = forecast["yhat"].iloc[-test_size:].values
        r = evaluate("Prophet", y_test, prophet_pred)
        r["tempo"] = t1 - t0
        results.append(r)
        print(f"  Prophet        | MAE={r['MAE']:.2f} | RMSE={r['RMSE']:.2f} | MAPE={r['MAPE']:.2f}% | {r['tempo']:.2f}s")
    except Exception as e:
        print(f"  Prophet        | ERRO: {e}")

    # ----- SUPERVISED (LAG-BASED) -----
    print("\n>> Métodos de Regressão com Lags (Supervisionados)")

    # XGBoost
    try:
        y_pred, y_tr, tempo = forecast_supervised(
            xgb.XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1),
            train["y"], test["y"], lags=lags)
        r = evaluate("XGBoost (lags)", y_test, y_pred)
        r["tempo"] = tempo
        results.append(r)
        print(f"  XGBoost(lags)  | MAE={r['MAE']:.2f} | RMSE={r['RMSE']:.2f} | MAPE={r['MAPE']:.2f}% | {r['tempo']:.2f}s")
    except Exception as e:
        print(f"  XGBoost(lags)  | ERRO: {e}")

    # Random Forest
    try:
        y_pred, y_tr, tempo = forecast_supervised(
            RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
            train["y"], test["y"], lags=lags)
        r = evaluate("RF (lags)", y_test, y_pred)
        r["tempo"] = tempo
        results.append(r)
        print(f"  RF(lags)       | MAE={r['MAE']:.2f} | RMSE={r['RMSE']:.2f} | MAPE={r['MAPE']:.2f}% | {r['tempo']:.2f}s")
    except Exception as e:
        print(f"  RF(lags)       | ERRO: {e}")

    # Tabela final
    print(f"\n{'='*95}")
    print(f"RESUMO - {name}")
    print(f"{'='*95}")
    print(f"{'Método':<20} {'Tipo':<20} {'MAE':<10} {'RMSE':<10} {'MAPE(%)':<10} {'Tempo(s)':<10}")
    print("-" * 80)

    best_unsup = min([r for r in results if r["nome"] in ["Holt-Winters", "SARIMA", "Prophet"]],
                     key=lambda x: x["RMSE"], default=None)
    best_sup = min([r for r in results if r["nome"] in ["XGBoost (lags)", "RF (lags)"]],
                   key=lambda x: x["RMSE"], default=None)

    tipo_map = {"Holt-Winters": "Decomposição", "SARIMA": "Decomposição", "Prophet": "Decomposição",
                "XGBoost (lags)": "Regressão", "RF (lags)": "Regressão"}

    for r in sorted(results, key=lambda x: x["RMSE"]):
        tipo = tipo_map.get(r["nome"], "?")
        print(f"  {r['nome']:<18} {tipo:<20} {r['MAE']:<10.2f} {r['RMSE']:<10.2f} {r['MAPE']:<10.2f} {r['tempo']:<10.2f}")

    best = min(results, key=lambda x: x["RMSE"])
    if best_unsup and best_sup:
        print(f"\n  Melhor decomposição: {best_unsup['nome']} (RMSE={best_unsup['RMSE']:.2f})")
        print(f"  Melhor regressão:    {best_sup['nome']} (RMSE={best_sup['RMSE']:.2f})")
    print(f"  Melhor geral:         {best['nome']} (RMSE={best['RMSE']:.2f}, MAPE={best['MAPE']:.2f}%)")

    return results

# ============================================================
# 3. Executar nos dois datasets
# ============================================================
r_air = run_comparison(air, "Air Passengers", test_size=24, lags=12)
r_sun = run_comparison(sun, "Sunspots", test_size=60, lags=12)

# ============================================================
# 4. Conclusão
# ============================================================
print("\n\n" + "=" * 95)
print("CONCLUSÃO SOBRE FORECASTING")
print("=" * 95)
print("""
Metodos de decomposicao (Holt-Winters, SARIMA, Prophet) sao considerados
"nao supervisionados" porque modelam a estrutura intrinseca da serie
(tendencia + sazonalidade) sem precisar de features externas.

Metodos de regressao (XGBoost, RF) sao "supervisionados" porque tratam
o forecasting como um problema de aprendizado com features (lags) e
target (proximo valor).

Na pratica:
- Se a serie tem sazonalidade e tendencia estaveis: decomposicao ganha
- Se ha multiplas series, features externas, ou padroes complexos: regressao ganha
- Prophet e um meio-termo: usa decomposicao mas permite incorporar
  regressores externos (semi-supervisionado)
""")
print("=" * 95)
