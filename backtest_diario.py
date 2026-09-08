import os
import pandas as pd
import numpy as np
import requests

def enviar_telegram(mensaje):
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        url = f"https://telegram.org{token}/sendMessage"
        try:
            requests.post(url, json={"chat_id": chat_id, "text": mensaje, "parse_mode": "Markdown"})
        except Exception as e:
            print(f"⚠️ No se pudo enviar reporte a Telegram: {e}")

def auditar_modelo_diario():
    print("📊 Iniciando Backtest Walk-Forward para el Modelo Macro Diario...")
    csv_filename = "soxl_predictions.csv"
    
    if not os.path.exists(csv_filename):
        print("⚠️ No se encontró el archivo 'soxl_predictions.csv'.")
        return
        
    try:
        df = pd.read_csv(csv_filename, index_col=0)
        
        # Mapeo dinámico para encontrar la columna High sin importar cómo se guardó antes
        col_high = [c for col in df.columns if 'high' in str(col).lower()]
        col_close = [c for col in df.columns if 'close' in str(col).lower()]
        col_real = [c for col in df.columns if 'real' in str(col).lower()]
        
        if not col_high or not col_close or not col_real:
            print(f"⏳ Columnas detectadas: {list(df.columns)}. Esperando acumulación de nuevas filas.")
            return
            
        c_high, c_close, c_real = col_high[0], col_close[0], col_real[0]
        
        # Filtrar solo filas con datos válidos mayores a cero
        df = df[(df[c_high] > 0) & (df[c_real] > 0)]
        
        if len(df) < 2:
            print("⏳ Muestras insuficientes en soxl_predictions.csv.")
            return
            
        df['Fecha_Dia'] = df.index.str.split('_').str[0]
        df['Momento'] = df.index.str.split('_').str[-1]
        
        cierres_reales = df[df['Momento'] == 'CIERRE'][['Fecha_Dia', c_real]].rename(columns={c_real: 'Cierre_Real_Definitivo'})
        df = df.merge(cierres_reales, on='Fecha_Dia', how='left')
        df.dropna(subset=['Cierre_Real_Definitivo'], inplace=True)
        
        if df.empty:
            print("⏳ Esperando cierres completos para auditar...")
            return
            
        df['Acierto_Direccional'] = np.sign(df[c_close] - df[c_real]) == np.sign(df['Cierre_Real_Definitivo'] - df[c_real])
        win_rate_macro = df['Acierto_Direccional'].mean() * 100
        
        reporte = (
            f"📈 *AUDITORÍA WALK-FORWARD: BOT DIARIO*\n\n"
            f"🔹 Días Operativos Evaluados: {df['Fecha_Dia'].nunique()}\n"
            f"🎯 *Win Rate Direccional:* {win_rate_macro:.2f}%\n"
        )
        print(reporte)
        enviar_telegram(reporte)
        
    except Exception as e:
        print(f"❌ Error en la auditoría diaria: {e}")

if __name__ == "__main__":
    auditar_modelo_diario()
