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
            print(f"⚠️ No se pudo enviar reporte diario a Telegram: {e}")

def auditar_modelo_diario():
    print("📊 Iniciando Backtest Walk-Forward para el Modelo Macro Diario...")
    csv_filename = "soxl_predictions.csv"
    
    if not os.path.exists(csv_filename):
        print("⚠️ No se encontró el archivo 'soxl_predictions.csv'.")
        return
        
    try:
        df = pd.read_csv(csv_filename, index_col=0)
        
        # NORMALIZACIÓN CRÍTICA: Forzar mayúsculas en las columnas para evitar el error 'High'
        df.columns = [str(col).capitalize() for col in df.columns]
        
        # Eliminar filas vacías o de contingencia iniciales en cero
        df = df[(df['High'] > 0) & (df['Real'] > 0)]
        
        if len(df) < 3:
            print("⏳ Muestras insuficientes en soxl_predictions.csv.")
            return
            
        df['Fecha_Dia'] = df.index.str.split('_').str[0]
        df['Momento'] = df.index.str.split('_').str[1]
        
        cierres_reales = df[df['Momento'] == 'CIERRE'][['Fecha_Dia', 'Real']].rename(columns={'Real': 'Cierre_Real_Definitivo'})
        df = df.merge(cierres_reales, on='Fecha_Dia', how='left')
        df.dropna(subset=['Cierre_Real_Definitivo'], inplace=True)
        
        if df.empty:
            print("⏳ Esperando cierres completos para auditar...")
            return
            
        df['Acierto_Direccional'] = np.sign(df['Close'] - df['Real']) == np.sign(df['Cierre_Real_Definitivo'] - df['Real'])
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
