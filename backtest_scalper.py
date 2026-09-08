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
            print("🚀 Mensaje enviado con éxito a Telegram.")
        except Exception as e:
            print(f"⚠️ No se pudo enviar reporte a Telegram: {e}")

def auditar_scalper():
    print("⚡ Iniciando Backtest Walk-Forward para el Motor Intradía Pro-Scalper...")
    csv_filename = "scalper_predictions.csv"
    COSTO_FRICCION = 0.03 
    
    if not os.path.exists(csv_filename):
        msg = "⚡ *AUDITORÍA WALK-FORWARD: SCALPER*\n\n⚠️ No se encontró la caja negra 'scalper_predictions.csv' en la nube."
        print(msg)
        enviar_telegram(msg)
        return
        
    try:
        df = pd.read_csv(csv_filename, index_col=0)
        
        col_actual = [col for col in df.columns if 'actual' in str(col).lower()]
        col_cambio = [col for col in df.columns if 'cambio' in str(col).lower()]
        
        if not col_actual or not col_cambio:
            msg = "⚡ *AUDITORÍA WALK-FORWARD: SCALPER*\n\n⏳ Estructurando base de datos. Esperando que el robot complete sus próximas ejecuciones regulares."
            print(msg)
            enviar_telegram(msg)
            return
            
        c_actual = col_actual[0]
        c_cambio = col_cambio[0]
        
        df = df[df[c_actual] > 0]
        
        if len(df) < 5:
            msg = f"⚡ *AUDITORÍA WALK-FORWARD: SCALPER*\n\n⏳ Muestras insuficientes ({len(df)}/5). Acumulando historial de ráfagas en vivo..."
            print(msg)
            enviar_telegram(msg)
            return
            
        df['Precio_Real_Futuro'] = df[c_actual].shift(-1)
        df['Cambio_Real_Neto'] = df['Precio_Real_Futuro'] - df[c_actual]
        df.dropna(subset=['Cambio_Real_Neto'], inplace=True)
        
        df['Dir_Predicha'] = np.where(df[c_cambio] > 0, 1, -1)
        df['Dir_Real'] = np.where(df['Cambio_Real_Neto'] > 0, 1, -1)
        
        df['Acierto_Bruto'] = df['Dir_Predicha'] == df['Dir_Real']
        df['Acierto_Neto'] = (df['Acierto_Bruto']) & (df['Cambio_Real_Neto'].abs() > COSTO_FRICCION)
        
        win_rate_bruto = df['Acierto_Bruto'].mean() * 100
        win_rate_neto = df['Acierto_Neto'].mean() * 100
        comidas_spread = df['Acierto_Bruto'].sum() - df['Acierto_Neto'].sum()
        
        reporte = (
            f"⚡ *AUDITORÍA WALK-FORWARD: SCALPER*\n\n"
            f"🔹 Ráfagas Auditadas: {len(df)}\n"
            f"📐 Win Rate Bruto (Teórico): {win_rate_bruto:.2f}%\n"
            f"🎯 *Win Rate Neto (Con Spread):* {win_rate_neto:.2f}%\n"
            f"💸 Perdidas por Fricción ($0.03): {comidas_spread}\n"
        )
        print(reporte)
        enviar_telegram(reporte)
        
    except Exception as e:
        print(f"❌ Error en la auditoría del scalper: {e}")
        enviar_telegram(f"⚠️ *Fallo en Auditoría Scalper*\nError: {e}")

if __name__ == "__main__":
    auditar_scalper()
