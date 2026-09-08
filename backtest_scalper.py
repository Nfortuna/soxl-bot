import os
import pandas as pd
import numpy as np

def auditar_scalper():
    print("⚡ Iniciando Backtest Walk-Forward para el Motor Intradía Pro-Scalper...")
    csv_filename = "scalper_predictions.csv"
    msg_filename = "telegram_scalper_audit.txt"
    COSTO_FRICCION = 0.03
    
    with open(msg_filename, "w", encoding="utf-8") as f:
        f.write("⚡ *Auditoría Walk-Forward: Scalper*\n\n⏳ Estructurando base de datos. Esperando ejecuciones del mercado vivo.")
        
    if not os.path.exists(csv_filename):
        return
        
    try:
        df = pd.read_csv(csv_filename, index_col=0)
        col_actual = [col for col in df.columns if 'actual' in str(col).lower()]
        col_cambio = [col for col in df.columns if 'cambio' in str(col).lower()]
        
        if not col_actual or not col_cambio:
            return
            
        c_actual, c_cambio = col_actual[0], col_cambio[0]
        df = df[df[c_actual] > 0]
        
        if len(df) < 5:
            with open(msg_filename, "w", encoding="utf-8") as f:
                f.write(f"⚡ *AUDITORÍA WALK-FORWARD: SCALPER*\n\n⏳ Muestras insuficientes ({len(df)}/5). Acumulando historial de ráfagas en vivo...")
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
        
        with open(msg_filename, "w", encoding="utf-8") as f:
            f.write(
                f"⚡ *AUDITORÍA WALK-FORWARD: SCALPER*\n\n"
                f"🔹 Ráfagas Auditadas: {len(df)}\n"
                f"📐 Win Rate Bruto (Teórico): {win_rate_bruto:.2f}%\n"
                f"🎯 *Win Rate Neto (Con Spread):* {win_rate_neto:.2f}%\n"
                f"💸 Perdidas por Fricción ($0.03): {comidas_spread}\n"
            )
        print("✅ Reporte scalper guardado.")
    except Exception as e:
        print(f"Nota en la auditoría del scalper: {e}")

if __name__ == "__main__":
    auditar_scalper()
