import os
import pandas as pd
import numpy as np

def auditar_scalper():
    print("⚡ Iniciando Backtest Walk-Forward para el Motor Intradía Pro-Scalper...")
    csv_filename = "scalper_predictions.csv"
    COSTO_FRICCION = 0.03 # $0.03 spread friction penalty
    
    if not os.path.exists(csv_filename):
        print("⚠️ No se encontró el archivo de datos 'scalper_predictions.csv'.")
        return
        
    try:
        df = pd.read_csv(csv_filename, index_col=0)
        df = df[df['Actual'] > 0]
        
        if len(df) < 5:
            print(f"⏳ Registros insuficientes ({len(df)}/5) para una muestra base.")
            return
            
        # Buscar el desenlace real en la fila cronológica posterior (30 min después)
        df['Precio_Real_Futuro'] = df['Actual'].shift(-1)
        df['Cambio_Real_Neto'] = df['Precio_Real_Futuro'] - df['Actual']
        df.dropna(subset=['Cambio_Real_Neto'], inplace=True)
        
        # Direcciones
        df['Dir_Predicha'] = np.where(df['Proyectado_Cambio_10m'] > 0, 1, -1)
        df['Dir_Real'] = np.where(df['Cambio_Real_Neto'] > 0, 1, -1)
        
        # Cálculos de Win Rate Bruto vs Neto Real
        df['Acierto_Bruto'] = df['Dir_Predicha'] == df['Dir_Real']
        df['Acierto_Neto'] = (df['Acierto_Bruto']) & (df['Cambio_Real_Neto'].abs() > COSTO_FRICCION)
        
        win_rate_bruto = df['Acierto_Bruto'].mean() * 100
        win_rate_neto = df['Acierto_Neto'].mean() * 100
        
        print("\n" + "="*50)
        print("⚡ RESULTADOS WALK-FORWARD: INTRADÍA PRO-SCALPER")
        print("="*50)
        print(f"🔹 Ráfagas Auditadas en Vivo: {len(df)}")
        print(f"📐 Win Rate Bruto (Teórico): {win_rate_bruto:.2f}%")
        print(f"🎯 Win Rate Neto (Con Spread de Fricción): {win_rate_neto:.2f}%")
        print(f"💸 Pérdidas Netas por Fricción de Costos: {df['Acierto_Bruto'].sum() - df['Acierto_Neto'].sum()}")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"❌ Error en la auditoría del scalper: {e}")

if __name__ == "__main__":
    auditar_scalper()
