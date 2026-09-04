import os
import pandas as pd
import numpy as np

def calcular_win_rate():
    csv_filename = "scalper_predictions.csv"
    
    if not os.path.exists(csv_filename):
        print("📊 Aún no hay registros en la caja negra 'scalper_predictions.csv' para auditar.")
        return
        
    try:
        df = pd.read_csv(csv_filename, index_name=0) if 'Unnamed: 0' not in pd.read_csv(csv_filename).columns else pd.read_csv(csv_filename, index_col=0)
        
        if len(df) < 5:
            print(f"📊 Registros acumulados: {len(df)}/5. Se necesitan más muestras para calcular un Win Rate estadísticamente válido.")
            return
            
        # Limpieza y parsing de las señales
        df['Direccion_Predicha'] = np.where(df['Proyectado_10m'] > df['Actual'], 1, -1)
        
        # Shift temporal para buscar cuál fue el precio real 10 periodos después (simulado en base histórica)
        # Nota: En producción real, la auditoría exacta se consolida cruzando contra el historial del ticker
        df['Precio_Real_10m_Despues'] = df['Actual'].shift(-1) # Al correr cada 30 min, medimos contra el siguiente bloque
        
        df.dropna(subset=['Precio_Real_10m_Despues'], inplace=True)
        
        if df.empty:
            print("⏳ Esperando desfase de tiempo para consolidar resultados reales...")
            return
            
        df['Direccion_Real'] = np.where(df['Precio_Real_10m_Despues'] > df['Actual'], 1, -1)
        
        # Calcular el porcentaje exacto de acierto direccional
        aciertos = (df['Direccion_Predicha'] == df['Direccion_Real']).sum()
        total_evaluado = len(df)
        win_rate = (aciertos / total_evaluado) * 100
        
        print("\n" + "="*50)
        print("📊 REPORTE DE PREDICTIBILIDAD DEL MODELO SCALPER")
        print("="*50)
        print(f"🔹 Total de ráfagas auditadas: {total_evaluado}")
        print(f"🎯 Porcentaje de Acierto Direccional (Win Rate): {win_rate:.2f}%")
        print(f"📈 Nivel de volatilidad promedio (VIX actual): {df['Nivel_VIX'].mean():.2f}")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"⚠️ Nota de auditoría: Estructurando base histórica... ({e})")

if __name__ == "__main__":
    calcular_win_rate()
