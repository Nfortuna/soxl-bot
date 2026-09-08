import os
import pandas as pd
import numpy as np

def auditar_modelo_diario():
    print("📊 Iniciando Backtest Walk-Forward para el Modelo Macro Diario...")
    csv_filename = "soxl_predictions.csv"
    
    if not os.path.exists(csv_filename):
        print("⚠️ No se encontró el archivo de datos 'soxl_predictions.csv'.")
        return
        
    try:
        df = pd.read_csv(csv_filename, index_col=0)
        # Limpiar filas vacías o de contingencia iniciales
        df = df[(df['High'] > 0) & (df['Real'] > 0)]
        
        if len(df) < 3:
            print(f"⏳ Muestras insuficientes ({len(df)}). Se requiere acumular más días.")
            return
            
        # El desenlace real del día es guardado por la ejecución vespertina (CIERRE)
        # Extraemos el valor real final combinando las filas del historial
        df['Fecha_Dia'] = df.index.str.split('_').str[0]
        df['Momento'] = df.index.str.split('_').str[1]
        
        # Agrupar para aislar los cierres reales definitivos por día
        cierres_reales = df[df['Momento'] == 'CIERRE'][['Fecha_Dia', 'Real']].rename(columns={'Real': 'Cierre_Real_Definitivo'})
        
        df = df.merge(cierres_reales, on='Fecha_Dia', how='left')
        df.dropna(subset=['Cierre_Real_Definitivo'], inplace=True)
        
        if df.empty:
            print("⏳ Esperando consolidación de cierres diarios completos...")
            return
            
        # Evaluar la precisión direccional del Cierre
        df['Acierto_Direccional'] = np.sign(df['Close'] - df['Real']) == np.sign(df['Cierre_Real_Definitivo'] - df['Real'])
        win_rate_macro = df['Acierto_Direccional'].mean() * 100
        
        print("\n" + "="*50)
        print("📈 RESULTADOS WALK-FORWARD: BOT DIARIO MAESTRO")
        print("="*50)
        print(f"🔹 Días Operativos Evaluados: {df['Fecha_Dia'].nunique()}")
        print(f"🎯 Win Rate Direccional del Cierre: {win_rate_macro:.2f}%")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"❌ Error en la auditoría diaria: {e}")

if __name__ == "__main__":
    auditar_modelo_diario()
