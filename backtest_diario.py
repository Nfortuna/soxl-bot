import os
import pandas as pd
import numpy as np

def auditar_modelo_diario():
    print("📊 Iniciando Backtest Walk-Forward para el Modelo Macro Diario...")
    csv_filename = "soxl_predictions.csv"
    msg_filename = "telegram_diario_audit.txt"
    
    # Escribir un archivo de texto por defecto por si está vacío
    with open(msg_filename, "w", encoding="utf-8") as f:
        f.write("📈 *Auditoría Walk-Forward: Bot Diario*\n\n⏳ Esperando acumulación de nuevas filas del mercado regular.")
        
    if not os.path.exists(csv_filename):
        return
        
    try:
        df = pd.read_csv(csv_filename, index_col=0)
        df.columns = [str(col).capitalize() for col in df.columns]
        
        col_high = [c for col in df.columns if 'high' in str(col).lower()]
        col_close = [c for col in df.columns if 'close' in str(col).lower()]
        col_real = [c for col in df.columns if 'real' in str(col).lower()]
        
        if not col_high or not col_close or not col_real:
            return
            
        c_high, c_close, c_real = col_high[0], col_close[0], col_real[0]
        df = df[(df[c_high] > 0) & (df[c_real] > 0)]
        
        if len(df) < 2:
            return
            
        df['Fecha_Dia'] = df.index.str.split('_').str[0]
        df['Momento'] = df.index.str.split('_').str[-1]
        
        cierres_reales = df[df['Momento'] == 'CIERRE'][['Fecha_Dia', c_real]].rename(columns={c_real: 'Cierre_Real_Definitivo'})
        df = df.merge(cierres_reales, on='Fecha_Dia', how='left')
        df.dropna(subset=['Cierre_Real_Definitivo'], inplace=True)
        
        if df.empty:
            return
            
        df['Acierto_Direccional'] = np.sign(df[c_close] - df[c_real]) == np.sign(df['Cierre_Real_Definitivo'] - df[c_real])
        win_rate_macro = df['Acierto_Direccional'].mean() * 100
        
        with open(msg_filename, "w", encoding="utf-8") as f:
            f.write(
                f"📈 *AUDITORÍA WALK-FORWARD: BOT DIARIO*\n\n"
                f"🔹 Días Operativos Evaluados: {df['Fecha_Dia'].nunique()}\n"
                f"🎯 *Win Rate Direccional:* {win_rate_macro:.2f}%\n"
            )
        print("✅ Reporte macro guardado.")
    except Exception as e:
        print(f"Nota en la auditoría diaria: {e}")

if __name__ == "__main__":
    auditar_modelo_diario()
