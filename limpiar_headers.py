import os
import pandas as pd

def reparar_cajas_negras():
    print("🧹 Iniciando saneamiento de archivos log CSV...")
    
    file_diario = "soxl_predictions.csv"
    if os.path.exists(file_diario):
        try:
            df_d = pd.read_csv(file_diario)
            if any(str(col).replace('.','',1).isdigit() for col in df_d.columns) or 'Alza' in df_d.columns:
                print("⚠️ Estructura corrupta detectada en soxl_predictions.csv. Forzando formateo...")
                columnas_correctas = ["ID", "Low", "High", "Close", "Real", "Tendencia"]
                df_nuevo = pd.DataFrame(columns=columnas_correctas)
                df_nuevo.set_index("ID", inplace=True)
                df_nuevo.to_csv(file_diario)
                print("✅ soxl_predictions.csv reseteado con éxito.")
            else:
                print("👍 El archivo soxl_predictions.csv ya cuenta con una estructura limpia.")
        except Exception as e:
            print(f"Nota en archivo diario: {e}")

    file_scalper = "scalper_predictions.csv"
    if os.path.exists(file_scalper):
        try:
            df_s = pd.read_csv(file_scalper)
            if any(str(col).replace('.','',1).isdigit() for col in df_s.columns):
                print("⚠️ Estructura corrupta detectada en scalper_predictions.csv. Forzando formateo...")
                columnas_scalper = ["ID", "Actual", "Proyectado_Cambio_10m", "Tendencia", "Sesgo_VWAP", "Impulso", "Senal_Alerta", "Nivel_Nasdaq", "Nivel_VIX"]
                df_nuevo_s = pd.DataFrame(columns=columnas_scalper)
                df_nuevo_s.set_index("ID", inplace=True)
                df_nuevo_s.to_csv(file_scalper)
                print("✅ scalper_predictions.csv reseteado con éxito.")
            else:
                print("👍 El archivo scalper_predictions.csv ya cuenta con una estructura limpia.")
        except Exception as e:
            print(f"Nota en archivo scalper: {e}")

if __name__ == "__main__":
    reparar_cajas_negras()
