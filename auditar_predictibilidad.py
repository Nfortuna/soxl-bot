import os
import pandas as pd
import numpy as np
import datetime

def calcular_win_rate():
    print("📊 Iniciando consolidación semanal de predictibilidad...")
    csv_filename = "scalper_predictions.csv"
    msg_filename = "telegram_audit_msg.txt"
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    
    if not os.path.exists(csv_filename):
        with open(msg_filename, "w", encoding="utf-8") as f:
            f.write("📊 *Auditoría Semanal Pro-Scalper*\n\n⚠️ Aún no se han registrado operaciones en el historial para calcular métricas.")
        return
        
    try:
        df = pd.read_csv(csv_filename, index_col=0)
        
        # Eliminar posibles filas de contingencia limpias en ceros para no alterar la estadística real
        df = df[(df['Actual'] > 0) & (df['Proyectado_10m'] > 0)]
        
        if len(df) < 5:
            with open(msg_filename, "w", encoding="utf-8") as f:
                f.write(f"📊 *Auditoría Semanal Pro-Scalper*\n\n⏳ Muestras insuficientes ({len(df)}/5). Se requiere acumular más ráfagas de mercado regular el lunes.")
            return
            
        # Determinar dirección predicha por la IA (1 = Alza, -1 = Baja)
        df['Direccion_Predicha'] = np.where(df['Proyectado_10m'] > df['Actual'], 1, -1)
        
        # Enfoque walk-forward retrospectivo: el precio actual de la siguiente corrida es el desenlace de la anterior
        df['Precio_Real_10m_Despues'] = df['Actual'].shift(-1)
        df.dropna(subset=['Precio_Real_10m_Despues'], inplace=True)
        
        if df.empty:
            with open(msg_filename, "w", encoding="utf-8") as f:
                f.write("📊 *Auditoría Semanal Pro-Scalper*\n\n⏳ Procesando cierres temporales en la nube...")
            return
            
        df['Direccion_Real'] = np.where(df['Precio_Real_10m_Despues'] > df['Actual'], 1, -1)
        
        # Cálculo del Win Rate de la Inteligencia Artificial
        aciertos = (df['Direccion_Predicha'] == df['Direccion_Real']).sum()
        total_evaluado = len(df)
        win_rate = (aciertos / total_evaluado) * 100
        
        # Clasificar la calidad del modelo según estándares institucionales
        if win_rate >= 60:
            status_ia = "Excelente Rentabilidad (Alfa Alta) 🟢"
        elif win_rate >= 50:
            status_ia = "Moderado / Rentable con Gestión de Riesgo 🟡"
        else:
            status_ia = "Optimizar Coeficientes (Ruido de Corto Plazo) 🔴"
            
        # Redactar el reporte ejecutivo automatizado para Telegram
        with open(msg_filename, "w", encoding="utf-8") as f:
            f.write(
                f"📊 *AUDITORÍA SEMANAL DE PREDICTIBILIDAD*\n"
                f"📅 Fecha de Cierre: {today_str}\n\n"
                f"🎯 *Win Rate Global:* {win_rate:.2f}%\n"
                f"🔹 Desempeño: {status_ia}\n\n"
                f"🔢 *Métricas Consolidadas:*\n"
                f"▪️ Ráfagas Auditadas: {total_evaluado}\n"
                f"▪️ Predicciones Correctas: {aciertos}\n"
                f"📈 Volatilidad Promedio (VIX): {df['Nivel_VIX'].mean():.2f}\n"
                f"🖥️ Puntos Promedio Nasdaq: {int(df['Nivel_Nasdaq'].mean())} pts\n\n"
                f"💾 Reporte estadístico archivado en la nube."
            )
        print("✅ Reporte escrito con éxito en telegram_audit_msg.txt")
        
    except Exception as e:
        print(f"❌ Error al procesar la auditoría: {e}")
        with open(msg_filename, "w", encoding="utf-8") as f:
            f.write(f"⚠️ *Fallo en el Reporte de Auditoría*\nDetalle técnico: {e}")

if __name__ == "__main__":
    calcular_win_rate()
