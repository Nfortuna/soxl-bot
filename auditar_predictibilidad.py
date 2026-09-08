import os
import pandas as pd
import numpy as np
import datetime

def calcular_win_rate_real():
    print("📊 Iniciando auditoría cuantitativa con descuento de fricción...")
    csv_filename = "scalper_predictions.csv"
    msg_filename = "telegram_audit_msg.txt"
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    
    # COSTO DE FRICCIÓN ASUMIDO: Spread Bid-Ask + Comisiones estimadas en dólares ($0.03 por acción)
    COSTO_FRICCION = 0.03
    
    if not os.path.exists(csv_filename):
        with open(msg_filename, "w", encoding="utf-8") as f:
            f.write("📊 *Auditoría Cuantitativa Semanal*\n\n⚠️ No se encontró la caja negra 'scalper_predictions.csv'.")
        return
        
    try:
        df = pd.read_csv(csv_filename, index_col=0)
        
        # Filtrar registros reales de producción (eliminar ceros de contingencia)
        df = df[df['Actual'] > 0]
        
        if len(df) < 2:
            with open(msg_filename, "w", encoding="utf-8") as f:
                f.write("📊 *Auditoría Cuantitativa Semanal*\n\n⏳ Muestras insuficientes en el historial para realizar la auditoría temporal.")
            return
            
        # --- LOGICA 10 MINUTOS REALES (LOOK-FORWARD HISTÓRICO) ---
        # Como el scalper guarda registros secuenciales en el CSV, el desenlace real de la predicción 
        # hecha en la fila T es el precio 'Actual' que se registra en la siguiente fila T+1 (30 minutos después)
        df['Precio_Desenlace_Real'] = df['Actual'].shift(-1)
        df['Cambio_Real_Dolares'] = df['Precio_Desenlace_Real'] - df['Actual']
        
        # Eliminamos la última fila activa porque aún está esperando su desenlace en el mercado vivo
        df_evaluado = df.dropna(subset=['Cambio_Real_Dolares']).copy()
        
        if df_evaluado.empty:
            with open(msg_filename, "w", encoding="utf-8") as f:
                f.write("📊 *Auditoría Cuantitativa Semanal*\n\n⏳ Esperando acumulación del siguiente bloque temporal...")
            return
            
        # --- CLASIFICACIÓN DIRECCIONAL CON FILTRO DE TRANSPARENCIA ---
        # Dirección predicha por la IA (1 = Alza, -1 = Baja)
        df_evaluado['Dir_Predicha'] = np.where(df_evaluado['Proyectado_Cambio_10m'] > 0, 1, -1)
        
        # Dirección real del mercado (1 = Alza, -1 = Baja)
        df_evaluado['Dir_Real'] = np.where(df_evaluado['Cambio_Real_Dolares'] > 0, 1, -1)
        
        # Acierto Direccional Bruto (Teórico sin costos)
        df_evaluado['Acierto_Bruto'] = df_evaluado['Dir_Predicha'] == df_evaluado['Dir_Real']
        win_rate_bruto = df_evaluado['Acierto_Bruto'].mean() * 100
        
        # --- ACIERTO NETO UTIL (DESCONTANDO EL SPREAD / FRICCIÓN DE COMPRA-VENTA) ---
        # Solo es un acierto útil si la dirección fue correcta Y la magnitud del movimiento 
        # superó el costo del spread bid-ask. De lo contrario, la fricción se come la ganancia.
        df_evaluado['Acierto_Neto_Util'] = (df_evaluado['Acierto_Bruto']) & (df_evaluado['Cambio_Real_Dolares'].abs() > COSTO_FRICCION)
        
        total_operaciones = len(df_evaluado)
        aciertos_brutos = df_evaluado['Acierto_Bruto'].sum()
        aciertos_netos_utiles = df_evaluado['Acierto_Neto_Util'].sum()
        
        win_rate_neto = (aciertos_netos_utiles / total_operaciones) * 100
        operaciones_comidas_por_spread = aciertos_brutos - aciertos_netos_utiles
        
        # Clasificación institucional de robustez
        if win_rate_neto >= 55:
            calidad_modelo = "Rentable y Seguro (Supera la Fricción) 🟢"
        elif win_rate_neto >= 48:
            calidad_modelo = "Punto de Equilibrio / Ajustar Gestión de Riesgo 🟡"
        else:
            calidad_modelo = "Alerta: El Spread consume la ventaja predictiva 🔴"
            
        # Escribir reporte para Telegram
        with open(msg_filename, "w", encoding="utf-8") as f:
            f.write(
                f"📊 *AUDITORÍA CUANTITATIVA SEMANAL*\n"
                f"📅 Fecha de Cierre: {today_str}\n\n"
                f"🎯 *Win Rate Neto Real:* {win_rate_neto:.2f}%\n"
                f"🔹 Calidad de Señal: {calidad_modelo}\n\n"
                f"📊 *Métricas de Fricción Financiera:*\n"
                f"▪️ Ráfagas Totales Auditadas: {total_operaciones}\n"
                f"▪️ Aciertos Brutos (Teóricos): {aciertos_brutos}\n"
                f"▪️ Aciertos Netos Útiles: {aciertos_netos_utiles}\n"
                f"💸 Comidas por Spread (${COSTO_FRICCION}): {operaciones_comidas_por_spread}\n"
                f"📐 Win Rate Bruto (Sin Spread): {win_rate_bruto:.2f}%\n\n"
                f"📈 *Fatores de Entorno Evaluados:*\n"
                f"▪️ Volatilidad Promedio (VIX): {df_evaluado['Nivel_VIX'].mean():.2f}\n"
                f"▪️ Nasdaq Promedio: {int(df_evaluado['Nivel_Nasdaq'].mean())} pts\n\n"
                f"💾 Análisis de predictibilidad archivado en la nube."
            )
        print("✅ Reporte de auditoría de fricción generado exitosamente.")
        
    except Exception as e:
        print(f"❌ Error en el cálculo de auditoría: {e}")
        with open(msg_filename, "w", encoding="utf-8") as f:
            f.write(f"⚠️ *Fallo en el Reporte de Auditoría Cuantitativa*\nDetalle técnico: {e}")

if __name__ == "__main__":
    calcular_win_rate_real()
