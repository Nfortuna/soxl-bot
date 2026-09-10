import yfinance as yf
import pandas as pd
import os, requests, pytz, time
from datetime import datetime

# Limpieza preventiva de variables de entorno
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

tickers = [
    "NVDA","AVGO","MU","AMD","AMAT","MRVL","INTC","KLAC","MPWR","TER","ADI","NXPI",
    "TXN","MCHP","SWKS","QRVO","ON","ENPH","FSLR","TSM","ASX","STM","UMC","LSCC",
    "OLED","RMBS","WOLF","SYNA","POWI","CRUS"
]

ny_tz = pytz.timezone("America/New_York")

def enviar_alerta(mensaje):
    # Verificación e impresión preventiva en la consola de GitHub
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == "":
        print("[ALERTA CRÍTICA] El secreto 'TELEGRAM_TOKEN' está vacío o no está configurado en GitHub Secrets.")
        return
    if not CHAT_ID or CHAT_ID == "":
        print("[ALERTA CRÍTICA] El secreto 'TELEGRAM_CHAT_ID' está vacío o no está configurado en GitHub Secrets.")
        return
        
    try:
        url = f"https://telegram.org{TELEGRAM_TOKEN}/sendMessage"
        resp = requests.post(url, data={
            "chat_id": CHAT_ID,
            "text": mensaje,
            "parse_mode": "Markdown"
        }, timeout=10)
        
        if resp.status_code != 200:
            print(f"[ERROR] API de Telegram rechazó el mensaje: {resp.text}")
    except Exception as e:
        print(f"[ERROR] Error de conexión con Telegram: {e}")

def calcular_pesos_reales_indice():
    market_caps = {}
    print("[INFO] Sincronizando pesos reales basados en Capitalización de Mercado...")
    
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            cap = info.get("marketCap", 0)
            if cap > 0:
                market_caps[t] = cap
            else:
                market_caps[t] = 10_000_000_000
        except Exception:
            market_caps[t] = 10_000_000_000
            
    ordenados = sorted(market_caps.items(), key=lambda item: item[1], reverse=True)
    
    pesos_calculados = {}
    suma_inicial_top5 = sum([val for idx, (tk, val) in enumerate(ordenados) if idx < 5])
    suma_inicial_resto = sum([val for idx, (tk, val) in enumerate(ordenados) if idx >= 5])
    
    for i, (ticker, cap_value) in enumerate(ordenados):
        if i < 5:
            peso_teorico = (cap_value / suma_inicial_top5) * 0.40
            pesos_calculados[ticker] = min(peso_teorico, 0.08)
        else:
            peso_teorico = (cap_value / suma_inicial_resto) * 0.60
            pesos_calculados[ticker] = min(peso_teorico, 0.04)
            
    total_pesos = sum(pesos_calculados.values())
    for ticker in pesos_calculados:
        pesos_calculados[ticker] /= total_pesos
        
    return pesos_calculados

def calcular_manual():
    ahora = datetime.now(ny_tz)
    hora_actual = ahora.hour * 60 + ahora.minute
    apertura = 9*60 + 30
    cierre = 16*60
    
    if hora_actual < apertura or hora_actual >= cierre:
        print("[INFO] Fuera de horario de mercado, no se envía alerta.")
        return

    pesos_reales = calcular_pesos_reales_indice()

    try:
        soxl_data = yf.download("SOXL", period="1d", interval="1m", group_by="ticker", progress=False)
        time.sleep(1)
        datos = yf.download(tickers, period="1d", interval="1m", group_by="ticker", progress=False)
        time.sleep(1)
        diarios = yf.download(["SOXL"] + tickers, period="7d", interval="1d", group_by="ticker", progress=False)
    except Exception as e:
        enviar_alerta(f"❌ Error al descargar datos en script principal: {e}")
        return

    df_soxl = soxl_data["SOXL"] if isinstance(soxl_data.columns, pd.MultiIndex) else soxl_data
    if df_soxl.empty:
        enviar_alerta("❌ Datos de SOXL vacíos.")
        return

    # CORRECCIÓN DE INDEXACIÓN CON .iloc CORRECTO
    p_real_val = df_soxl["Close"].iloc[-1]
    precio_real = float(p_real_val.iloc[0] if isinstance(p_real_val, pd.Series) else p_real_val)
    if pd.isna(precio_real) or precio_real == 0:
        enviar_alerta("❌ Precio real inválido.")
        return

    p_open_val = df_soxl["Open"].iloc[0]
    precio_open_soxl = float(p_open_val.iloc[0] if isinstance(p_open_val, pd.Series) else p_open_val)

    df_soxl_diario = diarios["SOXL"] if isinstance(diarios.columns, pd.MultiIndex) else diarios
    df_soxl_diario = df_soxl_diario.dropna(subset=["Close"])
    df_prev = df_soxl_diario[df_soxl_diario.index.date < ahora.date()]
    if df_prev.empty:
        enviar_alerta("❌ No hay cierre anterior válido para SOXL.")
        return
        
    p_cierre_val = df_prev["Close"].iloc[-1]
    cierre_prev_soxl = float(p_cierre_val.iloc[0] if isinstance(p_cierre_val, pd.Series) else p_cierre_val)
    
    alto_prev_val = df_prev["High"].iloc[-1]
    alto_prev_soxl = float(alto_prev_val.iloc[0] if isinstance(alto_prev_val, pd.Series) else alto_prev_val)
    
    bajo_prev_val = df_prev["Low"].iloc[-1]
    bajo_prev_soxl = float(bajo_prev_val.iloc[0] if isinstance(bajo_prev_val, pd.Series) else bajo_prev_val)

    pivot = (alto_prev_soxl + bajo_prev_soxl + cierre_prev_soxl) / 3
    r1 = (2 * pivot) - bajo_prev_soxl
    s1 = (2 * pivot) - alto_prev_soxl
    r2 = pivot + (alto_prev_soxl - bajo_prev_soxl)
    s2 = pivot - (alto_prev_soxl - bajo_prev_soxl)

    variaciones, suma_pesos = [], 0.0
    componentes_msg = ""
    
    for t in tickers:
        try:
            df_intradia = datos[t] if isinstance(datos.columns, pd.MultiIndex) else datos
            df_diario = diarios[t] if isinstance(diarios.columns, pd.MultiIndex) else diarios
            if df_intradia.empty or df_diario.empty: continue
            df_diario = df_diario.dropna(subset=["Close"])
            df_prev_t = df_diario[df_diario.index.date < ahora.date()]
            if df_prev_t.empty: continue
            
            p_momento = df_intradia["Close"].iloc[-1]
            c_prev = df_prev_t["Close"].iloc[-1]
            precio_momento = float(p_momento.iloc[0] if isinstance(p_momento, pd.Series) else p_momento)
            cierre_prev = float(c_prev.iloc[0] if isinstance(c_prev, pd.Series) else c_prev)
            
            if pd.isna(precio_momento) or pd.isna(cierre_prev) or cierre_prev == 0: continue
            
            var_pct = ((precio_momento - cierre_prev) / cierre_prev) * 100
            variaciones.append(var_pct * pesos_reales[t])
            suma_pesos += pesos_reales[t]
            
            if t in ["NVDA", "MU", "AMD", "AVGO"]:
                signo = "+" if var_pct >= 0 else ""
                componentes_msg += f"   • {t} ({pesos_reales[t]*100:.1f}%): ${precio_momento:.2f} ({signo}{var_pct:.2f}%)\n"
        except Exception:
            continue

    if suma_pesos == 0:
        enviar_alerta("❌ No se pudo calcular variaciones en script principal.")
        return

    var_total = sum(variaciones) / suma_pesos

    precio_estimado_close = cierre_prev_soxl * (1 + (var_total * 3)/100)
    desviacion_close = ((precio_estimado_close - precio_real) / precio_real) * 100

    precio_estimated_open = precio_open_soxl * (1 + (var_total * 3)/100)
    desviacion_open = ((precio_estimated_open - precio_real) / precio_real) * 100

    signo_c = "+" if desviacion_close >= 0 else ""
    signo_o = "+" if desviacion_open >= 0 else ""
    
    mensaje = (
        f"🚀 *SOXL Bot Principal*\n\n"
        f"💵 *Precio Real SOXL:* {precio_real:.2f}\n\n"
        f"📈 *Basado en Cierre Anterior*\n"
        f"   🔹 Estimado: {precio_estimado_close:.2f}\n"
        f"   🔹 Desviación: {signo_c}{desviacion_close:.2f}%\n\n"
        f"📉 *Basado en Apertura del Día*\n"
        f"   🔹 Estimado: {precio_estimated_open:.2f}\n"
        f"   🔹 Desviación: {signo_o}{desviacion_open:.2f}%\n\n"
        f"🧱 *Niveles Técnicos Clave:*\n"
        f"   🔺 R2: {r2:.2f} | R1: {r1:.2f}\n"
        f"   ⚪ Pivot: {pivot:.2f}\n"
        f"   🔻 S1: {s1:.2f} | S2: {s2:.2f}\n\n"
        f"🔍 *Desglose de Componentes Top:*\n"
        f"{componentes_msg}"
    )
    
    enviar_alerta(mensaje)

if __name__ == "__main__":
    calcular_manual()
