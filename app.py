import os
import pandas as pd, plotly.express as px, streamlit as st
from src.pipeline import load_all
from src.rag import Retriever
from src.agent import Toolbox, CoffeeAgent
from src.csp import DAYS
from src.data import TEMP_L, TREND_L, SEASON_L

st.set_page_config(page_title="Asistente agroclimatico - Cafe", layout="wide")
st.title("☕ Asistente agroclimático para café")
st.caption("Modelo basado SOLO en temperatura de una estación IDEAM. Herramienta analítica; NO es consejo agronómico profesional.")

path = st.sidebar.text_input("CSV de la estación", os.getenv("DATA_PATH", "data/raw/ideam_station.csv"))
if not os.path.exists(path): st.error(f"No encuentro {path}. Ver README."); st.stop()


@st.cache_resource
def get_state(p): return load_all(p), Retriever("corpus")


(w, d, s, bn, train, test), retr = get_state(path)
tb = Toolbox(w, d, bn, retr); last = tb._last()
st.sidebar.header("Escenario")
temp = st.sidebar.selectbox("Temperatura semana actual", TEMP_L, TEMP_L.index(last["temp"]))
trend = st.sidebar.selectbox("Tendencia", TREND_L, TREND_L.index(last["trend"]))
season = st.sidebar.selectbox("Trimestre", SEASON_L, SEASON_L.index(last["season"]))
cap = st.sidebar.slider("Jornaleros por día", 1, 10, 5); tb.capacity = cap

t1, t2, t3, t4 = st.tabs(["Clima", "Riesgo de roya", "Calendario de labores", "Chat"])
with t1:
    wk = w.reset_index(); wk.columns = ["semana", "temp (°C)"]
    st.plotly_chart(px.line(wk, x="semana", y="temp (°C)", title="Temperatura semanal media"), use_container_width=True)
    wk["anio"] = wk["semana"].dt.year; wk["mes"] = wk["semana"].dt.month
    st.plotly_chart(px.box(wk, x="mes", y="temp (°C)", title="Estacionalidad mensual"), use_container_width=True)
with t2:
    r = tb.get_risk(temp, trend, season); c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(x=list(r["P_roya"]), y=list(r["P_roya"].values()), labels={"x": "riesgo", "y": "probabilidad"}, title="P(roya) — CPT experta"), use_container_width=True)
    c2.plotly_chart(px.bar(x=list(r["P_temp_proxima_semana"]), y=list(r["P_temp_proxima_semana"].values()), labels={"x": "temperatura", "y": "probabilidad"}, title="P(temperatura próxima semana)"), use_container_width=True)
with t3:
    rc = st.selectbox("Lluvia esperada (SUPUESTO tuyo, sin datos de lluvia)", ["baja", "media", "alta"], 1); sol = tb.schedule(rc, cap)
    st.write(f"Días lluviosos asumidos: {', '.join(sol['wet_days'])} · nodos explorados: {sol['nodes']}")
    if sol["feasible"]:
        df = pd.DataFrame([{"día": DAYS.index(v), "labor": k} for k, v in sol["plan"].items()]).sort_values("día")
        df["día"] = df["día"].map(lambda i: DAYS[i]); st.dataframe(df, hide_index=True)
    else: st.error(sol["note"])
with t4:
    if "hist" not in st.session_state: st.session_state.hist = []
    for m in st.session_state.hist:
        with st.chat_message(m["role"]): st.write(m["content"])
    q = st.chat_input("Pregunta sobre roya, labores o documentos técnicos")
    if q:
        with st.chat_message("user"): st.write(q)
        with st.spinner("Pensando..."): ans, trace = CoffeeAgent(tb).run(st.session_state.hist, q)
        with st.chat_message("assistant"):
            st.write(ans)
            with st.expander("Herramientas usadas"): st.json(trace)
        st.session_state.hist += [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]
