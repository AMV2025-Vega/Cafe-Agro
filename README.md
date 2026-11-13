# Asistente agroclimático para café (SI3003 – Proyecto final)
Red bayesiana (eliminación de variables, propia) + CSP (backtracking, propio) + agente LLM con RAG y 3 herramientas + dashboard Streamlit.

## Ejecutar
```bash
pip install -r requirements.txt
# 1) Datos: descarga de datos.gov.co "Datos Hidrometeorológicos Crudos IDEAM" UNA estación cafetera (CSV) -> data/raw/ideam_station.csv
#python -m src.make_demo_data
# 2) Corpus: coloca PDFs/TXT de Cenicafé en corpus/ (borra el texto DEMO)
# 3) LLM (API compatible OpenAI; ej. Groq gratis):
export LLM_API_KEY=...  LLM_BASE_URL=https://api.groq.com/openai/v1  LLM_MODEL=llama-3.3-70b-versatile
export DATA_PATH=data/raw/ideam_station.csv
python -m src.evaluate_model        # métricas vs líneas base -> results/metricas_modelo.json
python -m src.evaluate_agent        # 12 preguntas fijas     -> results/evaluacion_agente.csv
streamlit run app.py
```
## Método (Plan B: solo temperatura)
- **Datos:** serie de TEMPERATURA de una estación IDEAM (CSV `fecha;valor`). No se usan datos simulados. No se consiguieron series de lluvia ni humedad de la misma estación.
- **Parte 1:** red bayesiana `TempNext | Temp, Trend, Season` (conteos + Laplace, eliminación de variables propia) y `Roya | TempNext` (CPT experta, no validada). Validación temporal 70/30; líneas base: persistencia y clase mayoritaria. CSP de labores con MRV (la lluvia es un supuesto del usuario).
- **Parte 2:** loop agéntico manual (T_max=6, errores como observación), RAG TF-IDF con citas `[fuente: archivo, p. N]`, herramientas `get_risk`, `schedule`, `search_docs`.
## Limitaciones
Solo temperatura y una estación (proxy de finca). Sin etiquetas reales de roya: el riesgo es un supuesto experto. Sin lluvia/humedad, factores clave de la roya. No es consejo agronómico.
## Fuentes
• https://publicaciones.cenicafe.org/index.php/index/search/search
• https://herramientas.datos.gov.co/usos/reporte-temperatura-red-de-estaciones-ideam-municipio-de-ibague
