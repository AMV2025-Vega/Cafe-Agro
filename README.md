# Asistente agroclimático para café (SI3003 – Proyecto final)
Red bayesiana (eliminación de variables, propia) + CSP (backtracking, propio) + agente LLM con RAG y 3 herramientas + dashboard Streamlit.

## Ejecutar
```bash
python -m venv venv && venv\Scripts\activate      # Windows
pip install -r requirements.txt
```
1. **Datos:** CSV de temperatura de la estación IDEAM en `data/raw/ideam_station.csv` (separador `;`, codificación latin-1, columnas `fecha` y `valor`). Fuente: [COMPLETAR enlace de datos.gov.co].
2. **Corpus:** los PDFs de Cenicafé están en `corpus/` (referencias completas en el informe).
3. **LLM (API compatible con OpenAI, p. ej. Groq).** No pongas la key en ningún archivo:
```powershell
$env:LLM_API_KEY="<tu key>"
$env:LLM_BASE_URL="https://api.groq.com/openai/v1"
$env:LLM_MODEL="openai/gpt-oss-120b"
$env:DATA_PATH="data/raw/ideam_station.csv"
```
4. **Evaluación y app:**
```bash
python -m src.evaluate_model
python -m src.evaluate_agent
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
