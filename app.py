import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# -------------------------------------------------
# CONFIGURACIÓN
# -------------------------------------------------
st.set_page_config(
    page_title="Dashboard COVID-19",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
        .block-container { padding-top: 2rem; }
        [data-testid="metric-container"] {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# CARGA DE DATOS (con caché para mejor rendimiento)
# -------------------------------------------------
@st.cache_data
def cargar_datos(ruta: str) -> pd.DataFrame:
    """Carga y limpia el CSV de COVID."""
    df = pd.read_csv(ruta)
    df.columns = df.columns.str.strip()

    # Fechas
    if "fecha reporte web" in df.columns:
        df["fecha_reporte"] = pd.to_datetime(df["fecha reporte web"], errors="coerce")
    
    # Edad
    if "Edad" in df.columns:
        df["edad"] = pd.to_numeric(df["Edad"], errors="coerce")

    # Variable objetivo
    if "Fecha de muerte" in df.columns:
        df["es_fallecido"] = df["Fecha de muerte"].notnull().astype(int)

    return df


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_NAME = "covid.csv"  # <-- cambia aquí el nombre de tu archivo
file_path = os.path.join(BASE_DIR, FILE_NAME)

if not os.path.exists(file_path):
    st.error(f"❌ No se encontró el archivo: `{file_path}`")
    st.stop()

df = cargar_datos(file_path)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.title("🦠 COVID-19")
    st.markdown("---")
    menu = st.radio(
        "Secciones",
        ["📋 Contexto", "🔍 EDA", "📊 Indicadores", "🤖 Modelo ML", "✅ Conclusiones"],
    )

    st.markdown("---")
    st.caption(f"Registros cargados: **{len(df):,}**")

# -------------------------------------------------
# CONTEXTO
# -------------------------------------------------
if menu == "📋 Contexto":
    st.title("🦠 Análisis Epidemiológico COVID-19")
    st.subheader("Contexto epidemiológico")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        Este dashboard permite analizar el comportamiento del **COVID-19** en Colombia,
        identificando patrones de contagio, mortalidad y factores de riesgo a partir de
        datos oficiales.

        **¿Qué encontrarás aquí?**
        - 🔍 **EDA**: Exploración y distribución de los datos
        - 📊 **Indicadores**: Métricas clave y evolución temporal
        - 🤖 **Modelo ML**: Predicción de mortalidad con regresión logística
        - ✅ **Conclusiones**: Hallazgos relevantes
        """)
    with col2:
        st.info(f"**Fuente de datos**\n\n`{FILE_NAME}`\n\n{len(df):,} registros")

# -------------------------------------------------
# EDA
# -------------------------------------------------
elif menu == "🔍 EDA":
    st.title("🔍 Exploración de Datos")

    with st.expander("Vista previa del dataset", expanded=False):
        st.dataframe(df.head(50), use_container_width=True)
        st.caption(f"Dimensiones: {df.shape[0]:,} filas × {df.shape[1]} columnas")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Distribución de edad")
        if "edad" in df.columns:
            fig = px.histogram(
                df.dropna(subset=["edad"]),
                x="edad",
                nbins=40,
                color_discrete_sequence=["#636EFA"],
                labels={"edad": "Edad", "count": "Cantidad"},
            )
            fig.update_layout(bargap=0.05)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Columna 'Edad' no encontrada.")

    with col2:
        st.markdown("#### Casos por sexo")
        if "Sexo" in df.columns:
            sexo = df["Sexo"].value_counts().reset_index()
            sexo.columns = ["Sexo", "Casos"]
            fig = px.pie(
                sexo,
                names="Sexo",
                values="Casos",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Columna 'Sexo' no encontrada.")

    st.markdown("#### Top 10 departamentos con más casos")
    if "Nombre departamento" in df.columns:
        depto = df["Nombre departamento"].value_counts().head(10).reset_index()
        depto.columns = ["Departamento", "Casos"]
        fig = px.bar(
            depto,
            x="Casos",
            y="Departamento",
            orientation="h",
            color="Casos",
            color_continuous_scale="Blues",
            labels={"Casos": "Número de casos"},
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Columna 'Nombre departamento' no encontrada.")

# -------------------------------------------------
# INDICADORES
# -------------------------------------------------
elif menu == "📊 Indicadores":
    st.title("📊 Indicadores Epidemiológicos")

    total = len(df)
    muertes = int(df["es_fallecido"].sum()) if "es_fallecido" in df.columns else 0
    tasa = round((muertes / total) * 100, 2) if total > 0 else 0
    recuperados = int((df["es_fallecido"] == 0).sum()) if "es_fallecido" in df.columns else total - muertes

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total casos", f"{total:,}")
    col2.metric("Total fallecidos", f"{muertes:,}")
    col3.metric("Recuperados", f"{recuperados:,}")
    col4.metric("Tasa mortalidad", f"{tasa}%", delta=None)

    st.markdown("---")

    if "fecha_reporte" in df.columns:
        st.markdown("#### Evolución temporal de casos")
        casos_fecha = (
            df.dropna(subset=["fecha_reporte"])
            .groupby("fecha_reporte")
            .size()
            .reset_index(name="casos")
        )
        fig = px.line(
            casos_fecha,
            x="fecha_reporte",
            y="casos",
            labels={"fecha_reporte": "Fecha", "casos": "Nuevos casos"},
            color_discrete_sequence=["#EF553B"],
        )
        fig.update_traces(line_width=1.5)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No se pudo generar la línea temporal.")

# -------------------------------------------------
# MACHINE LEARNING
# -------------------------------------------------
elif menu == "🤖 Modelo ML":
    st.title("🤖 Predicción de Mortalidad")
    st.markdown("Modelo de **Regresión Logística** entrenado con edad y sexo del paciente.")

    required_cols = {"edad", "Sexo", "es_fallecido"}
    if not required_cols.issubset(df.columns):
        st.error(f"Faltan columnas necesarias: {required_cols - set(df.columns)}")
    else:
        df_ml = df[["edad", "Sexo", "es_fallecido"]].dropna().copy()
        df_ml["Sexo"] = df_ml["Sexo"].astype("category").cat.codes

        X = df_ml[["edad", "Sexo"]]
        y = df_ml["es_fallecido"]

        test_size = st.slider("Tamaño del conjunto de prueba", 0.1, 0.4, 0.2, 0.05)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{acc:.1%}")
        col2.metric("Registros entrenamiento", f"{len(X_train):,}")
        col3.metric("Registros prueba", f"{len(X_test):,}")

        st.markdown("#### Matriz de confusión")
        labels = ["Sobrevivió", "Falleció"]
        fig = ff.create_annotated_heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            showscale=True,
        )
        fig.update_layout(
            xaxis_title="Predicho",
            yaxis_title="Real",
            yaxis_autorange="reversed",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Importancia de variables (coeficientes)")
        coef_df = pd.DataFrame({
            "Variable": ["Edad", "Sexo"],
            "Coeficiente": model.coef_[0],
        })
        fig2 = px.bar(
            coef_df,
            x="Variable",
            y="Coeficiente",
            color="Coeficiente",
            color_continuous_scale="RdBu",
            title="Mayor coeficiente → mayor influencia en la predicción",
        )
        st.plotly_chart(fig2, use_container_width=True)

# -------------------------------------------------
# CONCLUSIONES
# -------------------------------------------------
else:
    st.title("✅ Conclusiones")

    conclusiones = [
        ("🧓 Edad", "La edad es el principal factor de riesgo. Los pacientes mayores presentan tasas de mortalidad significativamente más altas."),
        ("⚕️ Sexo", "Se observan diferencias estadísticas entre hombres y mujeres en términos de contagio y gravedad."),
        ("🗺️ Territorio", "Existen concentraciones geográficas claras, con algunos departamentos concentrando la mayoría de los casos."),
        ("🤖 Modelo ML", "La regresión logística permite estimar riesgo individual con variables básicas, siendo un punto de partida para modelos más complejos."),
        ("📢 Política pública", "La analítica de datos epidemiológicos es una herramienta clave para la toma de decisiones en salud pública."),
    ]

    for icono_titulo, texto in conclusiones:
        st.markdown(f"**{icono_titulo}**")
        st.write(texto)
        st.markdown("---")
