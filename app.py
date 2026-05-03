import os

BASE_DIR = os.path.dirname(__file__)
file_path = os.path.join(BASE_DIR, "covid.csv")

df = pd.read_csv(file_path)

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# -------------------------------------------------
# CONFIGURACIÓN
# -------------------------------------------------
st.set_page_config(page_title="Dashboard COVID", layout="wide")

st.title("🦠 Análisis Epidemiológico COVID-19")

# -------------------------------------------------
# CARGA DE DATOS
# -------------------------------------------------
df = pd.read_csv("Covid_pequeño.csv")

# Limpieza
df.columns = df.columns.str.strip()

df["fecha_reporte"] = pd.to_datetime(df["fecha reporte web"], errors="coerce")
df["edad"] = pd.to_numeric(df["Edad"], errors="coerce")
df["es_fallecido"] = df["Fecha de muerte"].notnull().astype(int)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
menu = st.sidebar.radio(
    "Secciones",
    ["Contexto", "EDA", "Indicadores", "Modelo ML", "Conclusiones"]
)

# -------------------------------------------------
# CONTEXTO
# -------------------------------------------------
if menu == "Contexto":
    st.subheader("Contexto epidemiológico")
    st.write("""
    Este dashboard permite analizar el comportamiento del COVID-19,
    identificando patrones de contagio, mortalidad y factores de riesgo.
    """)

# -------------------------------------------------
# EDA
# -------------------------------------------------
elif menu == "EDA":
    st.subheader("Exploración de datos")

    st.dataframe(df.head())

    st.markdown("### Distribución de edad")
    st.plotly_chart(px.histogram(df, x="edad"))

    st.markdown("### Casos por sexo")
    sexo = df["Sexo"].value_counts().reset_index()
    sexo.columns = ["Sexo", "count"]
    st.plotly_chart(px.bar(sexo, x="Sexo", y="count"))

    st.markdown("### Top 10 departamentos")
    depto = df["Nombre departamento"].value_counts().head(10).reset_index()
    depto.columns = ["Departamento", "count"]
    st.plotly_chart(px.bar(depto, x="Departamento", y="count"))

# -------------------------------------------------
# INDICADORES
# -------------------------------------------------
elif menu == "Indicadores":
    st.subheader("Indicadores epidemiológicos")

    total = len(df)
    muertes = df["es_fallecido"].sum()
    tasa = (muertes / total) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Total casos", total)
    col2.metric("Total muertes", int(muertes))
    col3.metric("Tasa mortalidad (%)", round(tasa, 2))

    casos_fecha = df.groupby("fecha_reporte").size().reset_index(name="casos")
    st.plotly_chart(px.line(casos_fecha, x="fecha_reporte", y="casos"))

# -------------------------------------------------
# MACHINE LEARNING
# -------------------------------------------------
elif menu == "Modelo ML":
    st.subheader("🤖 Predicción de mortalidad")

    df_ml = df[["edad", "Sexo", "es_fallecido"]].dropna()
    df_ml["Sexo"] = df_ml["Sexo"].astype("category").cat.codes

    X = df_ml[["edad", "Sexo"]]
    y = df_ml["es_fallecido"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    st.metric("Accuracy", round(acc, 3))
    st.write("Matriz de confusión")
    st.write(cm)

# -------------------------------------------------
# CONCLUSIONES
# -------------------------------------------------
else:
    st.subheader("Conclusiones")
    st.write("""
    - La edad influye en la mortalidad
    - Existen diferencias por sexo y territorio
    - El modelo permite estimar riesgo
    - La analítica apoya la toma de decisiones en salud pública
    """)
