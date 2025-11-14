"""
Iris Species Classification – Streamlit App
Universidad de la Costa – Data Mining
Autores: Kevin David Gallardo, Mauricio Carrillo
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
import plotly.express as px
import plotly.graph_objects as go
import pickle
import io

# ============================================================
# Carga del dataset
# ============================================================
@st.cache_data
def load_data():
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df.columns = [
        "sepal_length", "sepal_width",
        "petal_length", "petal_width", "species"
    ]

    # Convertimos la especie en nombres
    target_names = iris.target_names
    df["species_name"] = df["species"].apply(lambda x: target_names[int(x)])

    return df, target_names


# ============================================================
# Preprocesamiento y división Train/Test
# ============================================================
@st.cache_data
def prepare_data(df, test_size=0.2, seed=42):
    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values
    y = df["species_name"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ============================================================
# Entrenamiento del modelo
# ============================================================
@st.cache_resource
def train_model(X_train, y_train, n_estimators=100, seed=42):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed
    )
    model.fit(X_train, y_train)
    return model


# ============================================================
# Cálculo de métricas
# ============================================================
def get_metrics(model, X_test, y_test):
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, average="macro"),
        "recall": recall_score(y_test, predictions, average="macro"),
        "f1": f1_score(y_test, predictions, average="macro"),
        "y_pred": predictions
    }
    return metrics


# ============================================================
# APP EN STREAMLIT
# ============================================================
st.set_page_config(page_title="Iris Classification", layout="wide")

st.title("🌸 Iris Species Classification")
st.caption("Proyecto Final – Universidad de la Costa (Data Mining)")

# Cargamos los datos
df, target_names = load_data()

# ------------------------------
# Sidebar
# ------------------------------
with st.sidebar:
    st.header("Configuración del modelo")

    test_percentage = st.slider("Porcentaje de Test", 10, 40, 20, step=5) / 100
    trees = st.slider("Número de árboles (Random Forest)", 10, 300, 100, step=10)
    seed = st.number_input("Random Seed", value=42)

    st.markdown("---")
    st.subheader("Integrantes")
    st.text("Kevin David Gallardo")
    st.text("Mauricio Carrillo")
    st.markdown("---")
    st.write("Dataset Iris: 150 muestras, 3 especies.")

# Preprocesamiento y modelo
X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(
    df, test_size=test_percentage, seed=seed
)
model = train_model(X_train_scaled, y_train, n_estimators=trees, seed=seed)

# Métricas
metrics = get_metrics(model, X_test_scaled, y_test)

# ===============================
# MÉTRICAS Y VISUALIZACIONES
# ===============================
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Métricas del modelo")

    st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    st.metric("Precision", f"{metrics['precision']:.4f}")
    st.metric("Recall", f"{metrics['recall']:.4f}")
    st.metric("F1-score", f"{metrics['f1']:.4f}")

    st.markdown("### Reporte de clasificación")
    st.text(classification_report(y_test, metrics["y_pred"]))

    # Matriz de confusión
    cm = confusion_matrix(y_test, metrics["y_pred"], labels=target_names)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)

    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Blues",
        labels={"x": "Predicho", "y": "Real"}
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    # Guardar modelo
    st.subheader("Descargar modelo entrenado")
    #convertir modelo para descargar
    model_bytes = io.BytesIO()
    pickle.dump({"model": model, "scaler": scaler}, model_bytes)
    model_bytes.seek(0)
    
    st.download_button(
        label = "📥 Descargar Model.pkl,
        data = model_bytes,
        file_name="model.pkl",
        mime="application/octet-stream"
)
with col2:
    st.subheader("🔎 Análisis exploratorio")

    st.markdown("**Scatter Matrix**")
    fig_matrix = px.scatter_matrix(
        df,
        dimensions=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        color="species_name",
        title="Scatter Matrix – Iris"
    )
    st.plotly_chart(fig_matrix, use_container_width=True)

    st.markdown("**Histogramas**")
    option = st.selectbox("Selecciona una característica", df.columns[:4])
    fig_hist = px.histogram(df, x=option, color="species_name", nbins=20)
    st.plotly_chart(fig_hist, use_container_width=True)


# ===============================
# PREDICCIÓN INTERACTIVA
# ===============================
st.markdown("---")
st.header("🌼 Predicción de especie")

colA, colB, colC, colD = st.columns(4)
with colA:
    sl = st.number_input("Sepal length", min_value=0.0, value=float(df["sepal_length"].mean()))
with colB:
    sw = st.number_input("Sepal width", min_value=0.0, value=float(df["sepal_width"].mean()))
with colC:
    pl = st.number_input("Petal length", min_value=0.0, value=float(df["petal_length"].mean()))
with colD:
    pw = st.number_input("Petal width", min_value=0.0, value=float(df["petal_width"].mean()))

if st.button("Predecir especie"):
    new_point = np.array([[sl, sw, pl, pw]])
    new_scaled = scaler.transform(new_point)
    pred = model.predict(new_scaled)[0]

    st.success(f"Especie predicha: **{pred}**")

    # Gráfico 3D
    fig3d = px.scatter_3d(
        df,
        x="petal_length",
        y="petal_width",
        z="sepal_length",
        color="species_name",
        title="Ubicación de la nueva muestra"
    )

    fig3d.add_trace(
        go.Scatter3d(
            x=[pl], y=[pw], z=[sl],
            mode="markers",
            marker=dict(size=8, color="black", symbol="diamond"),
            name="Nueva muestra"
        )
    )

    st.plotly_chart(fig3d, use_container_width=True)

# Notas finales
st.markdown("---")
st.subheader("Notas del proyecto")
st.write("""
- Se aplicó estandarización con StandardScaler.
- Algoritmo utilizado: Random Forest.
- Conjunto de prueba generado con Train/Test split estratificado.
- Se incluyen métricas principales + matriz de confusión.
""")
