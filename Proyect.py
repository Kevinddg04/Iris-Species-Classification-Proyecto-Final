# Proyect.py
"""
Iris Species Classification - Streamlit App
Universidad de la Costa - Data Mining
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

# ---------------------------------------------------------
# Carga y preprocesamiento del dataset
# ---------------------------------------------------------
@st.cache_data
def load_data():
    ds = load_iris(as_frame=True)
    df = ds.frame.copy()
    df.columns = ['sepal_length', 'sepal_width',
                  'petal_length', 'petal_width', 'species']
    
    df["species_name"] = df["species"].apply(lambda x: ds.target_names[int(x)])
    return df, ds.target_names

@st.cache_data
def preprocess_and_split(df, test_size=0.2, random_state=42):
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = df['species_name']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

@st.cache_resource
def train_model(X_train, y_train, n_estimators=100, random_state=42):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model

def compute_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    return acc, prec, rec, f1, y_pred

# ---------------------------------------------------------
# Configuración de la página
# ---------------------------------------------------------
st.set_page_config(page_title="Iris Species Classification",
                   layout="wide",
                   initial_sidebar_state="expanded")

st.title("🌸 Iris Species Classification")
st.caption("Proyecto Final — Universidad de la Costa — Data Mining")

# ---------------------------------------------------------
# Sidebar: configuración del modelo
# ---------------------------------------------------------
df, target_names = load_data()

with st.sidebar:
    st.header("⚙️ Configuración del Modelo")
    test_size = st.slider("Porcentaje de datos para Test", 10, 40, 20, step=5) / 100
    n_estimators = st.slider("Número de árboles (Random Forest)", 10, 300, 100, step=10)
    random_state = st.number_input("Random Seed", 0, 9999, 42)

    st.markdown("---")
    st.header("👥 Integrantes")
    st.text("Kevin David Gallardo")
    st.text("Mauricio Carrillo")
    st.markdown("---")

# ---------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------
X_train, X_test, y_train, y_test, scaler = preprocess_and_split(
    df, test_size=test_size, random_state=random_state)

model = train_model(X_train, y_train,
                    n_estimators=n_estimators,
                    random_state=random_state)

acc, prec, rec, f1, y_pred = compute_metrics(model, X_test, y_test)

# ---------------------------------------------------------
# Layout general
# ---------------------------------------------------------
col1, col2 = st.columns([1, 2])

# --------------------- Métricas --------------------------
with col1:
    st.subheader("📊 Métricas del Modelo")
    st.metric("Accuracy", f"{acc:.4f}")
    st.metric("Precisión (macro)", f"{prec:.4f}")
    st.metric("Recall (macro)", f"{rec:.4f}")
    st.metric("F1-Score (macro)", f"{f1:.4f}")

    st.markdown("**📄 Reporte de Clasificación:**")
    st.text(classification_report(y_test, y_pred))

    st.markdown("**🔢 Matriz de Confusión**")
    cm = confusion_matrix(y_test, y_pred, labels=target_names)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues")
    st.plotly_chart(fig_cm, use_container_width=True)

    # Descargar modelo
    st.subheader("📥 Descargar Modelo Entrenado")
    model_bytes = io.BytesIO()
    pickle.dump({"model": model, "scaler": scaler}, model_bytes)
    model_bytes.seek(0)

    st.download_button(
        label="⬇️ Descargar model.pkl",
        data=model_bytes,
        file_name="model.pkl",
        mime="application/octet-stream"
    )

# --------------------- Visualización ---------------------
with col2:
    st.subheader("🔍 Análisis Exploratorio")
    fig_sm = px.scatter_matrix(
        df,
        dimensions=['sepal_length', 'sepal_width',
                    'petal_length', 'petal_width'],
        color='species_name'
    )
    st.plotly_chart(fig_sm, use_container_width=True)

    hist_feature = st.selectbox("Selecciona un feature:", df.columns[:4])
    fig_hist = px.histogram(df, x=hist_feature, color="species_name")
    st.plotly_chart(fig_hist, use_container_width=True)

# ---------------------------------------------------------
# Predicción interactiva
# ---------------------------------------------------------
st.header("🔮 Predicción Interactiva")

c1, c2, c3, c4 = st.columns(4)
sepal_length = c1.number_input("Sepal length", 0.0, 10.0, float(df['sepal_length'].mean()))
sepal_width = c2.number_input("Sepal width", 0.0, 10.0, float(df['sepal_width'].mean()))
petal_length = c3.number_input("Petal length", 0.0, 10.0, float(df['petal_length'].mean()))
petal_width = c4.number_input("Petal width", 0.0, 10.0, float(df['petal_width'].mean()))

if st.button("Predecir especie"):
    X_new = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    X_scaled = scaler.transform(X_new)
    pred = model.predict(X_scaled)[0]

    st.success(f"🌼 Especie predicha: **{pred}**")

    fig3d = px.scatter_3d(
        df,
        x='petal_length', y='petal_width', z='sepal_length',
        color='species_name'
    )

    fig3d.add_trace(go.Scatter3d(
        x=[petal_length],
        y=[petal_width],
        z=[sepal_length],
        mode="markers",
        marker=dict(size=8, symbol="diamond", line=dict(width=2, color="black")),
        name="Nueva muestra"
    ))

    st.plotly_chart(fig3d, use_container_width=True)


