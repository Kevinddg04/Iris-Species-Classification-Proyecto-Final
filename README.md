# Iris-Species-Classification-Proyecto-Final
**Universidad de la Costa – Data Mining / Control Automáticos II**  
**Integrantes:**  
- Kevin David Gallardo  
- Mauricio Carrillo  

---

## 📌 Descripción del proyecto
Este proyecto consiste en entrenar un modelo capaz de **clasificar la especie de una flor Iris** usando sus cuatro medidas principales:  
- Largo del sépalo  
- Ancho del sépalo  
- Largo del pétalo  
- Ancho del pétalo  

El dataset utilizado es el clásico **Iris Dataset**, el cual contiene 150 muestras divididas en tres especies:  
- *Iris setosa*  
- *Iris versicolor*  
- *Iris virginica*  

El objetivo final es crear un **dashboard interactivo con Streamlit**, donde cualquier usuario pueda:
- Ver las métricas del modelo  
- Explorar el dataset mediante gráficos  
- Ingresar sus propios valores para obtener una predicción  
- Ver la ubicación del punto en un gráfico **3D**

---

## 🚀 Tecnologías utilizadas
- Python  
- Streamlit  
- Scikit-Learn  
- Pandas / Numpy  
- Plotly  

---

## 🧠 Metodología
El flujo de trabajo que seguimos fue:

### 1. **Comprensión del dataset (EDA)**
Exploramos la estructura del dataset e hicimos visualizaciones como:
- Histogramas por característica  
- Scatter Matrix  
- Gráfico 3D  
- Correlaciones  

### 2. **Preprocesamiento**
- Estandarización de los datos (StandardScaler)  
- División del dataset en Train/Test (estratificado)  

### 3. **Modelo**
Entrenamos un **Random Forest**, ya que:
- Funciona muy bien con datasets pequeños  
- Tiene buen desempeño sin tanto ajuste  
- Reduce riesgo de overfitting  

### 4. **Validación**
Medimos:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Matriz de confusión  

### 5. **Interfaz en Streamlit**
Se desarrolló un dashboard con:
- Panel de métricas  
- Exploración visual del dataset  
- Predicción interactiva  
- Gráfico 3D con la posición del punto ingresado  

---

## ▶️ Ejecución del proyecto
1. Instalar dependencias:
```bash
pip install -r requirements.txt
