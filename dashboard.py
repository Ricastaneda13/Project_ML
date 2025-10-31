#Librerias
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

#Modelo + Archivos

@st.cache_resource
def load_model():
    model = joblib.load('models/xgboost_model.joblib')
    le = joblib.load('models/income_label_encoder.joblib')
    model_cols = joblib.load('models/model_columns.joblib')
    return model, le, model_cols

# Ejecutamos la función y guardamos las 3 cosas en variables
model, le, model_cols = load_model()

st.title('Predicción de Ingresos (Modelo XGBoost)')
st.write('Esta app predice si una persona ganará más o menos de $50K al año.')

# --- 3. Crear el Formulario de Entradas ---
st.sidebar.header('Parámetros de Entrada')

input_data = {}

# --- Controles Numéricos ---
age = st.sidebar.slider('Edad (age)', 17, 90, 30)
input_data['age'] = age

education_num = st.sidebar.slider('Años de Educación (education-num)', 1, 16, 10)
input_data['education-num'] = education_num

capital_gain = st.sidebar.number_input('Ganancia de Capital (capital-gain)', 0, 99999, 0)
input_data['capital-gain'] = capital_gain

capital_loss = st.sidebar.number_input('Pérdida de Capital (capital-loss)', 0, 4356, 0)
input_data['capital-loss'] = capital_loss

hours_per_week = st.sidebar.slider('Horas por Semana (hours-per-week)', 1, 99, 40)
input_data['hours-per-week'] = hours_per_week

# --- Controles Categóricos (Menús desplegables) ---
workclass = st.sidebar.selectbox('Clase de Trabajo (workclass)', 
    ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Self-emp-inc', 'Federal-gov', 'Without-pay'])
input_data['workclass'] = workclass

marital_status = st.sidebar.selectbox('Estado Civil (marital-status)',
    ['Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
input_data['marital-status'] = marital_status

occupation = st.sidebar.selectbox('Ocupación (occupation)',
    ['Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical', 'Sales', 'Other-service', 'Machine-op-inspct', 
     'Transport-moving', 'Handlers-cleaners', 'Farming-fishing', 'Tech-support', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces'])
input_data['occupation'] = occupation

relationship = st.sidebar.selectbox('Relación (relationship)',
    ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative'])
input_data['relationship'] = relationship

race = st.sidebar.selectbox('Raza (race)',
    ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
input_data['race'] = race

sex = st.sidebar.selectbox('Sexo (sex)', ['Male', 'Female'])
input_data['sex'] = sex

native_country = st.sidebar.selectbox('País Nativo (native-country)',
    ['United-States', 'Mexico', 'Philippines', 'Germany', 'Puerto-Rico', 'Canada', 'El-Salvador', 'India', 'Cuba', 'England', 'Jamaica', 'South', 'China', 'Italy', 'Dominican-Republic', 'Vietnam', 'Guatemala', 'Japan', 'Poland', 'Columbia', 'Haiti', 'Portugal', 'Taiwan', 'Iran', 'Nicaragua'])
input_data['native-country'] = native_country


# --- 4. Botón de Predicción ---

if st.sidebar.button('Predecir Ingresos'):

    input_df = pd.DataFrame([input_data])
    
    # Aplicar 'get_dummies'
    input_processed = pd.get_dummies(input_df)
    
    # Reindexar para alinear columnas
    input_processed = input_processed.reindex(columns=model_cols, fill_value=0)
    
    # Predicción
    prediction = model.predict(input_processed)
    
    # Convertir respuesta a texto
    predicted_class = le.inverse_transform(prediction)[0]
    
    # Mostrar resultado
    st.subheader('Resultado de la Predicción:')
    if predicted_class == '<=50K':
        st.success(f'El modelo predice: **{predicted_class}** 💸')
    else:
        st.error(f'El modelo predice: **{predicted_class}** 💰')

#toactivate streamlit run dashboard.py