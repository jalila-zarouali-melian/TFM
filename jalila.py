from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Helper function to load the DataFrame using caching
@st.cache(allow_output_mutation=True)
def load_dataframe():
    if os.path.exists('./dataset.csv'):
        return pd.read_csv('dataset.csv', index_col=None)
    return None

# Load the DataFrame
df = load_dataframe()

with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Automodeladorv.1")
    choice = st.radio("Navigation", ["subir fichero","Análisis descriptivo","Modelaje", "Descargar modelo"])
    st.info("Esta aplicación automatiza el proceso de creación de un modelo.")

if choice == "subir fichero":
    st.title("subir fichero")
    file = st.file_uploader("subir fichero")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Análisis descriptivo":
    st.title("Exploratory Data Analysis")
    if df is not None:
        profile_df = df.profile_report()
        st_profile_report(profile_df)
    else:
        st.warning("Please upload a dataset first.")

def eliminar_columnas_con_nulos(dataset, porcentaje_limite=0.2):
    num_nulos_limite = len(dataset) * porcentaje_limite
    dataset_limpio = dataset.dropna(axis=1, thresh=num_nulos_limite)

    return dataset_limpio
def convertir_a_categoricas(dataset, valor_limite=10):
    columnas = dataset.columns
    for columna in columnas:
        if dataset[columna].nunique() < valor_limite:
            dataset[columna] = dataset[columna].astype('category')

    return dataset
def imputar_valores_nulos(dataset):
    variables_numericas = dataset.select_dtypes(include=['int64', 'float64'])
    for columna in variables_numericas:
        mediana = dataset[columna].median()
        dataset[columna].fillna(mediana, inplace=True)

    variables_categoricas = dataset.select_dtypes(include='category')
    for columna in variables_categoricas:
        moda = dataset[columna].mode().iloc[0]
        dataset[columna].fillna(moda, inplace=True)

    return dataset
def normalizar_dataset(dataset):
    scaler = StandardScaler()
    variables_numericas = dataset.select_dtypes(include=['int64', 'float64'])
    dataset[variables_numericas.columns] = scaler.fit_transform(variables_numericas)

    return dataset
def label_encoding(dataset):
    label_encoder = LabelEncoder()
    for columna in dataset.select_dtypes(include='category'):
        dataset[columna] = label_encoder.fit_transform(dataset[columna])

    return dataset
def obtener_columnas_orden_jerarquico(dataset):
    columnas_categoricas = dataset.select_dtypes(include='category').columns.tolist()
    columnas_orden_jerarquico = st.multiselect('Elige las variables con un orden jerárquico', columnas_categoricas)

    return columnas_orden_jerarquico
def verificar_desbalanceo(dataset, variable_objetivo):
    conteo_clases = dataset[variable_objetivo].value_counts()
    porcentaje_menor_clase = (conteo_clases.min() / conteo_clases.sum()) * 100
    umbral_desbalanceo = 10  # Por ejemplo, consideraremos que hay desbalanceo si una clase representa menos del 10% del total
    desbalanceo = porcentaje_menor_clase < umbral_desbalanceo

    return desbalanceo

if choice == "Preprocesado":
    st.title("Preprocesado")
    if df is not None:
        st.subheader("Tratamiento de nulos y missings")
        df_limpio = eliminar_columnas_con_nulos(df, porcentaje_limite=0.2)
        st.dataframe(df_limpio)

        st.subheader("Categorización")
        df_categorico = convertir_a_categoricas(df_limpio, valor_limite=10)
        st.dataframe(df_categorico.dtypes)

        st.subheader("Imputación de nulos")
        df_imputado = imputar_valores_nulos(df_categorico)
        st.dataframe(df_imputado)

        st.subheader("Normalización")
        respuesta = st.selectbox("¿Deseas normalizar el dataset?", ["Sí", "No"])
        if respuesta == "Sí":
            df_normalizado = normalizar_dataset(df_imputado)
            st.write("Dataset normalizado:")
            st.dataframe(df_normalizado)
        elif respuesta == "No":
            st.write("Dataset sin normalizar.")
            df_normalizado = df_imputado
        else:
            st.warning("Respuesta no válida. Por favor, selecciona 'Sí' o 'No'.")

        st.subheader("Encoding")
        columnas_orden_jerarquico = obtener_columnas_orden_jerarquico(df_normalizado)
        df_encoded = label_encoding(df_normalizado)
        if columnas_orden_jerarquico:
            df_encoded[columnas_orden_jerarquico] = df_encoded[columnas_orden_jerarquico].astype('category')
        st.dataframe(df_encoded)

        st.subheader("Checkear desbalanceo")
        variable_objetivo = st.selectbox('Elige la variable objetivo', df_encoded.columns)
        desbalanceo = verificar_desbalanceo(df_encoded, variable_objetivo)
        if desbalanceo:
            st.warning("El dataset sufre de desbalanceo en la variable objetivo.")
        else:
            st.success("El dataset no sufre de desbalanceo en la variable objetivo.")
    else:
        st.warning("Please upload a dataset first.")

if choice == "Modelaje":
    if df is not None:
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
        if st.button('Run Modelling'):
            # Convert 'price' column to numeric, coercing non-numeric values to NaN
            setup(df, target=chosen_target, categorical_features=['Sex'])  # Specify 'Sex' as a categorical feature
            setup_df = pull()
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.dataframe(compare_df)
            save_model(best_model, 'best_model')
    else:
        st.warning("Please upload a dataset first.")

if choice == "Descargar modelo":
    if df is not None:
        with open('best_model.pkl', 'rb') as f:
            st.download_button('Download Model', f, file_name="best_model.pkl")
    else:
        st.warning("Please upload a dataset first.")
