from streamlit_pandas_profiling import st_profile_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from io import BytesIO
import seaborn as sns
import os
from pandasai.llm.openai import OpenAI
from pandasai import PandasAI
import matplotlib.pyplot as plt
import matplotlib
from streamlit_option_menu import option_menu
from operator import index
import streamlit as st
import numpy as np
import seaborn as sns
import plotly.express as px
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from io import BytesIO
from sklearn.utils import resample
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import PyPDF2
import random
import itertools
import streamlit as st
from io import StringIO
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import SVMRetriever
from langchain.chains import QAGenerationChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import CallbackManager
from langchain.embeddings import HuggingFaceEmbeddings





with open('style.css', 'r') as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

uploaded_file = None

EXAMPLE_NO = 2

def streamlit_menu(example=2):
    if example == 2:
        selected = option_menu(
            menu_title=None,
            options=["Ciencia de Datos", "Análisis de datos"],
            icons=["house", "book"],
            menu_icon="cast",
            orientation="horizontal",
        )
        return selected

selected = streamlit_menu(example=2)

# Verificar si 'selected' no es None antes de intentar usarlo
if selected:
    st.title(f"Modo {selected}")

if selected == "Ciencia de Datos":
    # Aquí creamos un nuevo submenú cuando se selecciona "Science"
    sub_selected = option_menu(
        menu_title=None,
        options=["simple", "complex"],
        # default_index=0,  # Removido para evitar el error
        orientation="horizontal",
    )

    if sub_selected == "simple":

        @st.cache_data()
        def load_dataframe():
            if os.path.exists('./dataset.csv'):
                return pd.read_csv('dataset.csv', index_col=None)
            return None

        df = load_dataframe()

        with st.sidebar:
            st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
            st.title("Automodelador")
            choice = st.radio("Navigation", ["Subir ficheros", "Análisis descriptivo simple","Modelaje"
                                              ])
            st.info(
                "Esta aplicación automatiza el proceso de creación de un modelo.")

        if choice == "Subir ficheros":
            st.title("Sube los ficheros")
            file1 = st.file_uploader("Choose a CSV file", type="csv")

            if file1:
                df = pd.read_csv(file1, index_col=None)
                df.to_csv('dataset.csv', index=None)
                st.dataframe(df)

        if choice == "Análisis descriptivo simple":
            st.title("Exploratory Data Analysis")

            if df is not None:
                # Encontrar columna con más datos missing
                missing_data = df.isnull().sum()
                max_missing_col = missing_data.idxmax()
                max_missing_percent = (missing_data.max() / df.shape[0]) * 100
                total_missing = missing_data.sum()

                with st.container():
                    st.markdown("<h1>Análisis de datos missing</h1>", unsafe_allow_html=True)
                    st.markdown(f"""
                                     <div class="card-container">
                                         <div class="card blue">
                                             <h2>Columna con más datos missing</h2>
                                             <p>{max_missing_col}</p>
                                             <p>{max_missing_percent:.2f}%</p>
                                         </div>
                                         <div class="card yellow">
                                             <h2>Total de datos missing en el dataset</h2>
                                             <p>{total_missing}</p>
                                         </div>
                                     </div>
                                     """, unsafe_allow_html=True)

                # Información sobre tipos de variables
                var_types = df.dtypes
                categorical_vars = sum(var_types == 'object')
                numerical_vars = sum(var_types != 'object')

                with st.container():
                    st.markdown("<h1>Información de variables</h1>", unsafe_allow_html=True)
                    st.markdown(f"""
                        <div class="card-container">
                            <div class="card blue">
                                <h2>Variables categóricas</h2>
                                <p>{categorical_vars}</p>
                            </div>
                            <div class="card green">
                                <h2>Variables numéricas</h2>
                                <p>{numerical_vars}</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                # Heatmap de correlaciones
                st.subheader("Heatmap de correlaciones")
                corr = df.corr()
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
                st.pyplot(plt)
            else:
                st.warning("Please upload a dataset first.")

        if choice == "Modelaje":

            if choice == "Modelaje":

                data = pd.read_csv('dataset.csv')

                # Conservar solo las columnas con datos numéricos
                data_numeric = data.select_dtypes(include=['number'])

                st.write('### Dataset solo con Variables Numéricas:')
                st.write(data_numeric.head())

                # Permitir al usuario seleccionar la variable objetivo
                if 'target' not in st.session_state:
                    st.session_state.target = data_numeric.columns[0]

                st.session_state.target = st.selectbox('Elige la variable objetivo', data_numeric.columns)
                target = st.session_state.target

                imputer = SimpleImputer(strategy='mean')
                data_numeric_imputed = pd.DataFrame(imputer.fit_transform(data_numeric), columns=data_numeric.columns)

                # Definir la variable objetivo y las variables predictoras
                X = data_numeric_imputed.drop(
                    columns=[target])  # Todas las columnas excepto la seleccionada como objetivo
                y = data_numeric_imputed[target]  # Solo la columna seleccionada como objetivo

                # Dividir los datos en conjuntos de entrenamiento y prueba
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Entrenar un modelo Random Forest
                rf_model = RandomForestRegressor(random_state=42)
                rf_model.fit(X_train, y_train)

                # Hacer predicciones en el conjunto de prueba
                y_pred = rf_model.predict(X_test)

                # Obtener el score del modelo (R² score) en el conjunto de pruebas
                score = rf_model.score(X_test, y_test)

                # Mostrar las predicciones
                st.write('### Predicciones usando un modelo random forest:')
                st.write(y_pred)

                # Mostrar el error cuadrático medio y otras métricas
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                st.write('### Métricas del Modelo:')
                st.table({
                    "Métrica": ["Error Cuadrático Medio (MSE)", "Coeficiente de Determinación (R²)",
                                "Score (R² score en el conjunto de pruebas)"],
                    "Valor": [f"{mse}", f"{r2}", f"{score}"]
                })

                st.write("### Descargar Modelo")
                model_filename = 'random_forest_model.joblib'
                joblib.dump(rf_model, model_filename)

                with open(model_filename, 'rb') as f:
                    bytes_data = f.read()
                st.download_button(
                    label="Descargar Modelo",
                    data=bytes_data,
                    file_name=model_filename,
                    mime='application/octet-stream',
                )

    if sub_selected == "complex":


        # Definimos las funciones para la carga de los datasets que el usuario quiere utilizar para entrenar el modelo y luego para generar predicciones:

        def load_dataframe():
            if os.path.exists('./dataset.csv'):
                return pd.read_csv('dataset.csv', index_col=None)
            return None


        def load_prediction_data():
            if os.path.exists('./predictions.csv'):
                return pd.read_csv('predictions.csv', index_col=None)
            return None


        # Nombramos ambos datasets
        df = load_dataframe()
        prediction_df = load_prediction_data()

        # Utilizaremos la librería streamlit para generar una interfaz para el usuario, donde se mostrará un menú lateral con las opciones que tiene disponibles, esto se define a continuación:
        # Creamos el título en el menú lateral y una imagen por efecto visual y definimos las opciónes que el usuario podrá escoger en el menú lateral:
        with st.sidebar:
            st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
            st.title("Automodelador")
            choice = st.radio("Navigation", ["Subir ficheros", "Análisis descriptivo", "Preprocesado", "Modelaje",
                                             "Generar nuevas predicciones","Insights"])
            st.info(
                "Esta aplicación automatiza el proceso de creación de un modelo de predicción de datos para datasets con variables de respuesta binarias o multiclase.")

        # Ahora, para cada una de las opciones del menú generamos el código de lo que verá el usuario según la opción seleccionada:

        # Carga de ficheros: indicamos al usuario que debe subir el dataset para entrenar el modelo y luego el dataset para generar predicciones:
        if choice == "Subir ficheros":
            st.title("Sube los ficheros")
            file = st.file_uploader("**Subir dataset para entrenar el modelo**")
            if file:
                df = pd.read_csv(file, index_col=None)
                df.to_csv('dataset.csv', index=None)
                st.dataframe(df)
            predictions_file = st.file_uploader("**Subir dataset cuyos outputs desconozcas para generar predicciones**")
            if predictions_file:
                prediction_df = pd.read_csv(predictions_file, index_col=None)
                prediction_df.to_csv('predictions.csv', index=None)
                st.dataframe(prediction_df)

        # Se genera un análisis descriptivo simple que se mostrará cuando el usuario seleccione analisis descriptivo:
        if choice == "Análisis descriptivo":
            st.title("Análisis exploratorio")
            if df is not None:
                st.subheader("Análisis descriptivo simple")
                descriptive = df.describe()

                with st.container():
                    st.markdown("Datos generales por categoría", unsafe_allow_html=True)
                    st.dataframe(descriptive)

                var_types = df.dtypes
                categorical_vars = sum(var_types == 'object')
                numerical_vars = sum(var_types != 'object')

                with st.container():
                    st.subheader("Información sobre el tipo de variables")
                    st.write("Cantidad de variables categóricas: ", categorical_vars)
                    st.write('Cantidad de variables numéricas: ', numerical_vars)

                st.subheader("Heatmap de correlaciones")
                corr = df.select_dtypes(exclude=['object', 'category']).corr()
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr)
                st.pyplot(plt)

                # Almacenando información en session state
                st.session_state.descriptive = descriptive
                st.session_state.categorical_vars = categorical_vars
                st.session_state.numerical_vars = numerical_vars

                # Y le damos la opción al usuario de generar adicionalmente un análisis descriptivo más detallado a través de un botón:
                st.subheader("Análisis descriptivo detallado")
                if st.button('Ejecutar Análisis descriptivo avanzado'):
                    profile_df = df.profile_report()
                    st_profile_report(profile_df)


            else:
                st.warning("Please upload a dataset first.")

        # Preprocesado

        # Creamos una columna que identifique cada dataset para luego concatenarlos y aplicar el preprocesado en conjunto
        df['type'] = 'train'
        prediction_df['type'] = 'pred'
        df1 = pd.concat([df, prediction_df], sort=False, ignore_index=False)

        X_train = None
        y_train = None
        X_test = None
        y_test = None
        data_pred = None

        best_model_type = None
        best_accuracy = 0
        comparacion = None
        # A continuación definimos funciones que se utilizarán para el preprocesado de los datos:

        # Función para convertir los outliers (definidos con un umbral de 3 desviaciones estandar) en datos nulos:
        def eliminar_outliers(dataset, umbral=3):
            dataset_outliers = dataset.copy()
            numeric_columns = dataset_outliers.select_dtypes(include=[np.float64, np.int64]).columns
            numeric_columns = [col for col in numeric_columns if col != target]
            for columna in dataset_outliers.columns:
                if dataset_outliers[columna].dtype in numeric_columns:
                    z_scores = np.abs((dataset_outliers[columna] - dataset_outliers[columna].mean()) / dataset_outliers[
                        columna].std())
                    dataset_outliers[columna] = np.where(z_scores > umbral, np.nan, dataset_outliers[columna])
            return dataset_outliers


        # Función para eliminar las columnas con más de un 25% de datos nulos
        def eliminar_columnas_con_nulos(dataset, porcentaje_limite=0.25):
            num_nulos_limite = len(dataset) * porcentaje_limite
            dataset_limpio = dataset.dropna(axis=1, thresh=num_nulos_limite)
            return dataset_limpio


        # Función para eliminar registros duplicados si los hubiera:
        def eliminar_registros_duplicados(df):
            df_sin_duplicados = df.drop_duplicates()
            return df_sin_duplicados


        # Función para convertir variables categoricas a tipo category
        def convertir_a_categoricas(dataset, valor_limite=10):
            columnas = dataset.columns
            for columna in columnas:
                if dataset[columna].nunique() < valor_limite:
                    dataset[columna] = dataset[columna].astype('category')
            return dataset


        # Función para imputar valores nulos: se usará la mediana para las variables numéricas y la moda para las variables categóricas
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


        # Función para normalizar las variables numéricas
        def normalizar_dataset(dataset):
            scaler = StandardScaler()
            variables_numericas = dataset.select_dtypes(include=['int64', 'float64']).drop(columns=['id', target],
                                                                                           errors='ignore')
            dataset[variables_numericas.columns] = scaler.fit_transform(variables_numericas)
            return dataset


        # Función para transformar las variables categóricas a numéricas con label encoding:
        label_mapping = {}


        def label_encoding(dataset):
            # Generamos el label encoding para las columnas tipo categoricas u objeto:
            for columna in dataset.select_dtypes(include=['category', 'object']).columns.difference(['type']):
                label_encoder = LabelEncoder()
                dataset[columna] = label_encoder.fit_transform(dataset[columna])
                # Vamos guardando para cada categoría su valor original y el encoded en un diccionario, que luego usaremos para mostrar las predicciones con los nombres originales de cada categoría
                label_mapping[columna] = dict(
                    zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

            # Nos interesa que la key del diccionario sea el valor encoded, por lo que invertimos el orden
            lab_map_rev = {outer_key: {str(v): str(k) for k, v in inner_dict.items()} for outer_key, inner_dict in
                           label_mapping.items()}
            st.session_state.lab_map_rev = lab_map_rev

            return dataset


        # Función para verificar si los datos de la variable objetivo están desbalanceados:
        def verificar_desbalanceo(dataset, variable_objetivo):
            conteo_clases = dataset[variable_objetivo].value_counts()
            porcentaje_menor_clase = (conteo_clases.min() / conteo_clases.sum()) * 100
            umbral_desbalanceo = 10  # se considera que hay desbalanceo si una clase representa menos del 10% del total
            desbalanceo = porcentaje_menor_clase < umbral_desbalanceo

            return desbalanceo


        # Función para realizar un resampling en caso de que haya desbalanceo en la variable objetivo
        def resample_variable(df, target_column, random_state=None):
            is_categorical = target_column.dtype == 'object' or target_column.nunique() <= 10
            resample_strategy = 'over' if is_categorical else 'under'

            if resample_strategy == 'over':
                df_resampled = pd.concat([resample(df[df[target_column.name] == value],
                                                   replace=True,
                                                   random_state=random_state)
                                          for value in df[target_column.name].unique()])
            elif resample_strategy == 'under':
                minority_class = target_column.value_counts().idxmin()
                df_resampled = pd.concat([resample(df[df[target_column.name] == minority_class],
                                                   replace=False,
                                                   random_state=random_state)] + [
                                             df[df[target_column.name] != minority_class]])
            # Esta función la usaremos despues del split train-test, por lo que debemos devolver los valores de X y Y por separado:
            X = df_resampled.drop(columns=[target_column.name])
            Y = df_resampled[target_column.name]

            st.session_state.resampling_done = True  # Agregamos una bandera que indica que el resampling se ha realizado

            return X, Y


        if choice == "Preprocesado":
            st.title("Preprocesado")
            if df is not None:
                # Se le pide al usuario que elija el nombre de la variable objetivo de la lista de todas las variables y la elegida se guarda como target:
                st.session_state.target = st.selectbox('Elige la variable objetivo', df.columns)
                target = st.session_state.target

                # Cuando el usuario hace clic en el botón "Preprocesado", se ejecutan todas las funciones definidas anteriormente y se van mostrando los resultados al usuario:
                if st.button('Ejecutar preprocesado'):
                    st.subheader("Detección y eliminación de Outliers")
                    df_sin_outliers = eliminar_outliers(df1)
                    st.dataframe(df_sin_outliers)

                    st.subheader("Tratamiento de nulos y missings")
                    df_limpio = eliminar_columnas_con_nulos(df_sin_outliers, porcentaje_limite=0.25)
                    st.dataframe(df_limpio)

                    st.subheader('Eliminar registros duplicados')
                    df_sin_duplicados = eliminar_registros_duplicados(df_limpio)
                    st.dataframe(df_sin_duplicados)

                    st.subheader("Categorización")
                    df_categorico = convertir_a_categoricas(df_sin_duplicados, valor_limite=10)
                    st.dataframe(df_categorico.dtypes.astype(str))

                    st.subheader("Imputación de nulos")
                    df_imputado = imputar_valores_nulos(df_categorico)
                    st.dataframe(df_imputado)

                    st.subheader("Normalización")
                    df_normalizado = normalizar_dataset(df_imputado)
                    st.write("Dataset normalizado:")
                    st.dataframe(df_normalizado)

                    st.subheader("Encoding")
                    df_encoded = label_encoding(df_normalizado)
                    st.dataframe(df_encoded)

                    # Antes de revisar el desbalanceo y de hacer resampling, separamos de nuevo los dos datasets que habíamos unido al inicio y con el dataset de training, hacemos el split train-test:
                    data_train = df_encoded[df_encoded['type'] == 'train']
                    data_train.drop('type', axis=1, inplace=True)
                    st.session_state.data_pred = df_encoded[df_encoded['type'] == 'pred']
                    st.session_state.data_pred.drop('type', axis=1, inplace=True)
                    st.session_state.data_pred.drop(target, axis=1, inplace=True)
                    X_train1, st.session_state.X_test, y_train1, st.session_state.y_test = train_test_split(
                        data_train.drop(target, axis=1),
                        data_train[target],
                        test_size=0.2,
                        random_state=1234,
                        stratify=data_train[target])
                    # Ahora revisamos si hay desbalanceo únicamente sobre los datos que usaremos para entrenar al modelo (X_train)
                    st.subheader("Checkear desbalanceo")
                    desbalanceo = verificar_desbalanceo(data_train, target)
                    if desbalanceo:
                        st.warning(
                            "El dataset sufre de desbalanceo en la variable objetivo. Procedemos a realizar un resampling.")
                        st.session_state.X_train, st.session_state.y_train = resample_variable(
                            pd.concat([X_train1, y_train1], axis=1), y_train1)
                        st.dataframe(pd.concat([st.session_state.X_train, st.session_state.y_train], axis=1))
                    else:
                        st.success("El dataset no sufre de desbalanceo en la variable objetivo.")
                        st.session_state.X_train = X_train1
                        st.session_state.y_train = y_train1
            else:
                st.warning("Please upload a dataset first.")


        # Función para evaluación de los modelos:
        def eval_model(y_real, y_pred, n_classes):
            # Calcular las métricas
            confusion = confusion_matrix(y_real, y_pred)
            accuracy = accuracy_score(y_real, y_pred)
            precision = round(precision_score(y_real, y_pred, average='macro'), 3)
            recall = round(recall_score(y_real, y_pred, average='macro'), 3)
            f1 = round(f1_score(y_real, y_pred, average='macro'), 3)

            # Mostrar los resultados de la evaluación del modelo
            st.write('Confusion Matrix:')
            st.write(confusion)
            st.write('Accuracy:', accuracy)
            st.write('Precision:', precision)
            st.write('Recall:', recall)
            st.write('F1 Score:', f1)
            # En caso de que sea una clasificación binomial, además generaremos la curva ROC
            if n_classes == 2:
                false_positive_rate, recall, thresholds = roc_curve(y_real, y_pred)
                roc_auc = auc(false_positive_rate, recall)
                st.write('AUC:', roc_auc)
                # ROC curve
                plt.plot(false_positive_rate, recall, 'b')
                plt.plot([0, 1], [0, 1], 'r--')
                plt.title('ROC Curve')
                st.pyplot(plt)


        # Ahora generamos el código para cuando el usuario seleccione la opción de modelado
        predictions = None
        download_button_pressed = False
        best_model = None
        if choice == "Modelaje":
            st.title("Modelado y predicciones")
            if df is not None:
                # Nos aseguramos de que X_train esté en session_state lo que nos indicaría que el preprocesado ya se realizó, sino se indicaría al usuario que debe completarlo primero.
                if 'X_train' in st.session_state:
                    X_train, y_train = st.session_state.X_train, st.session_state.y_train
                    X_test, y_test = st.session_state.X_test, st.session_state.y_test
                    data_pred = st.session_state.data_pred
                    target = st.session_state.target

                    # Preguntamos al usuario si quiere tuneo de hiperparámetros:
                    optimize_hyperparams = st.checkbox("Modelo con hiperparámetros optimizados")
                    default_model = None
                    best_accuracy = 0
                    best_model = None

                    if st.button('Run Modelling'):
                        # Creamos el modelo de Random Forest:
                        modelo_rf = RandomForestClassifier(
                            n_estimators=10,
                            criterion='gini',
                            max_depth=None,
                            max_features='sqrt',
                            oob_score=False,
                            n_jobs=-1,
                            random_state=123
                        )
                        modelo_rf.fit(X_train, y_train)

                        # Creamos el modelo de Regresion logistica:
                        reg_log = LogisticRegression(max_iter=10000)
                        reg_log.fit(X_train, y_train)

                        # Creamos el modelo de XGBoost
                        xgb = XGBClassifier(n_jobs=-1, n_estimators=30, random_state=1234)
                        xgb.fit(X_train, y_train)

                        # Creamos una tabla comparativa con los resultados de los 3 modelos:
                        modelos = [modelo_rf, reg_log, xgb]
                        results = []
                        for model in modelos:
                            # Primero verificamos si el output es multiclase y en ese caso no se ejecutaría la regresión logística, solo los otros dos modelos
                            unique_classes = y_train.unique()
                            if len(unique_classes) > 2 and model.__class__.__name__ == "LogisticRegression":
                                st.warning(
                                    "La variable objetivo tiene más de 2 clases, por lo que no se ejecutará el modelo de regresión logística")
                            else:
                                pred = model.predict(X_test)
                                accuracy = round(accuracy_score(y_test, pred) * 100, 2)
                                precision = round(precision_score(y_test, pred, average='macro'), 3)
                                recall = round(recall_score(y_test, pred, average='macro'), 4)
                                f1 = round(f1_score(y_test, pred, average='macro'), 4)
                                results.append({'Model': model.__class__.__name__, 'Accuracy': accuracy,
                                                'Precision': precision, 'Recall': recall, 'F1': f1})
                        comparacion = pd.DataFrame(results)
                        st.subheader("Comparación entre los modelos")
                        st.write(comparacion)

                        # Seleccionamos el mejor modelo basado en accuracy y se lo indicamos al usuario:
                        comparacion_sorted = comparacion.sort_values(by='Accuracy', ascending=False)
                        best_model_type = comparacion_sorted.iloc[0]['Model']
                        best_accuracy = comparacion_sorted.iloc[0]['Accuracy']
                        st.write("**Mejor modelo basado en accuracy:**", best_model_type)
                        st.subheader('Resultados del mejor modelo:')

                        # Redefinimos default_model basados en el mejor modelo
                        if best_model_type == "RandomForestClassifier":
                            default_model = modelo_rf

                            # Evaluacion del modelo con la función creada antes:
                            pred1 = modelo_rf.predict(X_test)
                            eval_model(y_test, pred1, len(unique_classes))

                            # Lista de variables más importantes:
                            feature_importances1 = modelo_rf.feature_importances_
                            imp = {}
                            for i in range(len(X_train.columns)):
                                imp[X_train.columns[i]] = [feature_importances1[i]]
                            var_imp1 = pd.DataFrame.from_dict(imp,
                                                              orient="index",
                                                              columns=["Importance"]
                                                              ).sort_values("Importance",
                                                                            ascending=False).head(
                                20).style.background_gradient()
                            st.write("**Variables más importantes:**")
                            st.write(var_imp1)

                        elif best_model_type == "LogisticRegression":
                            default_model = reg_log
                            # Evaluacion del modelo:
                            pred2 = reg_log.predict(X_test)
                            eval_model(y_test, pred2, len(unique_classes))

                            # Lista de variables más importantes:
                            feature_importances2 = abs(reg_log.coef_[0])
                            imp2 = {}
                            for i in range(len(X_train.columns)):
                                imp2[X_train.columns[i]] = [feature_importances2[i]]
                            var_imp2 = pd.DataFrame.from_dict(imp2,
                                                              orient="index",
                                                              columns=["Importance"]
                                                              ).sort_values("Importance",
                                                                            ascending=False).head(
                                20).style.background_gradient()
                            st.write("**Variables más importantes:**")
                            st.write(var_imp2)

                        elif best_model_type == "XGBClassifier":
                            default_model = xgb
                            # Evaluacion del modelo:
                            pred3 = xgb.predict(X_test)
                            eval_model(y_test, pred3, len(unique_classes))

                            # Lista de variables más importantes:
                            feature_importances3 = xgb.feature_importances_
                            imp3 = {}
                            for i in range(len(X_train.columns)):
                                imp3[X_train.columns[i]] = [feature_importances3[i]]
                            var_imp3 = pd.DataFrame.from_dict(imp3,
                                                              orient="index",
                                                              columns=["Importance"]
                                                              ).sort_values("Importance",
                                                                            ascending=False).head(
                                20).style.background_gradient()
                            st.write("**Variables más importantes:**")
                            st.write(var_imp3)

                        # Hacemos tuneo de hiperparámetros para el mejor modelo en caso de que el usuario lo haya seleccionado:
                        if optimize_hyperparams:
                            st.subheader('Resultados con optimización de parámetros:')
                            if best_model_type == "RandomForestClassifier":
                                param_grid_rf = {
                                    'n_estimators': [10, 50, 100],
                                    'max_depth': [None, 20],
                                    'min_samples_split': [2, 10],
                                    'min_samples_leaf': [1, 4],
                                    'max_features': ['auto', 'sqrt'],
                                    'bootstrap': [True, False]
                                }
                                random_search_rf = RandomizedSearchCV(modelo_rf, param_distributions=param_grid_rf,
                                                                      n_iter=50, cv=5,
                                                                      n_jobs=-1, random_state=123)
                                random_search_rf.fit(X_train, y_train)
                                best_hyperparameters_rf = random_search_rf.best_params_
                                best_model = RandomForestClassifier(**best_hyperparameters_rf, random_state=123)
                                best_model.fit(X_train, y_train)

                            elif best_model_type == "LogisticRegression":
                                param_grid_lr = {
                                    'C': [0.01, 0.1, 1, 10, 100],
                                    'penalty': ['l1', 'l2'],
                                    'solver': ['liblinear']
                                }
                                grid_search_lr = GridSearchCV(reg_log, param_grid=param_grid_lr, cv=5, n_jobs=-1)
                                grid_search_lr.fit(X_train, y_train)

                                best_hyperparameters_lr = grid_search_lr.best_params_

                                best_model = LogisticRegression(**best_hyperparameters_lr, max_iter=10000,
                                                                random_state=123)
                                best_model.fit(X_train, y_train)

                            elif best_model_type == "XGBClassifier":
                                param_grid_xgb = {
                                    'n_estimators': [10, 50, 100],
                                    'max_depth': [3, 7],
                                    'learning_rate': [0.01, 0.1, 0.2],
                                    'subsample': [0.8, 1.0]
                                }
                                grid_search_xgb = GridSearchCV(xgb, param_grid=param_grid_xgb, cv=5, n_jobs=-1)
                                grid_search_xgb.fit(X_train, y_train)
                                best_hyperparameters_lr = grid_search_xgb.best_params_

                                best_model = XGBClassifier(**best_hyperparameters_lr, random_state=123)
                                best_model.fit(X_train, y_train)

                            pred4 = best_model.predict(X_test)
                            eval_model(y_test, pred4, len(unique_classes))

                        if not optimize_hyperparams:
                            best_model = default_model

                        st.session_state.best_model = best_model

                        # predecir los labels para el dataset de predictores (los datos que desconoce el usuario)
                        label_mapping = st.session_state.lab_map_rev
                        pred_final = best_model.predict(data_pred)
                        predictions = data_pred.copy()
                        predictions[target] = pred_final
                        # Reemplazamos los valores codificados del label encoding por los valores originales para mostrarlos al usuario
                        for column in label_mapping:
                            predictions[column] = predictions[column].astype(str).map(label_mapping[column])

                        st.subheader("Predicciones:")
                        st.write(predictions)

                        # Guardamos las predicciones en session state
                        st.session_state.predictions = predictions

                        # Incluimos un boton para descargar las predicciones en Excel:
                        if 'predictions' in st.session_state:
                            excel_file = BytesIO()
                            # Guardamos las predicciones en el objeto creado
                            st.session_state.predictions.to_excel(excel_file, index=False)
                            # Creamos un link para que el usuario seleccione donde guardar
                            if st.download_button(
                                    label="Descargar predicciones",
                                    data=excel_file.getvalue(),
                                    file_name="predictions.xlsx",
                                    key="predictions_download",
                                    help="Haz clic para descargar las predicciones en formato XLSX"
                            ):
                                st.success("Las predicciones se descargaron correctamente")

                else:
                    st.warning("Antes del modelado, ejecute el preprocesado")
            else:
                st.warning('Antes de ejecutar el modelado, debes cargar el dataset')

            st.session_state.best_model_type = best_model_type
            st.session_state.best_accuracy = best_accuracy
            st.session_state.comparacion = comparacion




        # Creamos la función para cargar el nuevo dataset para generar nuevas predicciones:
        def load_pred2():
            if os.path.exists('./predictions2.csv'):
                return pd.read_csv('predictions2.csv', index_col=None)
            return None


        to_pred = load_pred2()
        data_pred2 = None

        # Cuando el usuario selecciona del menú lateral "Generar nuevas predicciones", le pedimos que cargue un nuevo dataset para generar las predicciones del mismo
        if choice == "Generar nuevas predicciones":
            if 'predictions' in st.session_state:
                st.title("Generar nuevas predicciones")
                target = st.session_state.target
                best_model = st.session_state.best_model
                predictions_file2 = st.file_uploader(
                    "**Subir un dataset cuyos outputs desconozcas para generar predicciones**")
                if predictions_file2:
                    to_pred = pd.read_csv(predictions_file2, index_col=None)
                    to_pred.to_csv('predictions2.csv', index=None)
                    st.write('Dataset para generar predicciones:')
                    st.dataframe(to_pred)

                    # ejecutamos preprocesado nuevamente para limpiar los datos a predecir
                    to_pred['type'] = "pred2"
                    df2 = pd.concat([df, to_pred], sort=False, ignore_index=False)
                    df_limpio2 = eliminar_columnas_con_nulos(df2, porcentaje_limite=0.5)
                    df_categorico2 = convertir_a_categoricas(df_limpio2, valor_limite=10)
                    df_imputado2 = imputar_valores_nulos(df_categorico2)
                    df_normalizado2 = normalizar_dataset(df_imputado2)
                    df_clean2 = label_encoding(df_normalizado2)

                    # Volvemos a separar el dataset de predicciones ya limpio
                    st.session_state.data_pred2 = df_clean2[df_clean2['type'] == 'pred2']
                    st.session_state.data_pred2.drop('type', axis=1, inplace=True)
                    st.session_state.data_pred2.drop(target, axis=1, inplace=True)

                    st.write("Dataset limpio y transformado:")
                    st.dataframe(st.session_state.data_pred2)

                # Generar predicciones:
                if st.button('Generar predicciones'):
                    label_mapping = st.session_state.lab_map_rev
                    pred_nueva = best_model.predict(data_pred2)
                    predict2 = st.session_state.data_pred2.copy()
                    predict2[target] = pred_nueva
                    for column in label_mapping:
                        predict2[column] = predict2[column].astype(str).map(label_mapping[column])

                    st.subheader("Predicciones:")
                    st.write(predict2)

                    # Guardamos predicciones en session state
                    st.session_state.predict2 = predict2

                    # Boton para descargar las predicciones en Excel
                    if 'predict2' in st.session_state:
                        excel_file = BytesIO()
                        # Guardamos las predicciones en el objeto creado
                        st.session_state.predict2.to_excel(excel_file, index=False)
                        # Creamos un link para que el usuario seleccione donde guardar
                        if st.download_button(
                                label="Descargar predicciones",
                                data=excel_file.getvalue(),
                                file_name="predict2.xlsx",
                                key="predictions_download",
                                help="Haz clic para descargar las predicciones en formato XLSX"
                        ):
                            st.success("Las predicciones se descargaron correctamente")

            else:
                st.warning('Para generar nuevas predicciones, debes ejecutar el modelado')


        if choice == "Insights":
            def print_insights(best_model_type, best_accuracy, comparacion):
                best_model_details = comparacion.loc[comparacion['Model'] == best_model_type].iloc[0]

                with open('insights.txt', 'w') as file:
                    file.write(f"El mejor modelo ha sido el {best_model_type}.\n")
                    file.write(f"- El accuracy es de {best_accuracy / 100:.2f}\n")
                    file.write(f"- La precisión es de {best_model_details['Precision']:.3f}\n")
                    file.write(f"- El recall es de {best_model_details['Recall']:.3f}\n")
                    file.write(f"- El puntaje F1 es de {best_model_details['F1']:.3f}\n")

                    # Incluyendo datos del análisis descriptivo
                    if 'descriptive' in st.session_state:

                        file.write(f"- Resumen estadístico:\n{st.session_state.descriptive}\n")


            print_insights(st.session_state.best_model_type, st.session_state.best_accuracy,
                           st.session_state.comparacion)

            if __name__ == "__main__":
                def cargar_documentos(archivos):
                    texto_total = ""
                    for ruta_archivo in archivos:
                        extension_archivo = os.path.splitext(ruta_archivo)[1]
                        with open(ruta_archivo, 'rb') as file:
                            if extension_archivo == ".pdf":
                                lector_pdf = PyPDF2.PdfReader(file)
                                texto = ""
                                for pagina in lector_pdf.pages:
                                    texto += pagina.extract_text()
                                texto_total += texto
                            elif extension_archivo == ".txt":
                                texto = file.read().decode("utf-8")
                                texto_total += texto
                            else:
                                st.warning('Proporciona un archivo en formato PDF o TXT.', icon="⚠️")
                    return texto_total


                def crear_recuperador(embeddings, fragmentos):
                    try:
                        almacen_vector = FAISS.from_texts(fragmentos, embeddings)
                    except (IndexError, ValueError) as e:
                        st.error(f"Hubo un problema creando el almacenamiento de vectores: {e}")
                        return
                    recuperador = almacen_vector.as_retriever(k=5)
                    return recuperador


                @st.cache_resource
                def dividir_textos(texto, tamano_fragmento, solapamiento):
                    divisor_texto = RecursiveCharacterTextSplitter(chunk_size=tamano_fragmento,
                                                                   chunk_overlap=solapamiento)
                    fragmentos = divisor_texto.split_text(texto)
                    if not fragmentos:
                        return None
                    return fragmentos


                def principal():
                    st.title("Analisis del Modelo con inteligencia artificial")

                    if 'llave_api_openai' not in st.session_state:
                        llave_api_openai = st.text_input(
                            'Introduce tu llave API de OpenAI',
                            value="", placeholder="Introduce tu llave API que inicia con sk-", type="password")
                        if llave_api_openai:
                            st.session_state.llave_api_openai = llave_api_openai
                            os.environ["OPENAI_API_KEY"] = llave_api_openai
                        else:
                            return

                    archivos_subidos = ["insights.txt"]

                    # Aquí se elimina la necesidad de verificar una carga manual de archivos
                    texto_cargado = cargar_documentos(archivos_subidos)



                    fragmentos = dividir_textos(texto_cargado, tamano_fragmento=1000, solapamiento=0)

                    if fragmentos is None:
                        st.error("La división del texto no fue exitosa.")
                        st.stop()


                    embeddings = OpenAIEmbeddings()
                    recuperador = crear_recuperador(embeddings, fragmentos)

                    manejador_callback = StreamingStdOutCallbackHandler()
                    gestor_callback = CallbackManager([manejador_callback])

                    chat_openai = ChatOpenAI(streaming=True, callback_manager=gestor_callback, verbose=True,
                                             temperature=0)
                    sistema_qa = RetrievalQA.from_chain_type(llm=chat_openai, retriever=recuperador,
                                                             chain_type="stuff",
                                                             verbose=True)

                    pregunta_usuario = ""
                    if st.button("Generar Insights"):
                        pregunta_usuario = "Solo tienes estos datos, tienes que intentar DarME INSIGHTS DE UN POSIBLE MODELO  O POSIBLES MEJORAS, además se ha realizado un relleno de datos faltantes con la media"

                    if pregunta_usuario:
                        respuesta = sistema_qa.run(pregunta_usuario)
                        st.write(respuesta)


                principal()


if selected == "Análisis de datos":

    sub_selected = option_menu(
        menu_title=None,
        options=["Analizar Dataframe", "Analizar PDF"],
        # default_index=0,  # Removido para evitar el error
        orientation="horizontal",
    )
    if sub_selected == "Analizar Dataframe":
        # Inicializa API_KEY en el session_state si no existe
        if "API_KEY" not in st.session_state:
            st.session_state.API_KEY = ""

        st.sidebar.header("Configuración de API Key")
        api_key_input = st.sidebar.text_input("Ingrese su API Key de OpenAI:", type="password")
        refresh_api_button = st.sidebar.button("Refrescar API Key")

        if refresh_api_button:
            if api_key_input:
                st.sidebar.write("API Key actualizada con éxito")
                st.session_state.API_KEY = api_key_input
            else:
                st.sidebar.write("Por favor, ingrese una API Key válida.")

        if not st.session_state.API_KEY:
            st.sidebar.write("No se ha proporcionado una API Key válida.")
            # Podrías optar por detener la ejecución aquí si no hay una API Key válida
            # return

        temperature = st.slider("Temperatura", 0.0, 1.0)
        model_name = st.selectbox("Modelo de OpenAI", ["gpt-3", "curie", "davinci"])

        llm = OpenAI(api_token=st.session_state.API_KEY, temperature=temperature, model_name=model_name)
        pandas_ai = PandasAI(llm)
        uploaded_file = st.file_uploader("Suba un archivo CSV para análisis", type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.write(df.head(3))

        prompt = st.text_area("Ingrese su prompt:")

        if st.button("Generar"):
            if prompt:
                full_prompt = f"responde en español, {prompt}"
                # Ahora pasamos full_prompt al método que genera la respuesta
                st.write(pandas_ai.run(df, prompt=full_prompt))

            else:
                st.warning("Por favor, ingrese un prompt.")


    if sub_selected == "Analizar PDF":
        if __name__ == "__main__":
            def cargar_documentos(archivos):
                texto_total = ""
                for ruta_archivo in archivos:
                    extension_archivo = os.path.splitext(ruta_archivo.name)[1]
                    if extension_archivo == ".pdf":
                        lector_pdf = PyPDF2.PdfReader(ruta_archivo)
                        texto = ""
                        for pagina in lector_pdf.pages:
                            texto += pagina.extract_text()
                        texto_total += texto
                    elif extension_archivo == ".txt":
                        lector_txt = StringIO(ruta_archivo.getvalue().decode("utf-8"))
                        texto = lector_txt.read()
                        texto_total += texto
                    else:
                        st.warning('Proporciona un archivo en formato PDF o TXT.', icon="⚠️")
                return texto_total


            def crear_recuperador(embeddings, fragmentos):
                try:
                    almacen_vector = FAISS.from_texts(fragmentos, embeddings)
                except (IndexError, ValueError) as e:
                    st.error(f"Hubo un problema creando el almacenamiento de vectores: {e}")
                    return
                recuperador = almacen_vector.as_retriever(k=5)
                return recuperador


            @st.cache_resource
            def dividir_textos(texto, tamano_fragmento, solapamiento):
                divisor_texto = RecursiveCharacterTextSplitter(chunk_size=tamano_fragmento, chunk_overlap=solapamiento)
                fragmentos = divisor_texto.split_text(texto)
                if not fragmentos:
                    return None
                return fragmentos


            def principal():
                st.title("Herramienta de Análisis de PDFs")

                if 'llave_api_openai' not in st.session_state:
                    llave_api_openai = st.text_input(
                        'Introduce tu llave API de OpenAI',
                        value="", placeholder="Introduce tu llave API que inicia con sk-",type="password")
                    if llave_api_openai:
                        st.session_state.llave_api_openai = llave_api_openai
                        os.environ["OPENAI_API_KEY"] = llave_api_openai
                    else:
                        return

                archivos_subidos = st.file_uploader("Carga un documento PDF o TXT", type=["pdf", "txt"],
                                                    accept_multiple_files=True)

                if archivos_subidos:
                    if 'ultimos_archivos_subidos' not in st.session_state or st.session_state.ultimos_archivos_subidos != archivos_subidos:
                        st.session_state.ultimos_archivos_subidos = archivos_subidos

                    texto_cargado = cargar_documentos(archivos_subidos)
                    st.write("Los documentos han sido cargados y procesados.")

                    fragmentos = dividir_textos(texto_cargado, tamano_fragmento=1000, solapamiento=0)
                    if fragmentos is None:
                        st.error("La división del texto no fue exitosa.")
                        st.stop()
                    else:
                        st.info("Procesando documento ...")

                    embeddings = OpenAIEmbeddings()
                    recuperador = crear_recuperador(embeddings, fragmentos)

                    manejador_callback = StreamingStdOutCallbackHandler()
                    gestor_callback = CallbackManager([manejador_callback])

                    chat_openai = ChatOpenAI(streaming=True, callback_manager=gestor_callback, verbose=True,
                                             temperature=0)
                    sistema_qa = RetrievalQA.from_chain_type(llm=chat_openai, retriever=recuperador, chain_type="stuff",
                                                             verbose=True)

                    st.write("Preparado")

                    pregunta_usuario = st.text_input("Escribe tu pregunta aquí:")
                    if pregunta_usuario:
                        respuesta = sistema_qa.run(pregunta_usuario)
                        st.write(respuesta)


            principal()