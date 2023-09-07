from operator import index
import streamlit as st
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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from io import BytesIO


# Helper function to load the DataFrame using caching
@st.cache_data()
def load_dataframe():
    if os.path.exists('./dataset.csv'):
        return pd.read_csv('dataset.csv', index_col=None)
    return None

@st.cache_data()
def load_prediction_data():
    if os.path.exists('./predictions.csv'):
        return pd.read_csv('predictions.csv', index_col=None)
    return None

# Load the DataFrame
df = load_dataframe()
prediction_df = load_prediction_data()

with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Automodeladorv.1")
    choice = st.radio("Navigation", ["Subir ficheros", "Análisis descriptivo", "Preprocesado", "Modelaje", "Descargar modelo"])
    st.info("Esta aplicación automatiza el proceso de creación de un modelo de predicción de datos para datasets con variables de respuesta binarias.")

def run_ficheros():
    st.header("Sube los ficheros")
    file = st.file_uploader("**Subir dataset para entrenar el modelo**")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)
    predictions_file = st.file_uploader("**Subir dataset cuyos outputs desconoscas para generar predicciones**")
    if predictions_file:
        prediction_df = pd.read_csv(predictions_file, index_col=None)
        prediction_df.to_csv('predictions.csv', index=None)
        st.dataframe(prediction_df)
    result = {df, prediction_df}  # Store results or data for Section 1
    return result

def


if choice == "Subir ficheros":
    st.title("Sube los ficheros")
    file = st.file_uploader("**Subir dataset para entrenar el modelo**")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)
    predictions_file = st.file_uploader("**Subir dataset cuyos outputs desconoscas para generar predicciones**")
    if predictions_file:
        prediction_df = pd.read_csv(predictions_file, index_col=None)
        prediction_df.to_csv('predictions.csv', index=None)
        st.dataframe(prediction_df)

if choice == "Análisis descriptivo":
    st.title("Exploratory Data Analysis")
    if df is not None:
        profile_df = df.profile_report()
        st_profile_report(profile_df)
    else:
        st.warning("Please upload a dataset first.")


# Preprocesado

#Creamos una columna que identifique cada dataset para luego concatenarlos
df['type'] = 'train'
prediction_df['type'] = 'pred'
df = pd.concat([df, prediction_df], sort = False, ignore_index=False)

df_clean = None

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
    for columna in dataset.select_dtypes(include='category').columns.difference(['type']):
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
        target = st.selectbox('Elige la variable objetivo', df.columns)
        if st.button('Ejecutar preprocesado'):
            st.subheader("Tratamiento de nulos y missings")
            df_limpio = eliminar_columnas_con_nulos(df, porcentaje_limite=0.2)
            st.dataframe(df_limpio)

            st.subheader("Categorización")
            df_categorico = convertir_a_categoricas(df_limpio, valor_limite=10)
            st.dataframe(df_categorico.dtypes.astype(str))

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
                st.write("Dataset no ha sido normalizado.")
                df_normalizado = df_imputado
            else:
                st.warning("Respuesta no válida. Por favor, selecciona 'Sí' o 'No'.")

            st.subheader("Encoding")
            columnas_orden_jerarquico = obtener_columnas_orden_jerarquico(df_normalizado)
            st.session_state.df_clean = label_encoding(df_normalizado)
            if columnas_orden_jerarquico:
                st.session_state.df_clean[columnas_orden_jerarquico] = df_clean[columnas_orden_jerarquico].astype('category')
            st.dataframe(st.session_state.df_clean)

            st.subheader("Checkear desbalanceo")
            desbalanceo = verificar_desbalanceo(st.session_state.df_clean, target)
            if desbalanceo:
                st.warning("El dataset sufre de desbalanceo en la variable objetivo.")
            else:
                st.success("El dataset no sufre de desbalanceo en la variable objetivo.")
    else:
        st.warning("Please upload a dataset first.")

# Función para evaluación de los modelos:

def eval_model(y_real, y_pred):
    # Calcular las métricas
    confusion = confusion_matrix(y_real, y_pred)
    accuracy = accuracy_score(y_real, y_pred)
    precision = round(precision_score(y_real, y_pred, average='macro'), 3)
    recall = round(recall_score(y_real, y_pred, average='macro'), 3)
    f1 = round(f1_score(y_real, y_pred, average='macro'), 3)
    false_positive_rate, recall, thresholds = roc_curve(y_real, y_pred)
    roc_auc = auc(false_positive_rate, recall)

    # Mostrar los resultados de la evaluación del modelo
    st.write('Confusion Matrix:')
    st.write(confusion)
    st.write('Accuracy:', accuracy)
    st.write('Precision:', precision)
    st.write('Recall:', recall)
    st.write('F1 Score:', f1)
    st.write('AUC:', roc_auc)

    # ROC curve
    plt.plot(false_positive_rate, recall, 'b')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title('ROC Curve')
    st.pyplot(plt)


predictions = None
download_button_pressed = False
if choice == "Modelaje":
    st.title("Modelado y predicciones")
    if df is not None:
        if 'df_clean' in st.session_state:
            df_clean = st.session_state.df_clean
            target = st.selectbox('Choose the Target Column', df.columns)

            # Preguntamos al usuario si quiere tuneo de hiperparámetros:
            optimize_hyperparams = st.checkbox("Modelo con hiperparámetros optimizados")

            best_model = None
            default_model = None
            best_accuracy = 0

            if st.button('Run Modelling'):
                # Separar los dataset y eliminar la columna identificadora:

                data_train = df_clean[df_clean['type'] == 'train']
                data_train.drop('type', axis=1, inplace=True)
                data_pred = df_clean[df_clean['type'] == 'pred']
                data_pred.drop('type', axis=1, inplace=True)
                data_pred.drop(target, axis=1, inplace=True)
                X_train, X_test, y_train, y_test = train_test_split(data_train.drop(target, axis=1),
                                                                    data_train[target],
                                                                    test_size=0.2,
                                                                    random_state=1234,
                                                                    stratify=data_train[target])
                # Modelo de Random Forest:
                # Creación del modelo
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

                # Modelo Regresion logistica:
                # Creación del modelo:
                reg_log = LogisticRegression(max_iter=10000)
                reg_log.fit(X_train, y_train)

                # Comparación de los dos modelos:
                modelos = [modelo_rf, reg_log]
                results = []
                for model in modelos:
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

                # Selección del mejor modelo:
                comparacion_sorted = comparacion.sort_values(by='Accuracy', ascending=False)
                best_model_type = comparacion_sorted.iloc[0]['Model']
                best_accuracy = comparacion_sorted.iloc[0]['Accuracy']
                st.write("**Mejor modelo basado en accuracy:**", best_model_type)
                st.subheader('Resultados del modelo:')

                # Redefinimos default_model basados en el mejor modelo
                if best_model_type == "RandomForestClassifier":
                    default_model = modelo_rf
                    # Evaluacion del modelo

                    pred1 = modelo_rf.predict(X_test)
                    eval_model(y_test, pred1)

                    # Lista de variables más importantes:
                    feature_importances1 = modelo_rf.feature_importances_
                    imp = {}
                    for i in range(len(X_train.columns)):
                        imp[X_train.columns[i]] = [feature_importances1[i]]
                    var_imp1 = pd.DataFrame.from_dict(imp,
                                                      orient="index",
                                                      columns=["Importance"]
                                                      ).sort_values("Importance",
                                                                    ascending=False).head(20).style.background_gradient()
                    st.write("**Variables más importantes:**")
                    st.write(var_imp1)
                elif best_model_type == "LogisticRegression":
                    default_model = reg_log
                    # Evaluacion del modelo:
                    pred2 = reg_log.predict(X_test)
                    eval_model(y_test, pred2)

                    # Lista de variables más importantes:
                    feature_importances2 = abs(reg_log.coef_[0])
                    imp2 = {}
                    for i in range(len(X_train.columns)):
                        imp2[X_train.columns[i]] = [feature_importances2[i]]
                    var_imp2 = pd.DataFrame.from_dict(imp2,
                                                      orient="index",
                                                      columns=["Importance"]
                                                      ).sort_values("Importance",
                                                                    ascending=False).head(20).style.background_gradient()
                    st.write("**Variables más importantes:**")
                    st.write(var_imp2)

                # Hacemos tuneo de hiperparámetros para el mejor modelo:
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
                        random_search_rf = RandomizedSearchCV(modelo_rf, param_distributions=param_grid_rf, n_iter=50, cv=5,
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

                        # Create and train the final Logistic Regression model with the best hyperparameters
                        best_model = LogisticRegression(**best_hyperparameters_lr, max_iter=10000, random_state=123)
                        best_model.fit(X_train, y_train)

                    pred3 = best_model.predict(X_test)
                    eval_model(y_test, pred3)

                if not optimize_hyperparams:
                    best_model = default_model

                # predecir los labels para el dataset de predictores (los datos que desconoce el usuario)
                pred_final = best_model.predict(data_pred)
                predictions = data_pred.copy()
                predictions['Predictions'] = pred_final
                st.subheader("Predicciones:")
                st.write(predictions)

                # Store predictions in session state
                st.session_state.predictions = predictions

                # Boton para descargar las predicciones en Excel
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

if choice == "Descargar modelo":
    if df is not None:
        with open('best_model.pkl', 'rb') as f:
            st.download_button('Download Model', f, file_name="best_model.pkl")
    else:
        st.warning("Please upload a dataset first.")
