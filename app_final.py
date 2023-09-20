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


#Definimos las funciones para la carga de los datasets que el usuario quiere utilizar para entrenar el modelo y luego para generar predicciones:

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

#Utilizaremos la librería streamlit para generar una interfaz para el usuario, donde se mostrará un menú lateral con las opciones que tiene disponibles, esto se define a continuación:
#Creamos el título en el menú lateral y una imagen por efecto visual y definimos las opciónes que el usuario podrá escoger en el menú lateral:
with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Automodeladorv.1")
    choice = st.radio("Navigation", ["Subir ficheros", "Análisis descriptivo", "Preprocesado", "Modelaje", "Generar nuevas predicciones"])
    st.info("Esta aplicación automatiza el proceso de creación de un modelo de predicción de datos para datasets con variables de respuesta binarias o multiclase.")

#Ahora, para cada una de las opciones del menú generamos el código de lo que verá el usuario según la opción seleccionada:

#Carga de ficheros: indicamos al usuario que debe subir el dataset para entrenar el modelo y luego el dataset para generar predicciones:
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
        # Encontrar columna con más datos missing
        descriptive = df.describe()

        with st.container():
            st.markdown("Datos generales por categoría", unsafe_allow_html=True)
            st.dataframe(descriptive)

        # Información sobre tipos de variables
        var_types = df.dtypes
        categorical_vars = sum(var_types == 'object')
        numerical_vars = sum(var_types != 'object')

        with st.container():
            st.subheader("Información sobre el tipo de variables")
            st.write("Cantidad de variables categóricas: ", categorical_vars)
            st.write('Cantidad de variables numéricas: ', numerical_vars)

        # Heatmap de correlaciones
        st.subheader("Heatmap de correlaciones")
        corr = df.select_dtypes(exclude=['object', 'category']).corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr)
        st.pyplot(plt)

        # Y le damos la opción al usuario de generar adicionalmente un análisis descriptivo más detallado a través de un botón:
        st.subheader("Análisis descriptivo detallado")
        if st.button('Ejecutar Análisis descriptivo avanzado'):
            profile_df = df.profile_report()
            st_profile_report(profile_df)


    else:
        st.warning("Please upload a dataset first.")


# Preprocesado

#Creamos una columna que identifique cada dataset para luego concatenarlos y aplicar el preprocesado en conjunto
df['type'] = 'train'
prediction_df['type'] = 'pred'
df1 = pd.concat([df, prediction_df], sort = False, ignore_index=False)

X_train = None
y_train = None
X_test = None
y_test = None
data_pred = None

#A continuación definimos funciones que se utilizarán para el preprocesado de los datos:

#Función para convertir los outliers (definidos con un umbral de 3 desviaciones estandar) en datos nulos:
def eliminar_outliers(dataset, umbral=3):
    dataset_outliers = dataset.copy()
    numeric_columns = dataset_outliers.select_dtypes(include=[np.float64, np.int64]).columns
    numeric_columns = [col for col in numeric_columns if col != target]
    for columna in dataset_outliers.columns:
        if dataset_outliers[columna].dtype in numeric_columns:
            z_scores = np.abs((dataset_outliers[columna] - dataset_outliers[columna].mean()) / dataset_outliers[columna].std())
            dataset_outliers[columna] = np.where(z_scores > umbral, np.nan, dataset_outliers[columna])
    return dataset_outliers

#Función para eliminar las columnas con más de un 25% de datos nulos
def eliminar_columnas_con_nulos(dataset, porcentaje_limite=0.25):
    num_nulos_limite = len(dataset) * porcentaje_limite
    dataset_limpio = dataset.dropna(axis=1, thresh=num_nulos_limite)
    return dataset_limpio

#Función para eliminar registros duplicados si los hubiera:
def eliminar_registros_duplicados(df):
    df_sin_duplicados = df.drop_duplicates()
    return df_sin_duplicados

#Función para convertir variables categoricas a tipo category
def convertir_a_categoricas(dataset, valor_limite=10):
    columnas = dataset.columns
    for columna in columnas:
        if dataset[columna].nunique() < valor_limite:
            dataset[columna] = dataset[columna].astype('category')
    return dataset

#Función para imputar valores nulos: se usará la mediana para las variables numéricas y la moda para las variables categóricas
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

#Función para normalizar las variables numéricas
def normalizar_dataset(dataset):
    scaler = StandardScaler()
    variables_numericas = dataset.select_dtypes(include=['int64', 'float64']).drop(columns=['id', target], errors='ignore')
    dataset[variables_numericas.columns] = scaler.fit_transform(variables_numericas)
    return dataset

#Función para transformar las variables categóricas a numéricas con label encoding:
label_mapping = {}
def label_encoding(dataset):
    # Generamos el label encoding para las columnas tipo categoricas u objeto:
    for columna in dataset.select_dtypes(include=['category', 'object']).columns.difference(['type']):
        label_encoder = LabelEncoder()
        dataset[columna] = label_encoder.fit_transform(dataset[columna])
        # Vamos guardando para cada categoría su valor original y el encoded en un diccionario, que luego usaremos para mostrar las predicciones con los nombres originales de cada categoría
        label_mapping[columna] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    # Nos interesa que la key del diccionario sea el valor encoded, por lo que invertimos el orden
    lab_map_rev = {outer_key: {str(v): str(k) for k, v in inner_dict.items()} for outer_key, inner_dict in label_mapping.items()}
    st.session_state.lab_map_rev = lab_map_rev

    return dataset

#Función para verificar si los datos de la variable objetivo están desbalanceados:
def verificar_desbalanceo(dataset, variable_objetivo):
    conteo_clases = dataset[variable_objetivo].value_counts()
    porcentaje_menor_clase = (conteo_clases.min() / conteo_clases.sum()) * 100
    umbral_desbalanceo = 10  # se considera que hay desbalanceo si una clase representa menos del 10% del total
    desbalanceo = porcentaje_menor_clase < umbral_desbalanceo

    return desbalanceo

#Función para realizar un resampling en caso de que haya desbalanceo en la variable objetivo
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
                                           random_state=random_state)] + [df[df[target_column.name] != minority_class]])
    #Esta función la usaremos despues del split train-test, por lo que debemos devolver los valores de X y Y por separado:
    X = df_resampled.drop(columns=[target_column.name])
    Y = df_resampled[target_column.name]

    return X, Y

if choice == "Preprocesado":
    st.title("Preprocesado")
    if df is not None:
        # Se le pide al usuario que elija el nombre de la variable objetivo de la lista de todas las variables y la elegida se guarda como target:
        st.session_state.target = st.selectbox('Elige la variable objetivo', df.columns)
        target = st.session_state.target

        #Cuando el usuario hace clic en el botón "Preprocesado", se ejecutan todas las funciones definidas anteriormente y se van mostrando los resultados al usuario:
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

            #Antes de revisar el desbalanceo y de hacer resampling, separamos de nuevo los dos datasets que habíamos unido al inicio y con el dataset de training, hacemos el split train-test:
            data_train = df_encoded[df_encoded['type'] == 'train']
            data_train.drop('type', axis=1, inplace=True)
            st.session_state.data_pred = df_encoded[df_encoded['type'] == 'pred']
            st.session_state.data_pred.drop('type', axis=1, inplace=True)
            st.session_state.data_pred.drop(target, axis=1, inplace=True)
            X_train1, st.session_state.X_test, y_train1, st.session_state.y_test = train_test_split(data_train.drop(target, axis=1),
                                                                                                                    data_train[target],
                                                                                                                    test_size=0.2,
                                                                                                                    random_state=1234,
                                                                                                                    stratify=data_train[target])
            #Ahora revisamos si hay desbalanceo únicamente sobre los datos que usaremos para entrenar al modelo (X_train)
            st.subheader("Checkear desbalanceo")
            desbalanceo = verificar_desbalanceo(data_train, target)
            if desbalanceo:
                st.warning("El dataset sufre de desbalanceo en la variable objetivo. Procedemos a realizar un resampling.")
                st.session_state.X_train, st.session_state.y_train = resample_variable(pd.concat([X_train1, y_train1], axis=1), y_train1)
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
    #En caso de que sea una clasificación binomial, además generaremos la curva ROC
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
        #Nos aseguramos de que X_train esté en session_state lo que nos indicaría que el preprocesado ya se realizó, sino se indicaría al usuario que debe completarlo primero.
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
                    #Primero verificamos si el output es multiclase y en ese caso no se ejecutaría la regresión logística, solo los otros dos modelos
                    unique_classes = y_train.unique()
                    if len(unique_classes) > 2 and model.__class__.__name__ == "LogisticRegression":
                        st.warning("La variable objetivo tiene más de 2 clases, por lo que no se ejecutará el modelo de regresión logística")
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
                                                                    ascending=False).head(20).style.background_gradient()
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
                                                                    ascending=False).head(20).style.background_gradient()
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

                        best_model = LogisticRegression(**best_hyperparameters_lr, max_iter=10000, random_state=123)
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
                #Reemplazamos los valores codificados del label encoding por los valores originales para mostrarlos al usuario
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

#Creamos la función para cargar el nuevo dataset para generar nuevas predicciones:
def load_pred2():
    if os.path.exists('./predictions2.csv'):
        return pd.read_csv('predictions2.csv', index_col=None)
    return None
to_pred = load_pred2()
data_pred2 = None

#Cuando el usuario selecciona del menú lateral "Generar nuevas predicciones", le pedimos que cargue un nuevo dataset para generar las predicciones del mismo
if choice == "Generar nuevas predicciones":
    if 'predictions' in st.session_state:
        st.title("Generar nuevas predicciones")
        target = st.session_state.target
        best_model = st.session_state.best_model
        predictions_file2 = st.file_uploader("**Subir un dataset cuyos outputs desconozcas para generar predicciones**")
        if predictions_file2:
            to_pred = pd.read_csv(predictions_file2, index_col=None)
            to_pred.to_csv('predictions2.csv', index=None)
            st.write('Dataset para generar predicciones:')
            st.dataframe(to_pred)

            #ejecutamos preprocesado nuevamente para limpiar los datos a predecir
            to_pred['type'] = "pred2"
            df2 = pd.concat([df, to_pred], sort=False, ignore_index=False)
            df_limpio2 = eliminar_columnas_con_nulos(df2, porcentaje_limite=0.5)
            df_categorico2 = convertir_a_categoricas(df_limpio2, valor_limite=10)
            df_imputado2 = imputar_valores_nulos(df_categorico2)
            df_normalizado2 = normalizar_dataset(df_imputado2)
            df_clean2 = label_encoding(df_normalizado2)

            #Volvemos a separar el dataset de predicciones ya limpio
            st.session_state.data_pred2 = df_clean2[df_clean2['type'] == 'pred2']
            st.session_state.data_pred2.drop('type', axis=1, inplace=True)
            st.session_state.data_pred2.drop(target, axis=1, inplace=True)


            st.write("Dataset limpio y transformado:")
            st.dataframe(st.session_state.data_pred2)

        #Generar predicciones:
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

