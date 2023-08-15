
#PREPROCESADO


#IMPORTS

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


#SUBIR DATASETS
nombre_archivo = "C:/Users/jalil/Desktop/Violence_Reduction_-_Victims_of_Homicides_and_Non-Fatal_Shootings.csv"
df = pd.read_csv(nombre_archivo)


#DEFINICIONES

#Definicion - Eliminar variables con más del 20% de nulos o missings.
def eliminar_columnas_con_nulos(dataset, porcentaje_limite=0.2):
    # Calculamos el número mínimo de valores nulos permitidos por columna
    num_nulos_limite = len(dataset) * porcentaje_limite

    # Eliminamos las columnas con un número de valores nulos mayor o igual al límite
    dataset_limpio = dataset.dropna(axis=1, thresh=num_nulos_limite)

    return dataset_limpio

#Definicion - Pasar las variables con menos de 10 valores únicos a categorical.
def convertir_a_categoricas(dataset, valor_limite=10):
    # Obtenemos la lista de nombres de las columnas
    columnas = dataset.columns

    # Recorremos cada columna y verificamos si tiene menos de 'valor_limite' valores únicos
    for columna in columnas:
        if dataset[columna].nunique() < valor_limite:
            dataset[columna] = dataset[columna].astype('category')

    return dataset

#Definicion - Imputar nulos con mediana (numéricas) y moda (categóricas).
def imputar_valores_nulos(dataset):
    # Imputación de variables numéricas con la mediana
    variables_numericas = dataset.select_dtypes(include=['int64', 'float64'])
    for columna in variables_numericas:
        mediana = dataset[columna].median()
        dataset[columna].fillna(mediana, inplace=True)

    # Imputación de variables categóricas con la moda
    variables_categoricas = dataset.select_dtypes(include='category')
    for columna in variables_categoricas:
        moda = dataset[columna].mode().iloc[0]  # En caso de que haya múltiples modas, toma la primera
        dataset[columna].fillna(moda, inplace=True)

    return dataset

#Definicion - Normalización.
def normalizar_dataset(dataset):
    # Inicializamos el objeto del escalador
    scaler = StandardScaler()

    # Aplicamos la normalización a las variables numéricas
    variables_numericas = dataset.select_dtypes(include=['int64', 'float64'])
    dataset[variables_numericas.columns] = scaler.fit_transform(variables_numericas)

    return dataset

#Definicion - Label Encoding para aquellas variables categóricas con orden jerárquico (preguntando al usuario).
def label_encoding(dataset):
    label_encoder = LabelEncoder()
    for columna in dataset.select_dtypes(include='category'):
        dataset[columna] = label_encoder.fit_transform(dataset[columna])
    return dataset


def obtener_columnas_orden_jerarquico(dataset):
    columnas_orden_jerarquico = []
    columnas_categoricas = dataset.select_dtypes(include='category').columns.tolist()

    print("¿Existen variables categóricas con orden jerárquico?")
    for columna in columnas_categoricas:
        respuesta = input(f"¿La columna '{columna}' tiene orden jerárquico? (Sí/No): ")
        respuesta = respuesta.lower()
        if respuesta == "sí" or respuesta == "si":
            columnas_orden_jerarquico.append(columna)

    return columnas_orden_jerarquico

#Definicion - Checkear si el dataset tiene desbalanceo.
def verificar_desbalanceo(dataset, variable_objetivo):

    # Contamos la cantidad de muestras en cada clase de la variable objetivo
    conteo_clases = dataset[variable_objetivo].value_counts()

    # Calculamos el porcentaje de muestras en la clase más pequeña
    porcentaje_menor_clase = (conteo_clases.min() / conteo_clases.sum()) * 100

    # Definimos un umbral para considerar que hay desbalanceo (puedes ajustar este valor según tus necesidades)
    umbral_desbalanceo = 10  # Por ejemplo, consideraremos que hay desbalanceo si una clase representa menos del 10% del total

    # Verificamos si hay desbalanceo en el dataset
    desbalanceo = porcentaje_menor_clase < umbral_desbalanceo

    return desbalanceo

#Definicion - Dividir en train y test y aplicar resampling en train (solo si hay desbalanceo en el dataset).
# def resampling(dataset, variable_objetivo):
    #clase_mayoritaria = dataset[variable_objetivo].value_counts().idxmax()
    #clase_menoritaria = dataset[variable_objetivo].value_counts().idxmin()

    #df_mayoritaria = dataset[dataset[variable_objetivo] == clase_mayoritaria]
    #df_menoritaria = dataset[dataset[variable_objetivo] == clase_menoritaria]

    #df_menoritaria_resampled = resample(df_menoritaria,
                                        #replace=True,
                                        #n_samples=df_mayoritaria.shape[0],
                                        #random_state=42)

    #df_resampled = pd.concat([df_mayoritaria, df_menoritaria_resampled])

    #return df_resampled


#CODIGO

#Eliminar variables con más del 20% de nulos o missings.
if __name__ == "__main__":
    # Eliminamos las columnas con un 20% o más de valores nulos
    df_limpio = eliminar_columnas_con_nulos(df, porcentaje_limite=0.2)

    # Ahora df_limpio contiene el DataFrame sin las columnas con un alto porcentaje de valores nulos
    print(df_limpio)

#Pasar las variables con menos de 10 valores únicos a categorical.
if __name__ == "__main__":
    # Convertimos las variables con menos de 10 valores únicos en categóricas
    df_categorico = convertir_a_categoricas(df_limpio, valor_limite=10)

    # Ahora df_categorico contiene el DataFrame con las columnas convertidas a categóricas
    print(df_categorico.dtypes)

#Imputar nulos con mediana (numéricas) y moda (categóricas).
if __name__ == "__main__":
    # Imputamos los valores nulos de las variables numéricas con la mediana
    # y los valores nulos de las variables categóricas con la moda
    df_imputado = imputar_valores_nulos(df_categorico)

    # Ahora df_imputado contiene el DataFrame con los valores nulos imputados
    print(df_imputado)

#Normalización (preguntando al usuario).
if __name__ == "__main__":
    # Pedimos al usuario si desea realizar la normalización
    respuesta = input("¿Deseas normalizar el dataset? (Sí/No): ")

    # Convertimos la respuesta a minúsculas para manejar diferentes formas de escribir "Sí" o "No"
    respuesta = respuesta.lower()

    if respuesta == "sí" or respuesta == "si":
        # Realizamos la normalización del dataset
        df_normalizado = normalizar_dataset(df_imputado)
        print("Dataset normalizado:")
        print(df_normalizado)
    elif respuesta == "no":
        # Si el usuario no desea normalizar, simplemente mostramos el dataset original
        print("Dataset sin normalizar:")
        print(df_imputado)
    else:
        print("Respuesta inválida. Por favor, responde 'Sí' o 'No'.")

# Label Encoding para aquellas variables categóricas con orden jerárquico (preguntando al usuario).
if __name__ == "__main__":
    # Preguntamos al usuario por las columnas con orden jerárquico
    columnas_orden_jerarquico = obtener_columnas_orden_jerarquico(df_normalizado)

    # Realizamos el Label Encoding a las variables categóricas
    df_encoded = label_encoding(df_normalizado)

    # Si hay columnas con orden jerárquico, las convertimos nuevamente a categóricas
    if columnas_orden_jerarquico:
        df_encoded[columnas_orden_jerarquico] = df_encoded[columnas_orden_jerarquico].astype('category')

    # Mostramos el DataFrame resultante
    print(df_encoded)

#Checkear si el dataset tiene desbalanceo.
if __name__ == "__main__":
    # Preguntar al usuario por la variable objetivo del dataset
    variable_objetivo = input("Por favor, introduce el nombre de la variable objetivo: ")

    # Verificamos si el dataset sufre de desbalanceo en la variable objetivo
    desbalanceo = verificar_desbalanceo(df_encoded, variable_objetivo)

    # Imprimimos el resultado
    if desbalanceo:
        print("El dataset sufre de desbalanceo en la variable objetivo.")
    else:
        print("El dataset no sufre de desbalanceo en la variable objetivo.")

#Dividir en train y test y aplicar resampling en train (solo si hay desbalanceo en el dataset).
#if __name__ == "__main__":
    # Especificamos el nombre de la columna que representa la variable objetivo
    #variable_objetivo = 'clase_objetivo'

    # Verificamos si el dataset sufre de desbalanceo en la variable objetivo
    #desbalanceo = verificar_desbalanceo(df_encoded, variable_objetivo)

    # Dividimos el dataset en conjuntos de entrenamiento y prueba
    #X = df_encoded.drop(columns=[variable_objetivo])
    #y = df_encoded[variable_objetivo]
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #if desbalanceo:
        # Realizamos el resampling solo en el conjunto de entrenamiento
        #df_train = pd.concat([X_train, y_train], axis=1)
        #df_train_resampled = resampling(df_train, variable_objetivo)

        # Actualizamos X_train y y_train con el conjunto de entrenamiento resampleado
        #X_train_resampled = df_train_resampled.drop(columns=[variable_objetivo])
        #y_train_resampled = df_train_resampled[variable_objetivo]

        #print("Se aplicó resampling al conjunto de entrenamiento debido al desbalanceo.")
        #print("Tamaño del conjunto de entrenamiento antes del resampling:", X_train.shape[0])
        #print("Tamaño del conjunto de entrenamiento después del resampling:", X_train_resampled.shape[0])
    #else:
        #print("No se aplicó resampling. No existe desbalanceo en la variable objetivo.")
        #print("Tamaño del conjunto de entrenamiento:", X_train.shape[0])

    # Mostramos el tamaño del conjunto de prueba
    #print("Tamaño del conjunto de prueba:", X_test.shape[0])






