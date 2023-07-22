import pandas as pd


def dividir_dataframes(df):
    # Pedir al usuario que ingrese la columna que se utilizará como etiquetas (objetivo)
    columna_objetivo = input("Ingresa el nombre de la columna objetivo (etiquetas) para predecir: ")

    # Dividir el dataframe en datos para entrenamiento (df_train) y datos para predicción (df_pred)
    try:
        df_train = df.dropna(subset=[columna_objetivo])  # Eliminar filas con valores faltantes en la columna objetivo
        df_pred = df[~df.index.isin(df_train.index)]  # Filtrar las filas no presentes en df_train
        print("Dataframes para entrenamiento y predicción creados correctamente.")
        return df_train, df_pred
    except KeyError:
        print("La columna objetivo especificada no se encuentra en el dataset.")
        return None, None


if __name__ == "__main__":
    # Llamar a las funciones para leer el dataset y dividir los dataframes
    df = leer_dataset()
    if df is not None:
        df_train, df_pred = dividir_dataframes(df)


def analisis_descriptivo(df_train):
    # Mostrar la forma del dataframe (número de filas y columnas)
    print("Forma del DataFrame:")
    print(df_train.shape)

    # Mostrar información general del dataframe
    print("\nInformación general del DataFrame:")
    print(df_train.info())

    # Contar la cantidad de valores nulos por columna
    print("\nCantidad de valores nulos por columna:")
    print(df_train.isnull().sum())

    # Contar la cantidad de valores faltantes (missing values) por columna
    print("\nCantidad de valores faltantes por columna:")
    print(df_train.isna().sum())

    # Mostrar estadísticas descriptivas de las columnas numéricas
    print("\nEstadísticas descriptivas de columnas numéricas:")
    print(df_train.describe())
    print("\nValor más común en la columna 'nombre_columna':")
    print(df_train['nombre_columna'].mode().iloc[0])
    print("\nCorrelaciones entre columnas:")
    print(df_train.corr())


if __name__ == "__main__":
    # Suponiendo que ya tienes df_train creado desde la sección anterior

    import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html


def estadistica_descriptiva(df_train):
    # Utilizamos describe() para obtener estadísticas descriptivas de todas las columnas
    descripcion = df_train.describe()

    # Crear un gráfico de barras para visualizar la distribución de los datos
    fig_dist = px.bar(descripcion.loc["mean"], labels={"value": "Media"}, title="Distribución de la media")

    # Crear un box plot para identificar outliers en columnas numéricas
    fig_box = go.Figure()
    for col in df_train.select_dtypes(include=[float, int]):
        fig_box.add_trace(go.Box(y=df_train[col], name=col))
    fig_box.update_layout(title="Box Plot - Identificación de Outliers")

    # Crear una matriz de correlación
    matriz_correlacion = df_train.corr()
    fig_heatmap = px.imshow(matriz_correlacion,
                            labels=dict(x="Columnas", y="Columnas", color="Correlación"),
                            x=matriz_correlacion.columns,
                            y=matriz_correlacion.columns,
                            color_continuous_scale='RdBu',
                            title="Matriz de correlación")

    # Crear la aplicación Shiny para visualizar los gráficos interactivamente
    app = dash.Dash(__name__)
    app.layout = html.Div([
        dcc.Graph(figure=fig_dist),
        dcc.Graph(figure=fig_box),
        dcc.Graph(figure=fig_heatmap)
    ])

    if __name__ == "__main__":
        # Llamar a la función para realizar el análisis descriptivo
        estadistica_descriptiva(df_train)

    # Llamar a la función para realizar el análisis descriptivo
    analisis_descriptivo(df_train)