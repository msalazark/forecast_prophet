# Import Streamlit and other necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import openpyxl
import json

# Visualización
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
import seaborn as sns

# Pronósticos
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from prophet.plot import plot_plotly, plot_components_plotly

# Estadísticas
import statsmodels.api as sm
import statsmodels.formula.api as sm
from statsmodels.formula.api import ols
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats
from scipy.stats import shapiro, ttest_ind, chi2_contingency, levene, t, ttest_rel, ttest_ind
import statsmodels.api as sma
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Funciones
from data_functions import calcular_estadisticas_mensuales, xboxplots, remove_outliers, validacion_estadistica

fecha_inicio = date.today()
fecha_fin = date.today()
fecha_actual  = date.today()
uploaded_file = ""
ready = False
meta_publicaciones = 0


# Estilo CSS para el pie de página
footer_style = """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: black;
        text-align: center;
        padding: 10px;
        font-size: 14px;
    }
    </style>
    <div class="footer">
        <p>Aliado Digital © 2024. Todos los derechos reservados.</p>
    </div>
    """



# Define a function for forecasting
def forecast(data, date_col, variable_col, periods, frequency):
    data = data[[date_col, variable_col]].dropna()
    data.rename(columns={date_col: 'ds', variable_col: 'y'}, inplace=True)

    model = Prophet()
    model.fit(data)

    future = model.make_future_dataframe(periods=periods, freq=frequency)
    forecast = model.predict(future)

    return forecast, model


# Streamlit user interface
def main():
    global uploaded_file, fecha_inicio, fecha_fin, fecha_hoy
    st.set_page_config(layout="wide")
    sns.set(font_scale=0.8)
    st.title("Data Analysis and Forecasting")
    st.sidebar.title('Análitica')
    fecha_inicio = st.sidebar.date_input("Fecha de inicio", date.today())
    fecha_fin = st.sidebar.date_input("Fecha de fin", date.today())
    fecha_actual = st.sidebar.date_input("Fecha de corte", date.today())
    meta_publicaciones = st.sidebar.number_input( "Publicaciones" , 10 )

    # Añadir el pie de página a la aplicación
    st.markdown(footer_style, unsafe_allow_html=True)

    if not uploaded_file:
        uploaded_file = st.sidebar.file_uploader("Carga el excel que contiene fecha y dato a pronósticar", type=['xlsx'])
        if uploaded_file:
            data = pd.read_excel(uploaded_file)
            ready = True
        else:
            ready = False

    if ready:
        tab1, tab2, tab3 = st.tabs(
            ["Validación estadística", "Forcasting", "Análisis"])

        with tab1:
            st.header("Validación estadística")
            if uploaded_file is not None:

                numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
                columns_to_plot = st.multiselect("Seleccione las columnas para analizar:", numeric_columns,
                                                 default=numeric_columns[1:])
                dimension_columna = st.selectbox("Selecciona la variabla clasificadora", data.columns, key="selectbox_dimension_validacion")

                if st.button("validar"):
                    validacion_estadistica(data, columns_to_plot, columna_cat=dimension_columna)


        with tab2:
            st.header("Forecasting")
            if uploaded_file is not None:

                date_col = st.selectbox("Selecciona campo fecha", data.columns)
                variable_col = st.selectbox("Selecciona el indicador a pronósticar", data.columns)
                periods = st.slider("Número de periodos a Pronósticar", 30, 365, 90)
                frequency = st.selectbox("Selecciona la frecuencia", ["D", "W", "M", "Q", "Y"], index=0)

                if st.button("Forecast"):
                    forecast_df, model = forecast(data, date_col, variable_col, periods, frequency)
                    st.write(forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
                    fig1 = model.plot(forecast_df)
                    st.pyplot(fig1)

                    fig2 = plot_plotly(model, forecast_df)
                    st.plotly_chart(fig2, use_container_width=True)

                    fig3 = plot_components_plotly(model, forecast_df)
                    st.plotly_chart(fig3, use_container_width=True)

                    dfMes = data[(data[date_col].dt.date >= fecha_inicio)][[date_col, variable_col]]
                    dfMes["trend"] = dfMes[variable_col]
                    dfMes["yhat_lower"] = dfMes[variable_col]
                    dfMes["yhat_upper"] = dfMes[variable_col]

                    st.write( dfMes[variable_col].describe() )

                    publicaciones = dfMes.shape[0]
                    print("Publicaciones realizadas: ", publicaciones)
                    publicaciones_pendientes = (meta_publicaciones * 12) - publicaciones
                    print("Publicaciones pendientes: ", publicaciones_pendientes)

                    dfComMes = \
                    forecast_df[(forecast_df["ds"].dt.date >= fecha_actual) & (forecast_df["ds"].dt.date <= fecha_fin)][
                        ["ds", "trend", "yhat_lower", "yhat_upper", "yhat"]]
                    dfComMes = dfComMes[dfComMes["yhat"] > 0]
                    dfComMes.rename(columns={"ds": date_col, "yhat": variable_col}, inplace=True)

                    print(dfComMes.shape)
                    #dfComMes.head()

                    dfFinal = pd.concat([dfMes, dfComMes ])
                    dfFinal[variable_col] = dfFinal[variable_col].astype(int)
                    print(dfFinal.shape)
                    print(dfFinal.head())
                    fig4, ax = plt.subplots(figsize=(15, 6))
                    dfFinal.plot(x=date_col , ax=ax)
                    st.pyplot(fig4)


                    st.write( dfFinal.describe() )

                    dfPronostico = calcular_estadisticas_mensuales(dfFinal, date_col, variable_col)
                    st.write(  dfPronostico )

                    ###################################################################################
                    st.title("Sin Outliers")
                    print(data.shape)
                    data_so = remove_outliers(data, variable_col)
                    print(data_so.shape)

                    forecast_dfso, modelso = forecast(data_so, date_col, variable_col, periods, frequency)
                    st.write(forecast_dfso[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
                    fig1 = modelso.plot(forecast_dfso)
                    st.pyplot(fig1)

                    fig2 = plot_plotly(modelso, forecast_dfso)
                    st.plotly_chart(fig2, use_container_width=True)

                    fig3 = plot_components_plotly(modelso, forecast_dfso)
                    st.plotly_chart(fig3, use_container_width=True)

                    dfMes = data_so[(data_so[date_col].dt.date >= fecha_inicio)][[date_col, variable_col]]
                    dfMes["trend"] = dfMes[variable_col]
                    dfMes["yhat_lower"] = dfMes[variable_col]
                    dfMes["yhat_upper"] = dfMes[variable_col]

                    st.write(dfMes[variable_col].describe())

                    publicaciones = dfMes.shape[0]
                    print("Publicaciones realizadas: ", publicaciones)
                    publicaciones_pendientes = (meta_publicaciones * 12) - publicaciones
                    print("Publicaciones pendientes: ", publicaciones_pendientes)

                    dfComMes = \
                        forecast_dfso[
                            (forecast_dfso["ds"].dt.date >= fecha_actual) & (forecast_dfso["ds"].dt.date <= fecha_fin)][
                            ["ds", "trend", "yhat_lower", "yhat_upper", "yhat"]]
                    dfComMes = dfComMes[dfComMes["yhat"] > 0]
                    dfComMes.rename(columns={"ds": date_col, "yhat": variable_col}, inplace=True)
                    dfComMes[variable_col] = dfComMes[variable_col].astype(int)
                    print(dfComMes.shape)
                    dfComMes.head()

                    dfFinal = pd.concat([dfMes, dfComMes])
                    print(dfFinal.shape)
                    fig4, ax = plt.subplots(figsize=(15, 6))
                    dfFinal.plot(x=date_col , ax=ax)
                    st.pyplot(fig4)

                    st.write(dfFinal.describe())

                    dfPronostico = calcular_estadisticas_mensuales(dfFinal, date_col, variable_col)
                    st.write(dfPronostico)



        with tab3:

            st.header("Análisis")
            if uploaded_file is not None:
                variable_col = st.selectbox("Selecciona la variable a analizar", data.columns)
                dimension_col = st.selectbox("Selecciona la variabla clasificadora", data.columns)
                if st.button("Analisis"):

                    resultado_anova, figura_boxplot, comentario = xboxplots(data, variable_col, dimension_col)
                    # Mostrar resultados de ANOVA
                    st.write('Resultados de ANOVA:')
                    st.dataframe(resultado_anova)
                    st.write(comentario)

                    # Mostrar el gráfico boxplot
                    st.write('Boxplot:')
                    plt.figure(figsize=(15, 6))
                    plt.title("Boxplot por Categoría")
                    figura_boxplot = sns.boxplot(x=dimension_col, y=variable_col, data=data)
                    st.pyplot(plt)
                    #st.pyplot(figura_boxplot)


if __name__ == "__main__":
    main()
