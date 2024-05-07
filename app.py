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
from data_functions import calcular_estadisticas_mensuales

fecha_inicio = date.today()
fecha_fin = date.today()
fecha_actual  = date.today()
uploaded_file = ""
ready = False
meta_publicaciones = 0

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
    st.sidebar.title('TBWA \\ Data ')
    fecha_inicio = st.sidebar.date_input("Fecha de inicio", date.today())
    fecha_fin = st.sidebar.date_input("Fecha de fin", date.today())
    fecha_actual = st.sidebar.date_input("Fecha de corte", date.today())
    meta_publicaciones = st.sidebar.number_input( "Publicaciones" , 10 )

    if not uploaded_file:
        uploaded_file = st.sidebar.file_uploader("Carga el excel que contiene fecha y dato a pronósticar", type=['xlsx'])
        if uploaded_file:
            data = pd.read_excel(uploaded_file)
            ready = True
        else:
            ready = False

    if ready:
        tab1, tab2 = st.tabs(
            ["Forcasting (Prophet)", "Análisis estadístico"])

        with tab1:
            st.header("Forecasting with Prophet")
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
                    dfComMes.rename(columns={"ds": "fecha", "yhat": variable_col}, inplace=True)
                    dfComMes[variable_col] = dfComMes[variable_col].astype(int)
                    print(dfComMes.shape)
                    dfComMes.head()

                    dfFinal = pd.concat([dfMes, dfComMes ])
                    print(dfFinal.shape)
                    fig4 = dfFinal.plot(x=date_col).get_figure()
                    st.pyplot(fig4)

                    st.write( dfFinal.describe() )

                    dfPronostico = calcular_estadisticas_mensuales(dfFinal, date_col, variable_col)
                    st.write(  dfPronostico )

        with tab2:
            st.header("Análisis")




if __name__ == "__main__":
    main()
