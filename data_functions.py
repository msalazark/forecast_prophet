# Import Streamlit and other necessary libraries
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import openpyxl
import json
import streamlit as st

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

def resultadoEstadistico(data, campo, clasificador, valor ):
  dataCon =  data[ (data[clasificador] == valor) ]
  dataCon = dataCon[campo]
  dataSin = data[ (data[clasificador] != valor) ]
  dataSin = dataSin[campo]
  # Aplica el test t de Student
  t_statistic, p_value = ttest_ind(dataCon, dataSin)
  # Imprime los resultados
  print("Estadística t:", t_statistic)
  print("Valor p:", p_value)
  # Grados de libertad
  error_est = 0.05
  df = 20
  # Valor crítico para una prueba de dos colas
  critical_value = t.ppf(1 - error_est / 2, df)
  print("Valor crítico:", critical_value)

  # Intervalo de confianza para la media de data1
  mean_ci_data1 = stats.t.interval(0.95, len(dataCon)-1, loc=np.mean(dataCon), scale=stats.sem(dataCon))
  mean_ci_data2 = stats.t.interval(0.95, len(dataSin)-1, loc=np.mean(dataSin), scale=stats.sem(dataSin))

  print("Intervalo de confianza (95%) de data Con:", mean_ci_data1)
  print("Intervalo de confianza (95%) de data Sin:", mean_ci_data2)

  sns.boxplot(x=clasificador, y=campo, data=data)
  plt.show()


def pronostico( data0 , campoFecha, campoVariable, periodos, frecuencia ):
  dataPron = data0[[campoFecha,campoVariable]].copy()
  dataPron.rename(columns={campoFecha:"ds", campoVariable:"y"}, inplace = True)

  m = Prophet()
  m.interval_width = 0.95
  m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
  m.fit(dataPron)

  future = m.make_future_dataframe(periods=periodos, freq=frecuencia )
  forecastInter = m.predict(future)

  figInter1 = m.plot(forecastInter)

  return forecastInter, m


def xboxplots(data, campo, clasificador ):

  # Aplicamos Anova
  modelo = ols( campo + ' ~ '+ clasificador, data=data).fit()
  anova_tabla = sma.stats.anova_lm(modelo, typ=2)
  #print(anova_tabla)

  # Mostramos Box Plot por categoría
  fig1 = plt.figure()
  sns.boxplot(x=clasificador, y=campo, data=data)
  plt.title(f'Box Plot de {campo} por {clasificador}')
  plt.grid(True)

  comentario = "PR(>F): Un valor p pequeño (generalmente menor a 0.05) indica que al menos dos grupos son significativamente diferentes entre sí"
  #plt.show()

  return anova_tabla, fig1, comentario


def select_outliers(df, columna, threshold=1.5):
    # Calcular el rango intercuartílico (IQR) para cada columna
    q1_col1 = df[columna].quantile(0.25)
    q3_col1 = df[columna].quantile(0.75)
    iqr_col1 = q3_col1 - q1_col1

    # Definir los límites para identificar outliers
    lower_bound_col1 = q1_col1 - threshold * iqr_col1
    upper_bound_col1 = q3_col1 + threshold * iqr_col1

    # Filtrar el DataFrame con outliers
    df_filtered_low  = df[(df[columna] < lower_bound_col1) ]
    df_filtered_high = df[(df[columna] > upper_bound_col1) ]

    return df_filtered_low, df_filtered_high

def remove_outliers(df, columna , threshold=1.5):
    # Calcular el rango intercuartílico (IQR) para cada columna
    q1_col1 = df[columna].quantile(0.25)
    q3_col1 = df[columna].quantile(0.75)
    iqr_col1 = q3_col1 - q1_col1

    # Definir los límites para identificar outliers
    lower_bound_col1 = q1_col1 - threshold * iqr_col1
    upper_bound_col1 = q3_col1 + threshold * iqr_col1

    # Filtrar el DataFrame para excluir outliers
    df_filtered = df[
        (df[columna] >= lower_bound_col1) & (df[columna] <= upper_bound_col1)
    ]

    return df_filtered

def remove_outliers_combined(df, column1, column2, threshold=1.5):
    # Calcular el rango intercuartílico (IQR) para cada columna
    q1_col1 = df[column1].quantile(0.25)
    q3_col1 = df[column1].quantile(0.75)
    iqr_col1 = q3_col1 - q1_col1

    q1_col2 = df[column2].quantile(0.25)
    q3_col2 = df[column2].quantile(0.75)
    iqr_col2 = q3_col2 - q1_col2

    # Definir los límites para identificar outliers
    lower_bound_col1 = q1_col1 - threshold * iqr_col1
    upper_bound_col1 = q3_col1 + threshold * iqr_col1

    lower_bound_col2 = q1_col2 - threshold * iqr_col2
    upper_bound_col2 = q3_col2 + threshold * iqr_col2

    # Filtrar el DataFrame para excluir outliers
    df_filtered = df[
        (df[column1] >= lower_bound_col1) & (df[column1] <= upper_bound_col1) &
        (df[column2] >= lower_bound_col2) & (df[column2] <= upper_bound_col2)
    ]

    return df_filtered

def calcular_estadisticas_mensuales(df, fecha_columna, valor_columna):
    df[fecha_columna] = pd.to_datetime(df[fecha_columna])
    # Agregar columnas de mes y año
    df['mes'] = df[fecha_columna].dt.month
    df['anio'] = df[fecha_columna].dt.year

    # Calcular el promedio, máximo y mínimo por mes
    resultados = df.groupby(['anio', 'mes']).agg(
        promedio_mensual=(valor_columna, 'mean'),
        maximo_mensual=(valor_columna, 'max'),
        minimo_mensual=(valor_columna, 'min'),
        total_publicaciones=(valor_columna, 'count')
    ).reset_index()

    return resultados


def calcular_estadisticas_trimestrales(df, fecha_columna, valor_columna):
    df[fecha_columna] = pd.to_datetime(df[fecha_columna])
    dfc = df.copy()
    dfc.set_index(fecha_columna, inplace=True)
    promedio_trimestral = dfc.resample('Q').mean()
    return promedio_trimestral


def validacion_estadistica(df, variables, columna_cat=None):
    """
    - df: DataFrame de pandas.
    - columna_cat: Columna categórica para usar en la prueba de Levene (si se proporciona).
    """

    for columna in variables:
        st.title(f"Análisis de la columna: {columna}")
        datos = df[columna].dropna()  # Excluir NaNs

        # Determinar si la columna es numérica o categórica
        if pd.api.types.is_numeric_dtype(datos):
            # Histograma para inspección visual

            fig, ax = plt.subplots(figsize=(15,6))
            ax.hist(datos, bins=10)
            ax.set_title(f"Histograma de {columna}")
            plt.tight_layout()
            st.pyplot(fig)

            st.write("------------------------------------------------------------\n")

            # Prueba de Shapiro-Wilk para normalidad
            stat, p = shapiro(datos)
            st.write(f"Prueba de Shapiro-Wilk: Estadístico={stat:.3f}, p={p:.3f}")
            if p > 0.05:
                st.write("Los datos parecen seguir una distribución normal.")
            else:
                st.write("Los datos no parecen seguir una distribución normal.")

            st.write("------------------------------------------------------------\n")

            # Prueba de Levene si se proporciona una columna categórica
            if columna_cat:
                grupos = df.groupby(columna_cat)[columna].apply(list)
                stat, p = levene(*grupos)
                st.write(f"Prueba de Levene: Estadístico={stat:.3f}, p={p:.3f}")
                if p > 0.05:
                    st.write("Las varianzas son homogéneas.")
                else:
                    st.write("Las varianzas no son homogéneas.")

            st.write("------------------------------------------------------------\n")

        elif pd.api.types.is_categorical_dtype(datos) or pd.api.types.is_object_dtype(datos):
            # Conteo de valores para variables categóricas
            st.write(f"Conteo de valores para {columna}:\n{datos.value_counts()}")

        st.write("------------------------------------------------------------\n")


###########################################################
def pronostico_arima(df, variable, puntos, p,d,q):
    data = df[variable]
    model = ARIMA(data, order=(p,d,q))
    model_fit = model.fit()
    print(model_fit.summary())
    preds = model_fit.forecast(steps=puntos) # Pronosticar los siguientes puntos

    plt.figure(figsize=(12, 8))

    t_obs = range(len(data))
    t_future = range(len(data), len(data) + len(preds))

    plt.plot(t_obs, data, label='Observaciones', marker='o', color='blue')
    plt.plot(t_future, preds, label='Pronóstico ARIMA', color='green', linestyle='--', marker='x')

    plt.legend()
    plt.title('Observaciones y Pronóstico ARIMA')
    plt.xlabel('Fecha')
    plt.ylabel('Valor')
    plt.show()

    return preds

def pronostico_sarimax(df, variable, variable_fecha , puntos, estacionalidad):
    dfLocal = df.copy()
    data = dfLocal[variable]
    #dfLocal.set_index(variable_fecha, inplace=True)
    model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, estacionalidad)) # 12 indica una estacionalidad anual
    model_fit = model.fit()
    print(model_fit.summary())
    preds = model_fit.forecast(steps=puntos)

    plt.figure(figsize=(12, 8))

    t_obs = range(len(data))
    t_future = range(len(data), len(data) + len(preds))

    plt.plot(t_obs, data, label='Observaciones', marker='o', color='blue')
    plt.plot(t_future, preds, label='Pronóstico SARIMAX', color='green', linestyle='--', marker='x')

    plt.legend()
    plt.title('Observaciones y Pronóstico SARIMAX')
    plt.xlabel('Fecha')
    plt.ylabel('Valor')
    plt.show()

    return preds


def pronostico_kalman( df, variable , iteracciones, n_future_steps ):
    observations = df[variable]
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
    kf = kf.em(observations, iteracciones)
    states_pred , states_cov = kf.smooth(observations)

    # nuevos puntos ########
    # Último estado medio y covarianza (estimados con .smooth() o .filter())
    last_state_mean = states_pred[-1]
    last_state_covariance = states_cov[-1]

    # Predecir futuros estados
    future_states_mean = np.zeros((n_future_steps, kf.n_dim_state))
    future_states_covariance = np.zeros((n_future_steps, kf.n_dim_state, kf.n_dim_state))

    for i in range(n_future_steps):
        # Utiliza la última media y covarianza para predecir el siguiente estado
        next_state_mean, next_state_covariance = kf.filter_update(
            filtered_state_mean=last_state_mean,
            filtered_state_covariance=last_state_covariance,
            observation=None  # No hay nueva observación para el futuro
        )

        # Almacena las predicciones
        future_states_mean[i] = next_state_mean
        future_states_covariance[i] = next_state_covariance

        # Actualiza la última media y covarianza para la siguiente iteración
        last_state_mean, last_state_covariance = next_state_mean, next_state_covariance

    t_obs = range(len(observations))
    t_future = range(len(observations), len(observations) + len(future_states_mean))

    plt.plot(t_obs, observations, label='Observaciones', color='blue', linestyle='None', marker='o')
    plt.plot(t_obs, states_pred, label='Estimaciones Kalman', color='red', linewidth=2)
    plt.plot(t_future, future_states_mean, label='Predicciones Futuras', color='green', linestyle='--', marker='x')
    plt.legend()
    plt.title('Comparación de Observaciones y Estimaciones del Filtro de Kalman')
    plt.xlabel('Tiempo')
    plt.ylabel('Valor')
    plt.xlim([0, max(t_future)])
    plt.show()

    return states_pred


def pronostico_rnn_recurrente( X, y , n_timesteps, n_features, pasos ):

    # Definir modelo
    model = Sequential()
    model.add(SimpleRNN(units=50, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=200, verbose=0)

    predictions = []
    X_new = X[-1]

    for _ in range(pasos):
        # Hacer una predicción de un solo paso
        pred = model.predict(X_new)
        print(pred[0][0])
        predictions.append(pred[0][0])
        X_new = pred

    return predictions


def pronostico_media_movil(data, variable, variable_fecha, periodos, pasos ):

    pronosticos = []
    df = data.copy()
    df['media_movil'] = df[variable].rolling(window=periodos).mean()

    # Obtener el último valor de la media móvil
    ultimos_valores = list(df[variable][-periodos:])

    for _ in range(pasos):  # Pronosticar 30 pasos hacia adelante
        nuevo_pronostico = np.mean(ultimos_valores)  # Calcula la nueva media móvil
        pronosticos.append(nuevo_pronostico)  # Añade el pronóstico a la lista
        ultimos_valores.append(nuevo_pronostico)  # Añade el pronóstico a la lista de los últimos valores
        ultimos_valores = ultimos_valores[1:]  # Elimina el valor más antiguo

    # Añade los pronósticos al DataFrame para visualización
    fechas_futuras = pd.date_range(start=df[variable_fecha].iloc[-1], periods=pasos+1, freq='D')[1:]  # Crea fechas futuras, excluyendo el último día ya presente en df
    df_pronostico = pd.DataFrame({variable_fecha: fechas_futuras, 'pronostico': pronosticos})

    # Visualizar los resultados
    plt.figure(figsize=(12, 6))
    plt.plot(df[variable_fecha], df[variable], label=variable)
    plt.plot(df[variable_fecha], df['media_movil'], label='Media Móvil', color='red')
    plt.plot(df_pronostico[variable_fecha], df_pronostico['pronostico'], label='Pronóstico', color='green', linestyle='--')
    plt.legend()
    plt.title('Serie Temporal con Media Móvil')
    plt.xlabel(variable_fecha)
    plt.ylabel(variable)
    plt.show()

    return df, df_pronostico


def preparar_data( df, variable, n_timesteps):
    data = df[variable].values
    X, y = [], []

    for i in range(len(data) - n_timesteps):
        X.append(data[i:i + n_timesteps])
        y.append(data[i + n_timesteps])

    # Convertir listas a arrays de NumPy para el entrenamiento
    X, y = np.array(X), np.array(y)

    # Reshape de X para [muestras, pasos de tiempo, características]
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y


def pronostico_lstm():
    return None