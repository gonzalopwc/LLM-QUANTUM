import os
import time
from dotenv import load_dotenv
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_groq import ChatGroq
import pandas as pd
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from skfolio import RiskMeasure
from skfolio.optimization import MeanRisk
from matplotlib.ticker import FuncFormatter
from skfolio import PerfMeasure, RatioMeasure, RiskMeasure
from plotly import graph_objects as go

# -------------------------------------------------------------------------------
# 0. Generar diccionarios
# -------------------------------------------------------------------------------

# Mapeo de los nombres completos de las empresas con sus tickers
sp500_companies = {
    'Apple Inc.': 'AAPL',
    'Microsoft Corporation': 'MSFT',
    'Alphabet Inc. (Google)': 'GOOGL',
    'Amazon.com, Inc.': 'AMZN',
    'Meta Platforms, Inc. (Facebook)': 'META',
    'Tesla, Inc.': 'TSLA',
    'Berkshire Hathaway Inc. (Class B)': 'BRK-B',
    'Johnson & Johnson': 'JNJ',
    'JPMorgan Chase & Co.': 'JPM',
    'Visa Inc.': 'V',
    'NVIDIA Corporation': 'NVDA',
    'Procter & Gamble Co.': 'PG',
    'Walt Disney Company': 'DIS',
    'Home Depot, Inc.': 'HD',
    'Pfizer Inc.': 'PFE',
    'Coca-Cola Company': 'KO',
    'PepsiCo, Inc.': 'PEP',
    'McDonald\'s Corporation': 'MCD',
    'Exxon Mobil Corporation': 'XOM',
    'AbbVie Inc.': 'ABBV',
    'Chevron Corporation': 'CVX',
    'Salesforce, Inc.': 'CRM',
    'Intel Corporation': 'INTC',
    'Nike, Inc.': 'NKE',
    'AT&T Inc.': 'T',
    'Merck & Co., Inc.': 'MRK',
    'Adobe Inc.': 'ADBE',
    'Citigroup Inc.': 'C',
    'Lockheed Martin Corporation': 'LMT',
    'UnitedHealth Group Incorporated': 'UNH'
    }

# -------------------------------------------------------------------------------
# 1. Cargar entorno
# -------------------------------------------------------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# -------------------------------------------------------------------------------
# 2. Configuración de la página
# -------------------------------------------------------------------------------
st.set_page_config(
    page_title="LLM QUANTUM",
    page_icon="⚛️",
    initial_sidebar_state="collapsed" 
)

# -------------------------------------------------------------------------------
# 3. Cargar CSS personalizado desde un archivo externo
# -------------------------------------------------------------------------------
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css('styles.css')

# -------------------------------------------------------------------------------
# 4. Inicializar estados de sesión
# -------------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferWindowMemory(k=5)
if "first" not in st.session_state:
    st.session_state["first"] = False
if "question" not in st.session_state:
    st.session_state["question"] = ""
if "box" not in st.session_state:
    st.session_state["box"] = ""
if "stocks" not in st.session_state:
    st.session_state["stocks"] = ""
if "fig1" not in st.session_state:
    st.session_state["fig1"] = ""
if "fig2" not in st.session_state:
    st.session_state["fig2"] = ""
if "response_generated" not in st.session_state:
    st.session_state["response_generated"] = False
if "first_min_var" not in st.session_state:
    st.session_state["first_min_var"] = False
if "first_cov_mat" not in st.session_state:
    st.session_state["first_cov_mat"] = False
if "first_min_var_res" not in st.session_state:
    st.session_state["first_min_var_res"] = False
if "first_min_var_max_rent" not in st.session_state:
    st.session_state["first_min_var_max_rent"] = False

# -------------------------------------------------------------------------------
# 5. Definir el prompt y configurar el chain
# -------------------------------------------------------------------------------
system_prompt_template = [
    ("system", """
    Actúa como un experto en finanzas y optimización de carteras de inversión.
    Tu objetivo es ayudar a los usuarios proporcionando información detallada y consejos sobre cómo optimizar una cartera de inversión, 
    teniendo en cuenta factores como el rendimiento esperado, el riesgo, la diversificación, y las restricciones del inversor 
    (por ejemplo, horizonte temporal, tolerancia al riesgo, liquidez, entre otros).
    Proporciona ejemplos prácticos y recursos relevantes cuando sea necesario, 
    y personaliza las respuestas según las necesidades y el contexto de cada usuario.
    """),
    ("placeholder", "{chat_history}"),
    ("user", "{question}")
]


prompt_response = ChatPromptTemplate.from_messages(system_prompt_template)
llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")

chain_response = prompt_response | llm | StrOutputParser()

# -------------------------------------------------------------------------------
# 6. Función para generar respuesta en streaming
# -------------------------------------------------------------------------------
def generar_respuesta_stream(chain, question, chat_history):
    """
    Llama al 'chain' (modelo) en modo streaming para una pregunta dada.
    'chat_history' es una lista de mensajes que sirve como contexto.
    Yields:
        token (str): Siguiente token generado por el modelo.
    """
    for token in chain.stream({"question": question, "chat_history": chat_history}):
        yield token
        time.sleep(0.01)

# -------------------------------------------------------------------------------
# 7. Funciones auxiliares
# -------------------------------------------------------------------------------
def build_chat_history_for_prompt(messages, max_turns=5):
    """
    Convierte los últimos 'max_turns' intercambios del historial en formato
    de ChatPrompt (HumanMessage y AIMessage).
    Retorna la lista de mensajes para usar como contexto en la llamada al modelo.
    """
    last_msgs = messages[-(max_turns*2):]  # cada turno consta de (usuario, asistente)
    chat_history = []
    for msg in last_msgs:
        if msg["role"] == "assistant":
            chat_history.append(AIMessage(content=msg["content"]))
        else:
            chat_history.append(HumanMessage(content=msg["content"]))
    return chat_history

def mostrar_logo(image_path, width=200):
    """
    Muestra un logo si existe el archivo en la ruta dada.
    """
    if os.path.exists(image_path):
        st.markdown('<div class="logo-container">', unsafe_allow_html=True)
        st.image(image_path, width=width)
        st.markdown('</div>', unsafe_allow_html=True)

def render_user_message(user_question):
    """
    Muestra el mensaje del usuario en el chat y lo agrega al historial.
    """
    with st.chat_message("user", avatar="👨‍💻"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

def render_assistant_response(question):
    """
    Realiza la llamada al modelo en streaming para una pregunta y
    muestra la respuesta del asistente en tiempo real.
    Al final, agrega la respuesta completa al historial de mensajes.
    """
    chat_history = build_chat_history_for_prompt(st.session_state["messages"])
    
    # Bloque de mensaje del asistente (streaming)
    with st.chat_message("assistant", avatar='🤖'):
        assistant_placeholder = st.empty()  # Placeholder para la respuesta
        partial_response = ""
        for token in generar_respuesta_stream(chain_response, question, chat_history):
            partial_response += token
            assistant_placeholder.markdown(partial_response)
    
    # Guardar la respuesta final del asistente en el historial
    st.session_state["messages"].append({"role": "assistant", "content": partial_response})

# -------------------------------------------------------------------------------
# 9. Interfaz principal del chat
# -------------------------------------------------------------------------------
def mostrar_chat():
    """
    Muestra la interfaz de chat con:
      - Logo (opcional)
      - Historial de mensajes
      - Input para escribir nuevas preguntas
    """
    # Mostrar logo en la parte superior (opcional)
    # mostrar_logo("Logo.png", width=200)

    # Renderizar historial de mensajes
    with st.container():
        for message in st.session_state.messages:
            avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
            # Si hay imagen en el mensaje, mostrarla
            if "image" in message:
                st.image(message["image"], use_container_width=True)

    # Lógica para la primera pregunta si la pantalla inicial ya se usó
    if st.session_state.first:
        # Renderizar la primera pregunta almacenada
        user_question = st.session_state.question
        render_assistant_response(user_question)
        st.session_state.first = False

    # Input para preguntas posteriores
    user_question = st.chat_input("Escribe tu pregunta...")
    if user_question:
        render_user_message(user_question)
        render_assistant_response(user_question)

# -------------------------------------------------------------------------------
# 8. Pantalla inicial
# -------------------------------------------------------------------------------

def mostrar_pantalla_inicial():

    # Lista de nombres de las empresas
    company_names = list(sp500_companies.keys())

   # Mostrar controles iniciales solo si no hay datos cargados
    if "loaded_companies" not in st.session_state:
        selected_companies = st.multiselect('Selecciona las empresas del S&P 500', company_names)
        start_date = st.date_input('Fecha de inicio', value=datetime(2020, 1, 1))
        end_date = st.date_input('Fecha de fin', value=datetime(2025, 12, 31))

        if st.button('Aceptar'):
            if not selected_companies:
                st.error('Por favor, selecciona al menos una empresa.')
            elif start_date >= end_date:
                st.error('La fecha de inicio debe ser anterior a la fecha de fin.')
            else:
                selected_tickers = [sp500_companies[company] for company in selected_companies]
                
                with st.spinner('Cargando datos...'):
                    stocks = yf.download(selected_tickers, start=start_date, end=end_date, auto_adjust=False)
                    stocks_adj = stocks["Adj Close"].dropna()
                    stocks_adj.index = pd.to_datetime(stocks_adj.index)
                    
                    st.session_state["stocks_data"] = stocks_adj
                    st.session_state["loaded_companies"] = selected_companies
                    st.session_state["initial_load"] = True  # Marcar carga inicial
                    st.rerun()  # Forzar actualización de la UI

    # Mostrar controles de filtrado y optimización después de carga inicial
    if "loaded_companies" in st.session_state:
        # Sección de filtrado de empresas
        filtered_companies = st.multiselect(
            'Filtrar empresas mostradas',
            options=st.session_state["loaded_companies"],
            default=st.session_state["loaded_companies"]
        )
        
        # Actualizar gráficos con el filtrado
        plt.style.use('seaborn-v0_8-whitegrid')   # Alternativa si seaborn no está disponible
        sns.set_palette("Dark2")  # Paleta de colores atractiva

        if filtered_companies:
            stocks_adj = st.session_state["stocks_data"]
            selected_tickers = [sp500_companies[company] for company in filtered_companies]

            # Gráfico 1 - Precios ajustados
            fig1, ax1 = plt.subplots(figsize=(14, 7))
            for ticker in selected_tickers:
                ax1.plot(stocks_adj.index, stocks_adj[ticker], label=ticker, lw=2)  # Líneas más visibles
            ax1.set_title("Precios Ajustados de Acciones", fontsize=16, pad=15)  # Título más grande y espaciado
            ax1.set_xlabel("Fecha", fontsize=16)  # Etiqueta eje x
            ax1.set_ylabel("Precio Ajustado", fontsize=16)  # Etiqueta eje y
            # Leyenda adaptable
            if len(selected_tickers) > 10:
                # Para muchas empresas, usa una leyenda con varias columnas
                ncol = (len(selected_tickers) // 10) + 1
                ax1.legend(loc="upper left", fontsize=12, ncol=ncol, bbox_to_anchor=(0, -0.1))
                plt.subplots_adjust(bottom=0.2)  # Dar espacio a la leyenda
            else:
                ax1.legend(loc="best", fontsize=14)
            ax1.grid(True, alpha=0.3)  # Grilla ligera
            st.pyplot(fig1)

            # Gráfico 2 - Porcentaje de cambio
            fig2, ax2 = plt.subplots(figsize=(14, 7))
            initial_prices = stocks_adj.iloc[0][selected_tickers]
            percentage_change = (stocks_adj[selected_tickers] - initial_prices) / initial_prices * 100
            for ticker in selected_tickers:
                ax2.plot(percentage_change.index, percentage_change[ticker], label=ticker, lw=2)  # Líneas más visibles
            ax2.set_title("Porcentaje de Cambio en Precios de Acciones", fontsize=16, pad=15)  # Título descriptivo
            ax2.set_xlabel("Fecha", fontsize=16)  # Etiqueta eje x
            ax2.set_ylabel("Porcentaje de Cambio (%)", fontsize=16)  # Etiqueta eje y
            # Leyenda adaptable
            if len(selected_tickers) > 10:
                # Para muchas empresas, usa una leyenda con varias columnas
                ncol = (len(selected_tickers) // 10) + 1
                ax2.legend(loc="upper left", fontsize=12, ncol=ncol, bbox_to_anchor=(0, -0.1))
                plt.subplots_adjust(bottom=0.2)  # Dar espacio a la leyenda
            else:
                ax2.legend(loc="best", fontsize=14)
            ax2.grid(True, alpha=0.3)  # Grilla ligera
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))  # Formato de porcentaje
            st.pyplot(fig2)

        # Selector de optimización (frutas)
        opciones_optimizacion = ["matriz de covarianza",
                                "Portafolio de mínima varianza",
                                "Portafolio de mínima varianza con restricciones",
                                "Portafolio considerando retorno y riesgo simultáneamente"]
        seleccion = st.selectbox(
            'Selecciona el tipo de optimización',
            options=opciones_optimizacion,
            index=0
        )

        if seleccion == "matriz de covarianza" and "stocks_data" in st.session_state:
            if filtered_companies:  # Solo si hay empresas filtradas
                # Obtener tickers de las empresas filtradas
                selected_tickers = [sp500_companies[company] for company in filtered_companies]
                
                # Filtrar los datos solo para las empresas seleccionadas
                stocks_filtered = st.session_state["stocks_data"][selected_tickers]
                
                try:
                    # Calcula retornos logarítmicos diarios
                    returns = stocks_filtered.pct_change().dropna()
                    
                    # Calcular matriz de covarianzas anualizada
                    cov_matrix = returns.cov() * 252
                    
                    # Crear y mostrar el heatmap
                    fig_cov = plt.figure(figsize=(12, 8))
                    sns.heatmap(
                        cov_matrix,
                        annot=True,
                        fmt=".6f",
                        cmap="coolwarm",
                        xticklabels=filtered_companies,  # Usar nombres completos filtrados
                        yticklabels=filtered_companies,
                        cbar_kws={'label': 'Covarianza Anualizada'}
                    )
                    plt.title("Matriz de Covarianzas (Empresas Filtradas)")
                    plt.xticks(rotation=45)
                    plt.yticks(rotation=0)
                    st.session_state.fig_cov = fig_cov
                    st.pyplot(fig_cov)

                    cov_first = f'''
                    Como experto en finanzas cuantitativas, analiza la matriz de covarianza anualizada de estos activos: {filtered_companies}. 

                    La matriz obtenida es:
                    {cov_matrix}

                    Proporciona una explicación estructurada que incluya:
                    1. **Interpretación general**: 
                    - Explica qué miden los valores de covarianza en este contexto
                    - Menciona cómo la anualización (multiplicar por 252 días) afecta la interpretación

                    2. **Relaciones clave**:
                    - Identifica los 3 pares con mayor covarianza positiva y su posible interpretación
                    - Identifica los 3 pares con mayor covarianza negativa y su significado
                    - Destaca cualquier patrón sectorial o de mercado observable

                    3. **Análisis por activo**:
                    - Para cada activo en {filtered_companies}:
                        * Explica a que se dedican
                        * Menciona con qué otro activo tiene mayor relación positiva
                        * Menciona con qué otro activo tiene mayor relación negativa
                        * Explica posibles razones fundamentales (sector, mercado, etc.)

                    4. **Implicaciones prácticas**:
                    - ¿Qué pares podrían ofrecer mejores oportunidades de diversificación?
                    - ¿Qué activos podrían funcionar como cobertura natural entre sí?
                    - ¿Qué sectores/momentos de mercado sugiere esta estructura de covarianzas?

                    5. **Advertencias y limitaciones**:
                    - Menciona 3 consideraciones importantes al interpretar covarianzas
                    - Explica cómo se relaciona esto con el coeficiente de correlación
                    - Precaución sobre la estabilidad temporal de estas relaciones

                    Formato requerido:
                    - Lenguaje claro y accesible para inversores no técnicos
                    - Usar ejemplos específicos de la matriz presentada
                    - Incluir recomendaciones accionables para gestión de portafolio
                    - Destacar conclusiones sorprendentes o contra-intuitivas
                    '''
                    if st.session_state.first == False and st.session_state.first_cov_mat == False:
                        st.session_state.question = cov_first
                        st.session_state.first = True
                        st.session_state.first_cov_mat = True
                    
                except Exception as e:
                    st.error(f"Error calculando la matriz de covarianza: {str(e)}")
            else:
                st.warning("Selecciona al menos una empresa para filtrar")
            
        elif seleccion == "Portafolio de mínima varianza" and "stocks_data" in st.session_state:
            if filtered_companies:
                try:
                    # Obtener datos filtrados
                    selected_tickers = [sp500_companies[company] for company in filtered_companies]
                    stocks_filtered = st.session_state["stocks_data"][selected_tickers]
                    
                    # Calcular retornos logarítmicos
                    returns = returns = stocks_filtered.pct_change().dropna()

                    if len(returns) < 2:
                        st.error("No hay suficientes datos para calcular el portafolio")
                        return
                    
                    # Configurar y entrenar el modelo
                    model = MeanRisk(
                        risk_measure=RiskMeasure.VARIANCE,
                        portfolio_params=dict(name="Portafolio Mínima Varianza")
                    )

                    model.fit(returns)
                    
                    # Extraer los pesos (usar predict para obtener el portafolio)
                    portfolio = model.predict(returns)
                    weights = portfolio.weights
                    
                    # Calcular métricas manualmente
                    cov_matrix = returns.cov() * 252
                    
                    sns.set_palette("viridis")  # Paleta de colores atractiva

                    # Crear DataFrame con los pesos
                    weights_df = pd.DataFrame({
                        'Empresa': filtered_companies,
                        'Ticker': selected_tickers,
                        'Peso (%)': np.round(weights * 100, 2)
                    }).sort_values('Peso (%)', ascending=False)

                    # Gráfico de barras horizontales
                    fig, ax = plt.subplots(figsize=(12, 8))  # Tamaño más grande para mejor legibilidad
                    bars = ax.barh(
                        weights_df['Ticker'], 
                        weights_df['Peso (%)'], 
                        height=0.6,  # Barras más delgadas para un look más elegante
                        color=sns.color_palette("viridis", len(weights_df)),  # Colores degradados
                        edgecolor='black',  # Bordes para mejor definición
                        alpha=0.9  # Ligera transparencia para suavidad
                    )

                    # Configuraciones del gráfico
                    ax.set_xlabel('Porcentaje (%)', fontsize=14)
                    ax.set_ylabel('Ticker', fontsize=14)
                    ax.set_title('Distribución de Pesos del Portafolio', fontsize=16, pad=20)
                    ax.set_xlim(0, max(weights_df['Peso (%)'].max() * 1.1, 100))  # Ajustar límite dinámicamente
                    ax.grid(axis='x', linestyle='--', alpha=0.5)  # Grilla más suave

                    # Añadir etiquetas de valor dentro de las barras
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(
                            width - 3 if width > 10 else width + 1,  # Ajustar posición para mejor visibilidad
                            bar.get_y() + bar.get_height()/2,
                            f'{width:.1f}%',
                            va='center',
                            ha='right' if width > 10 else 'left',
                            color='white' if width > 20 else 'black',  # Ajustar umbral para mejor contraste
                            fontweight='bold',
                            fontsize=16
                        )

                    # Invertir eje Y para mostrar mayor valor arriba
                    plt.gca().invert_yaxis()
                    plt.tight_layout()

                    # Mostrar en Streamlit
                    st.pyplot(fig)

                    # Configurar el estilo visual para un diseño más profesional
                    plt.style.use('seaborn-v0_8-whitegrid')

                    # Calcular pesos equiponderados
                    n_assets = len(filtered_companies)
                    equal_weights = np.array([1/n_assets] * n_assets)

                    # Calcular retornos acumulados de ambos portafolios
                    returns = stocks_filtered.pct_change().dropna()

                    # Portafolio de mínima varianza
                    min_var_returns = (returns * weights).sum(axis=1)
                    min_var_cumulative = (1 + min_var_returns).cumprod()

                    # Portafolio equiponderado
                    equal_returns = (returns * equal_weights).sum(axis=1)
                    equal_cumulative = (1 + equal_returns).cumprod()

                    # Calcular la rentabilidad acumulada en porcentaje
                    rentabilidad_acumulada_min_var = (min_var_cumulative - 1) * 100
                    rentabilidad_acumulada_equal = (equal_cumulative - 1) * 100

                    # Crear figura comparativa con dimensiones óptimas
                    fig_comp, ax_comp = plt.subplots(figsize=(14, 8), dpi=100)

                    # Colores mejorados
                    color_min_var = '#4C72B0'  # Azul elegante para mínima varianza
                    color_equal = 'black'      # Negro para equiponderado

                    # Calcular la varianza del portafolio de mínima varianza
                    variance_min_var = np.dot(weights.T, np.dot(cov_matrix, weights))
                    
                    # Calcular la varianza del portafolio equiponderado
                    variance_equal = np.dot(equal_weights.T, np.dot(cov_matrix, equal_weights))

                    # Graficar las series con mayor grosor y mejor diseño
                    ax_comp.plot(rentabilidad_acumulada_min_var.index, rentabilidad_acumulada_min_var,
                                label=f'Portafolio de Mínima Varianza (σ² = {variance_min_var:.4f})',
                                linewidth=2.5,
                                color=color_min_var)

                    ax_comp.plot(rentabilidad_acumulada_equal.index, rentabilidad_acumulada_equal,
                                label=f'Portafolio Equiponderado (σ² = {variance_equal:.4f})',
                                linewidth=1,
                                color=color_equal,
                                linestyle='--')  # Línea discontinua para el portafolio equiponderado

                    # Mejorar la apariencia del gráfico
                    ax_comp.set_title('Comparación de Rentabilidad Acumulada: Mínima Varianza vs Equiponderado',
                                    fontsize=22,
                                    fontweight='bold',
                                    pad=20)

                    ax_comp.set_xlabel('Fecha', fontsize=18, fontweight='medium', labelpad=10)
                    ax_comp.set_ylabel('Rentabilidad Acumulada (%)', fontsize=18, fontweight='medium', labelpad=10)

                    # Formatear el eje Y para mostrar porcentajes correctamente
                    ax_comp.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}%'))

                    # Mejorar la leyenda
                    ax_comp.legend(fontsize=14,
                                loc='best',
                                frameon=True,
                                framealpha=0.9,
                                edgecolor='lightgray')

                    # Añadir grid sutil
                    ax_comp.grid(True, linestyle='--', alpha=0.6)

                    # Mejorar márgenes y ticks
                    plt.xticks(fontsize=14, rotation=45)  # Rotar fechas para mejor visualización
                    plt.yticks(fontsize=14)

                    # Ajustar bordes y espaciado
                    plt.tight_layout()

                    # Mostrar en Streamlit
                    st.pyplot(fig_comp)
                    
                    st.session_state.fig_port_bar = fig
                    st.session_state.fig_port_comp = fig_comp

                    portfolio_prompt = f'''
                    Como estratega cuantitativo senior, analiza este portafolio de mínima varianza para los activos.
                    Donde los activos son: {weights_df['Ticker'].tolist()} y sus pesos respectivamente {weights_df['Peso (%)'].tolist()}
                    Explica antes que es un portafolio de mínima varianza y porque es importante.
                    
                    Por si te sirve en tu anális la matriz de covarianzas es:
                    {cov_matrix}

                    Proporciona un análisis estructurado que incluya:

                    1. **Composición del portafolio**:
                    - Explica por qué los 3 activos con mayor peso son: {weights_df['Ticker'].head(3).tolist()} con un peso respectivamente de: {weights_df['Peso (%)'].head(3).tolist()} dominan la asignación
                    - Identifica qué características comunes (volatilidad, correlaciones, sector) comparten
                    - Analiza los 2 activos con menor peso son: {weights_df['Ticker'].tail(2).tolist()} con un peso respectivamente de: {weights_df['Peso (%)'].tail(2).tolist()} y su impacto marginal

                    2. **Análisis de riesgo-rendimiento**:
                    - Compara la distribución de pesos vs. portafolio equiponderado
                    - Explica las diferencias en rentabilidad acumulada mostradas en el gráfico:
                        * ¿En qué condiciones de mercado supera uno al otro?
                        * ¿Qué eventos relevantes ocurrieron en los picos/valles del gráfico?

                    3. **Dinámicas sectoriales**:
                    - Agrupa los activos en sectores y explica la actividad de cada activo
                    - Identifica concentración sectorial de los activos
                    - ¿Qué sectores están sobrerrepresentados/subrepresentados?
                    - ¿Cómo afecta esto al riesgo sistémico del portafolio?


                    Formato requerido:
                    - Lenguaje claro y accesible para inversores no técnicos
                    - Incluir recomendaciones accionables para gestión de portafolio
                    - Destacar conclusiones sorprendentes o contra-intuitivas
                    - Usar comparaciones concretas con valores numéricos del gráfico (ej: "X activo con Y% peso vs Z% promedio")
                    - Destacar insights contraintuitivos (ej: "Aunque [activo] muestra alta volatilidad, su baja correlación...")
                    '''

                    if st.session_state.first == False and st.session_state.first_min_var == False:
                        st.session_state.question = portfolio_prompt
                        st.session_state.first = True
                        st.session_state.first_min_var = True

                except Exception as e:
                    st.error(f"Error calculando el portafolio: {str(e)}")
            
            else:
                st.warning("Selecciona al menos una empresa para calcular el portafolio")

        elif seleccion == "Portafolio de mínima varianza con restricciones" and "stocks_data" in st.session_state:
            if filtered_companies:
                try:
                    # Obtener datos filtrados
                    selected_tickers = [sp500_companies[company] for company in filtered_companies]
                    stocks_filtered = st.session_state["stocks_data"][selected_tickers]

                    # Calcular retornos logarítmicos
                    returns = returns = stocks_filtered.pct_change().dropna()
                    

                    if len(returns) < 2:
                        st.error("No hay suficientes datos para calcular el portafolio")
                        return
                    
                    # Verificar que hay activos seleccionados
                    n_assets = len(selected_tickers)
                    if n_assets == 0:
                        st.error("No hay activos seleccionados")
                        return

                    # Calcular el peso mínimo permitido (1/n_assets)
                    min_permitted_weight = 1.0 / n_assets

                    # Determinar el valor por defecto según la condición
                    default_value = 0.3 if min_permitted_weight < 0.2 else min_permitted_weight + 0.05

                    # Asegurar que el valor por defecto no exceda el máximo (1.0)
                    default_value = min(default_value, 1.0)

                    # Selector deslizante para la restricción máxima
                    max_weight = st.slider(
                        "Restricción máxima de peso por activo",
                        min_value=min_permitted_weight,  # Mínimo técnicamente viable
                        max_value=1.0,  # Máximo teórico
                        value=default_value,  # Valor por defecto (ajustado automáticamente si no es válido)
                        step=0.01,
                    )

                    # Optimizar el portafolio para mínima varianza con restricción
                    model_resct = MeanRisk(
                        risk_measure=RiskMeasure.VARIANCE,  # Minimizar varianza
                        max_weights=max_weight,
                        min_weights=0,
                        portfolio_params=dict(name="Minimum Variance Portfolio With Weight Restriction")
                    )
                    
                    model_resct.fit(returns)

                    # Obtener resultados
                    portfolio_resct = model_resct.predict(returns)
                    weights_resct = portfolio_resct.weights
                    
                    # Calcular métricas manualmente
                    cov_matrix = returns.cov() * 252

                    sns.set_palette("viridis")  # Paleta de colores atractiva

                    # Crear DataFrame con los pesos
                    weights_resct_df = pd.DataFrame({
                        'Empresa': filtered_companies,
                        'Ticker': selected_tickers,
                        'Peso (%)': np.round(weights_resct * 100, 2)
                    }).sort_values('Peso (%)', ascending=False)

                    # Gráfico de barras horizontales
                    fig, ax = plt.subplots(figsize=(12, 8))  # Tamaño más grande para mejor legibilidad
                    bars = ax.barh(
                        weights_resct_df['Ticker'], 
                        weights_resct_df['Peso (%)'], 
                        height=0.6,  # Barras más delgadas para un look más elegante
                        color=sns.color_palette("viridis", len(weights_resct_df)),  # Colores degradados
                        edgecolor='black',  # Bordes para mejor definición
                        alpha=0.9  # Ligera transparencia para suavidad
                    )

                    # Configuraciones del gráfico
                    ax.set_xlabel('Porcentaje (%)', fontsize=14)
                    ax.set_ylabel('Ticker', fontsize=14)
                    ax.set_title('Distribución de Pesos del Portafolio', fontsize=16, pad=20)
                    ax.set_xlim(0, max(weights_resct_df['Peso (%)'].max() * 1.1, 100))  # Ajustar límite dinámicamente
                    ax.grid(axis='x', linestyle='--', alpha=0.5)  # Grilla más suave

                    # Añadir etiquetas de valor dentro de las barras
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(
                            width - 3 if width > 10 else width + 1,  # Ajustar posición para mejor visibilidad
                            bar.get_y() + bar.get_height()/2,
                            f'{width:.1f}%',
                            va='center',
                            ha='right' if width > 10 else 'left',
                            color='white' if width > 20 else 'black',  # Ajustar umbral para mejor contraste
                            fontweight='bold',
                            fontsize=16
                        )

                    # Invertir eje Y para mostrar mayor valor arriba
                    plt.gca().invert_yaxis()
                    plt.tight_layout()

                    # Mostrar en Streamlit
                    st.pyplot(fig)

                    # Configurar el estilo visual para un diseño más profesional
                    plt.style.use('seaborn-v0_8-whitegrid')

                    # Configurar y entrenar el modelo
                    model = MeanRisk(
                        risk_measure=RiskMeasure.VARIANCE,
                        portfolio_params=dict(name="Portafolio Mínima Varianza")
                    )

                    model.fit(returns)
                    
                    # Extraer los pesos (usar predict para obtener el portafolio)
                    portfolio = model.predict(returns)
                    weights = portfolio.weights

                    # Crear DataFrame con los pesos
                    weights_df = pd.DataFrame({
                        'Empresa': filtered_companies,
                        'Ticker': selected_tickers,
                        'Peso (%)': np.round(weights * 100, 2)
                    }).sort_values('Peso (%)', ascending=False)

                    # Calcular pesos equiponderados
                    n_assets = len(filtered_companies)
                    equal_weights = np.array([1/n_assets] * n_assets)

                    # Calcular retornos acumulados de ambos portafolios
                    returns = stocks_filtered.pct_change().dropna()

                    # Portafolio de mínima varianza
                    min_var_returns = (returns * weights).sum(axis=1)
                    min_var_cumulative = (1 + min_var_returns).cumprod()

                    # Portafolio de mínima varianza con restricciones
                    min_var_resct_returns = (returns * weights_resct).sum(axis=1)
                    min_var_resct_cumulative = (1 + min_var_resct_returns).cumprod()

                    # Portafolio equiponderado
                    equal_returns = (returns * equal_weights).sum(axis=1)
                    equal_cumulative = (1 + equal_returns).cumprod()

                    # Calcular la rentabilidad acumulada en porcentaje
                    rentabilidad_acumulada_min_var = (min_var_cumulative - 1) * 100
                    rentabilidad_acumulada_equal = (equal_cumulative - 1) * 100
                    rentabilidad_acumulada_resct = (min_var_resct_cumulative - 1) * 100

                    # Crear figura comparativa con dimensiones óptimas
                    fig_comp, ax_comp = plt.subplots(figsize=(14, 8), dpi=100)

                    # Colores mejorados
                    color_equal = 'black'  # Negro para mínima varianza
                    color_resct = '#4C72B0'  # Azul elegante
                    color_min_var = '#C44E52' 

                    # Calcular la varianza del portafolio de mínima varianza
                    variance_min_var = np.dot(weights.T, np.dot(cov_matrix, weights))
                    
                    # Calcular la varianza del portafolio equiponderado
                    variance_equal = np.dot(equal_weights.T, np.dot(cov_matrix, equal_weights))
                    
                    # Calcular la varianza del portafolio de mínima varianza con restricción 
                    variance_min_var_constrained = np.dot(weights_resct.T, np.dot(cov_matrix, weights_resct))
                    
                    # Graficar las series con mayor grosor y mejor diseño
                    ax_comp.plot(rentabilidad_acumulada_min_var.index, rentabilidad_acumulada_min_var,
                                label=f'Portafolio de Mínima Varianza (σ² = {variance_min_var:.4f})',
                                linewidth=1,
                                color=color_min_var)

                    ax_comp.plot(rentabilidad_acumulada_equal.index, rentabilidad_acumulada_equal,
                                label=f'Portafolio Equiponderado (σ² = {variance_equal:.4f})',
                                linewidth=1,
                                color=color_equal,
                                linestyle='--')  # Línea discontinua para el portafolio equiponderado

                    ax_comp.plot(rentabilidad_acumulada_resct.index, rentabilidad_acumulada_resct, 
                                label=f'Portafolio con Restricciones (σ² = {variance_min_var_constrained:.4f})', 
                                linewidth=2.5, 
                                color=color_resct)
                    
                    # Mejorar la apariencia del gráfico
                    ax_comp.set_title('Comparaciones de Rentabilidad Acumulada Mínima Varianza', 
                                fontsize=22, 
                                fontweight='bold', 
                                pad=20)

                    ax_comp.set_xlabel('Fecha', fontsize=18, fontweight='medium', labelpad=10)
                    ax_comp.set_ylabel('Rentabilidad Acumulada (%)', fontsize=18, fontweight='medium', labelpad=10)

                    # Formatear el eje Y para mostrar porcentajes correctamente
                    ax_comp.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}%'))

                    # Mejorar la leyenda
                    ax_comp.legend(fontsize=14,
                                loc='best',
                                frameon=True,
                                framealpha=0.9,
                                edgecolor='lightgray')

                    # Añadir grid sutil
                    ax_comp.grid(True, linestyle='--', alpha=0.6)

                    # Mejorar márgenes y ticks
                    plt.xticks(fontsize=14, rotation=45)  # Rotar fechas para mejor visualización
                    plt.yticks(fontsize=14)

                    # Ajustar bordes y espaciado
                    plt.tight_layout()

                    # Mostrar en Streamlit
                    st.pyplot(fig_comp)

                    portfolio_prompt_constrained = f'''
                    Como estratega cuantitativo senior, analiza este portafolio de mínima varianza con restricciones de pesos para los activos de {max_weight*100:.2f}%.  
                    Donde los activos son: {weights_resct_df['Ticker'].tolist()} y sus pesos respectivamente {weights_resct_df['Peso (%)'].tolist()}.  
                    Explica primero qué es un portafolio de mínima varianza con restricciones de pesos y por qué es importante en la gestión de inversiones.

                    Por si te sirve en tu análisis, la matriz de covarianzas es:  
                    {cov_matrix}

                    Proporciona un análisis estructurado que incluya:

                    ### 1. Composición del portafolio con restricciones
                    - Compara la distribución de pesos del portafolio con restricciones con:
                    - El portafolio de mínima varianza sin restricciones (pesos: {weights_df['Peso (%)'].tolist()}).
                    - El portafolio equiponderado (pesos uniformes de {100/n_assets:.2f}% por activo).
                    - Analiza cómo la restricción de peso máximo ({max_weight*100:.2f}%) afecta la asignación de los activos.  
                    - Identifica qué activos alcanzan el límite máximo y cuáles se ven más reducidos respecto al portafolio sin restricciones.  
                    - Ejemplo: "El activo X tiene un peso de Y% con restricciones vs. Z% sin restricciones".

                    ### 2. Análisis de riesgo-rendimiento
                    - Compara la varianza del portafolio con restricciones ({variance_min_var_constrained:.4f}) con:
                    - La varianza del portafolio sin restricciones ({variance_min_var:.4f}).
                    - La varianza del portafolio equiponderado ({variance_equal:.4f}).
                    - Analiza las diferencias en la rentabilidad acumulada mostradas en el gráfico:
                    - ¿En qué condiciones de mercado el portafolio con restricciones podría superar al sin restricciones o al equiponderado? Por ejemplo, ¿durante periodos de alta volatilidad o caídas bruscas?
                    - ¿Qué patrones se observan en los picos y valles de las curvas de rentabilidad acumulada?

                    ### 3. Impacto de la restricción en la diversificación
                    - Evalúa si la restricción de peso máximo mejora o empeora la diversificación del portafolio.  
                    - Compara la concentración de pesos entre los tres portafolios (con restricciones, sin restricciones y equiponderado).  
                    - Analiza si la restricción provoca una sobrerrepresentación o subrepresentación de ciertos activos o sectores.  
                    - Discute los pros y contras de imponer un límite máximo de peso en términos de riesgo y estabilidad.

                    ### Formato requerido
                    - Usa un lenguaje claro y accesible para inversores no técnicos.  
                    - Incluye recomendaciones prácticas sobre cuándo usar un portafolio con restricciones de pesos (ejemplo: "Es útil en mercados volátiles para evitar concentración excesiva").  
                    - Destaca insights específicos con valores numéricos (ejemplo: "La varianza del portafolio con restricciones es un X% mayor/menor que la del sin restricciones").  
                    - Resalta resultados contraintuitivos si los hay (ejemplo: "Aunque el activo X tiene alta volatilidad, su peso no se reduce tanto por su baja correlación").  
                    '''

                    if st.session_state.first == False and st.session_state.first_min_var_res == False:
                        st.session_state.question = portfolio_prompt_constrained
                        st.session_state.first = True
                        st.session_state.first_min_var_res = True

                except Exception as e:
                    st.error(f"Error calculando el portafolio: {str(e)}")

        elif seleccion == "Portafolio considerando retorno y riesgo simultáneamente" and "stocks_data" in st.session_state:
            if filtered_companies:
                try:
                    # Obtener datos filtrados
                    selected_tickers = [sp500_companies[company] for company in filtered_companies]
                    stocks_filtered = st.session_state["stocks_data"][selected_tickers]

                    # Calcular retornos logarítmicos
                    returns = stocks_filtered.pct_change().dropna()
                    
                    if len(returns) < 2:
                        st.error("No hay suficientes datos para calcular el portafolio")
                        return
                    
                    # Verificar que hay activos seleccionados
                    n_assets = len(selected_tickers)
                    if n_assets == 0:
                        st.error("No hay activos seleccionados")
                        return

                    # Selector deslizante para la tasa libre de riesgo
                    risk_free_rate = st.slider(
                        "Tasa libre de riesgo (%)",
                        min_value=0.0,  # Mínimo técnicamente viable
                        max_value=6.0,  # Máximo teórico
                        value=0.0,  # Valor por defecto
                        step=0.1,
                    ) / 100  # Convertir de porcentaje a decimal

                    # Selector deslizante para la tasa libre de riesgo
                    portfolio_num = st.slider(
                        "Número de carteras a calcular",
                        min_value=5,  # Mínimo técnicamente viable
                        max_value=100,  # Máximo teórico
                        value=30,  # Valor por defecto
                        step=1,
                    )  # Convertir de porcentaje a decimal
                    
                    # Dividir los datos en entrenamiento y prueba (70/30)
                    train_size = int(len(returns) * 0.7)
                    X_train = returns.iloc[:train_size]
                    X_test = returns.iloc[train_size:]
                    
                    # Crear el modelo de optimización con más puntos en la frontera eficiente
                    model = MeanRisk(
                        risk_measure=RiskMeasure.VARIANCE,
                        efficient_frontier_size=portfolio_num,  # Más carteras para una curva más suave
                        portfolio_params=dict(name="Variance"),
                        risk_free_rate=risk_free_rate  # Usar la tasa definida por el usuario
                    )

                    # Ajustar el modelo a los datos de entrenamiento
                    model.fit(X_train)

                    # Predecir las composiciones de las carteras para el conjunto de entrenamiento
                    population_train = model.predict(X_train)

                    # Extraer las medidas de riesgo y rentabilidad para graficar
                    risks = population_train.measures(measure=RiskMeasure.ANNUALIZED_STANDARD_DEVIATION)
                    returns_annual = population_train.measures(measure=PerfMeasure.ANNUALIZED_MEAN)
                    sharpe_ratios = population_train.measures(measure=RatioMeasure.ANNUALIZED_SHARPE_RATIO)
                    
                    # Encontrar el índice del máximo Sharpe Ratio
                    max_sharpe_idx = np.argmax(sharpe_ratios)
                    
                    # Mostrar una descripción general del análisis
                    st.subheader("Resumen de la Optimización de Portafolio")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label="Máximo Ratio de Sharpe",
                            value=f"{sharpe_ratios[max_sharpe_idx]:.3f}"
                        )
                    
                    with col2:
                        st.metric(
                            label="Rentabilidad Anualizada",
                            value=f"{returns_annual[max_sharpe_idx]:.2%}"
                        )
                        
                    with col3:
                        st.metric(
                            label="Riesgo Anualizado",
                            value=f"{risks[max_sharpe_idx]:.2%}"
                        )
                    
                    # Crear pestañas para los diferentes gráficos
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "Frontera Eficiente", 
                        "Composición del Portafolio", 
                        "Rendimiento Acumulado",
                        "Cartera Óptima"
                    ])
                    
                    with tab1:
                        # GRÁFICO 1: FRONTERA EFICIENTE MEJORADA
                        fig = go.Figure()

                        # Agregar la curva de la frontera eficiente con gradiente de color
                        fig.add_trace(
                            go.Scatter(
                                x=risks,
                                y=returns_annual,
                                mode="lines+markers",
                                marker=dict(
                                    size=8,
                                    color=sharpe_ratios,
                                    colorscale="Viridis",
                                    showscale=True,
                                    colorbar=dict(
                                        title=dict(
                                            text="Ratio de Sharpe",
                                            font=dict(size=14, family="Arial, sans-serif")
                                        ),
                                        tickfont=dict(size=12, family="Arial, sans-serif")
                                    ),
                                    line=dict(width=1, color="white")
                                ),
                                line=dict(color='rgba(100, 110, 250, 0.6)', width=2),
                                text=[f"Sharpe: {s:.3f}<br>Riesgo: {r:.2%}<br>Retorno: {ret:.2%}" 
                                    for s, r, ret in zip(sharpe_ratios, risks, returns_annual)],
                                hovertemplate="<b>Cartera</b><br>%{text}<extra></extra>",
                            )
                        )

                        # Destacar el punto con máximo Sharpe en rojo
                        fig.add_trace(
                            go.Scatter(
                                x=[risks[max_sharpe_idx]],
                                y=[returns_annual[max_sharpe_idx]],
                                mode="markers",
                                marker=dict(
                                    symbol="star",
                                    color="#E50914",  # Rojo brillante
                                    size=16,
                                    line=dict(width=2, color="white")
                                ),
                                text=[f"<b>MÁXIMO SHARPE: {sharpe_ratios[max_sharpe_idx]:.3f}</b><br>Riesgo: {risks[max_sharpe_idx]:.2%}<br>Retorno: {returns_annual[max_sharpe_idx]:.2%}"],
                                hovertemplate="%{text}<extra></extra>",
                                name="Máximo Sharpe Ratio"
                            )
                        )

                        # Configurar los ejes y el título con mejor diseño
                        fig.update_layout(
                            title=dict(
                                text="Frontera Eficiente de Carteras",
                                font=dict(size=24, family="Arial, sans-serif", color="#2F4F4F")
                            ),
                            xaxis=dict(
                                title=dict(
                                    text="Desviación Estándar Anualizada",
                                    font=dict(size=16, family="Arial, sans-serif")
                                ),
                                tickformat=".1%",
                                gridcolor='rgba(211, 211, 211, 0.6)',
                                zeroline=False
                            ),
                            yaxis=dict(
                                title=dict(
                                    text="Rentabilidad Anualizada",
                                    font=dict(size=16, family="Arial, sans-serif")
                                ),
                                tickformat=".1%",
                                gridcolor='rgba(211, 211, 211, 0.6)',
                                zeroline=False
                            ),
                            showlegend=True,
                            legend=dict(
                                x=0.02,
                                y=0.98,
                                bgcolor='rgba(255, 255, 255, 0.8)',
                                bordercolor='rgba(211, 211, 211, 0.8)'
                            ),
                            plot_bgcolor='white',
                            height=600,
                            margin=dict(l=60, r=60, t=80, b=60),
                            shapes=[
                                # Línea para indicar el rendimiento libre de riesgo
                                dict(
                                    type="line",
                                    xref="x",
                                    yref="y",
                                    x0=0,
                                    y0=risk_free_rate,  # Tasa libre de riesgo del usuario
                                    x1=risks[max_sharpe_idx],
                                    y1=returns_annual[max_sharpe_idx],
                                    line=dict(
                                        color="rgba(0, 150, 136, 0.7)",
                                        width=2,
                                        dash="dash",
                                    ),
                                )
                            ],
                            annotations=[
                                dict(
                                    x=risks[max_sharpe_idx]/2,
                                    y=(returns_annual[max_sharpe_idx] + risk_free_rate)/2,
                                    text="Línea de Mercado de Capitales",
                                    showarrow=True,
                                    arrowhead=2,
                                    arrowcolor="rgba(0, 150, 136, 0.7)",
                                    ax=40,
                                    ay=-40,
                                    font=dict(
                                        family="Arial, sans-serif",
                                        size=12,
                                        color="rgba(0, 150, 136, 1)"
                                    ),
                                )
                            ]
                        )

                        # Mostrar el gráfico en Streamlit
                        st.plotly_chart(fig, use_container_width=True)
                        
                    
                    with tab2:
                        # GRÁFICO 2: COMPOSICIÓN DE CARTERAS MEJORADO
                        # Obtener los pesos de todas las carteras
                        weights_all = np.array([portfolio.weights for portfolio in population_train])
                        
                        # Obtener los nombres de los activos
                        asset_names = X_train.columns.tolist()
                        
                        # Crear una paleta de colores más atractiva
                        colors = plt.cm.tab20(np.linspace(0, 1, len(asset_names)))
                        colors = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b, _ in colors]
                        
                        # Ordenar por riesgo (eje x)
                        sort_indices = np.argsort(risks)
                        sorted_risks = risks[sort_indices]
                        sorted_weights = weights_all[sort_indices]
                        
                        # Posiciones para el eje x
                        x = np.arange(len(sorted_risks))
                        
                        # Crear el gráfico de composiciones
                        fig_comp = go.Figure()
                        
                        # Agregar cada activo como una capa apilada
                        for i, asset in enumerate(asset_names):
                            asset_weights = sorted_weights[:, i]
                            fig_comp.add_trace(
                                go.Bar(
                                    x=x,
                                    y=asset_weights,
                                    name=asset,
                                    marker_color=colors[i % len(colors)],
                                    hovertemplate=f"{asset}: %{{y:.2%}}<extra></extra>",
                                )
                            )
                        
                        # Marcar la cartera con máximo Sharpe
                        max_sharpe_pos = np.where(sort_indices == max_sharpe_idx)[0][0]
                        fig_comp.add_trace(
                            go.Scatter(
                                x=[max_sharpe_pos],
                                y=[1.05],
                                mode="markers+text",
                                marker=dict(
                                    symbol="triangle-down",
                                    color="#E50914",
                                    size=16
                                ),
                                text=["Máximo Sharpe"],
                                textposition="top center",
                                hoverinfo="none",
                                showlegend=False
                            )
                        )
                        
                        # Configurar el diseño del gráfico
                        fig_comp.update_layout(
                            title=dict(
                                text="Composición de las Carteras de la Frontera Eficiente",
                                font=dict(size=24, family="Arial, sans-serif", color="#2F4F4F")
                            ),
                            xaxis=dict(
                                title="Carteras (ordenadas por riesgo creciente)",
                                tickvals=[],
                                showgrid=False
                            ),
                            yaxis=dict(
                                title="Ponderación en la Cartera",
                                tickformat=".0%",
                                range=[0, 1.1],
                                gridcolor='rgba(211, 211, 211, 0.6)'
                            ),
                            barmode='stack',
                            plot_bgcolor='white',
                            height=600,
                            margin=dict(l=60, r=60, t=80, b=60),
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=-0.15,
                                xanchor="center",
                                x=0.5,
                                font=dict(size=10)
                            )
                        )
                        
                        # Mostrar el gráfico en Streamlit
                        st.plotly_chart(fig_comp, use_container_width=True)

                    
                    with tab3:
                        # GRÁFICO 3: RENDIMIENTO ACUMULADO MEJORADO
                        # Calcular los retornos diarios de todas las carteras usando X_test
                        retornos_diarios_all = X_test.dot(weights_all.T)
                        
                        # Calcular la rentabilidad acumulada
                        cumulative_returns = (1 + retornos_diarios_all).cumprod(axis=0) - 1
                        
                        # Crear el gráfico con plotly para mejor interactividad
                        fig_returns = go.Figure()
                        
                        # Agregar todas las carteras en gris claro
                        for i in range(len(population_train)):
                            if i != max_sharpe_idx:
                                fig_returns.add_trace(
                                    go.Scatter(
                                        x=cumulative_returns.index,
                                        y=cumulative_returns.iloc[:, i],
                                        mode='lines',
                                        line=dict(color='rgba(200, 200, 200, 0.3)', width=1),
                                        showlegend=False,
                                        hoverinfo='skip'
                                    )
                                )
                        
                        # Agregar la cartera de máximo Sharpe en rojo brillante
                        fig_returns.add_trace(
                            go.Scatter(
                                x=cumulative_returns.index,
                                y=cumulative_returns.iloc[:, max_sharpe_idx],
                                mode='lines',
                                line=dict(color='#E50914', width=3),
                                name='Cartera de Máximo Sharpe',
                                hovertemplate='<b>%{x}</b><br>Rendimiento: %{y:.2%}<extra></extra>'
                            )
                        )
                        
                        # Mejorar el diseño del gráfico
                        fig_returns.update_layout(
                            title=dict(
                                text='Rendimiento Acumulado de las Carteras en Período de Prueba',
                                font=dict(size=24, family="Arial, sans-serif", color="#2F4F4F")
                            ),
                            xaxis=dict(
                                title='Fecha',
                                gridcolor='rgba(211, 211, 211, 0.6)',
                                zeroline=False
                            ),
                            yaxis=dict(
                                title='Rendimiento Acumulado',
                                tickformat='.0%',
                                gridcolor='rgba(211, 211, 211, 0.6)',
                                zeroline=True,
                                zerolinecolor='rgba(0, 0, 0, 0.2)',
                                zerolinewidth=1
                            ),
                            plot_bgcolor='white',
                            height=500,
                            margin=dict(l=60, r=60, t=80, b=60),
                            legend=dict(
                                y=0.99,
                                x=0.01,
                                bgcolor='rgba(255, 255, 255, 0.8)',
                                bordercolor='rgba(211, 211, 211, 0.8)'
                            ),
                            hovermode='x unified'
                        )
                        
                        # Resaltar el rendimiento final
                        final_returns = cumulative_returns.iloc[-1, :]
                        max_final_return = final_returns.max()
                        max_final_return_idx = final_returns.argmax()
                        
                        # Agregar anotación para destacar el mejor rendimiento
                        if max_final_return_idx == max_sharpe_idx:
                            best_text = "¡La cartera de máximo Sharpe logró el mejor rendimiento!"
                        else:
                            best_text = f"Cartera de máximo Sharpe: {final_returns[max_sharpe_idx]:.2%} de rendimiento"
                        
                        fig_returns.add_annotation(
                            x=cumulative_returns.index[-1],
                            y=cumulative_returns.iloc[-1, max_sharpe_idx],
                            text=best_text,
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowcolor="#E50914",
                            ax=-150,
                            ay=-40,
                            bgcolor="rgba(255, 255, 255, 0.8)",
                            bordercolor="#E50914",
                            font=dict(color="#E50914")
                        )
                        
                        # Mostrar el gráfico en Streamlit
                        st.plotly_chart(fig_returns, use_container_width=True)
                        
                    with tab4:
                        # GRÁFICO 4: ESTADÍSTICAS DE LA CARTERA ÓPTIMA
                        # Obtener los pesos de la cartera óptima
                        optimal_weights = weights_all[max_sharpe_idx]
                        
                        # Crear un dataframe para mostrar todos los activos y sus pesos
                        weights_df = pd.DataFrame({
                            'Activo': asset_names,
                            'Peso (%)': optimal_weights * 100
                        }).sort_values(by='Peso (%)', ascending=False)
                        
                        # Dividir la pantalla en dos columnas
                        col1, col2 = st.columns([3, 2])
                        
                        with col1:
                            # Crear un gráfico de pastel para los componentes principales
                            # Filtrar solo activos con peso > 1% para el gráfico
                            significant_assets = [asset for asset, weight in zip(asset_names, optimal_weights) if weight > 0.01]
                            significant_weights = [weight for weight in optimal_weights if weight > 0.01]
                            
                            # Si hay muchos activos pequeños, agruparlos
                            if len(significant_assets) < len(asset_names):
                                otros_weight = sum(optimal_weights) - sum(significant_weights)
                                if otros_weight > 0:
                                    significant_assets.append("Otros")
                                    significant_weights.append(otros_weight)
                            
                            # Crear el gráfico de pastel
                            fig_pie = go.Figure(
                                data=[go.Pie(
                                    labels=significant_assets,
                                    values=significant_weights,
                                    hole=0.4,
                                    textinfo='label+percent',
                                    marker=dict(
                                        colors=plt.cm.tab20(np.linspace(0, 1, len(significant_assets))),
                                        line=dict(color='white', width=1.5)
                                    ),
                                    textfont=dict(size=12),
                                    insidetextorientation='radial'
                                )]
                            )
                            
                            fig_pie.update_layout(
                                title={
                                    'text': "Composición de la Cartera Óptima",
                                    'font': {'size': 25}  # Tamaño de fuente del título
                                },
                                height=400,
                                margin=dict(t=40, b=0, l=0, r=0)
                            )
                            
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        with col2:
                            st.subheader("Asignación de Activos")
                            
                            # Formateo de datos
                            weights_df = weights_df.copy()
                            weights_df['Peso (%)'] = weights_df['Peso (%)'].round(2).astype(str) + '%'
                            
                            # Estilo para el dataframe
                            st.dataframe(
                                weights_df.style
                                    .set_properties(**{
                                        'background-color': '#f8f9fa',
                                        'color': '#212529',
                                        'border': '1px solid #dee2e6'
                                    }),
                                hide_index=True,
                                use_container_width=True,
                                height=(len(weights_df) * 35 + 40)  # Altura dinámica
                            )
                        
                    

                    portfolio_prompt_efficient_frontier = f'''
                        Como estratega cuantitativo senior con expertise en teoría moderna de portafolios, analiza este ejercicio de optimización que incluye {portfolio_num} carteras en la frontera eficiente con una tasa libre de riesgo del {risk_free_rate*100:.2f}%. 
                        Los activos seleccionados son: {asset_names} con pesos óptimos en la cartera de máximo Sharpe de {optimal_weights.tolist()}.

                        Realiza un análisis estructurado que contemple:

                        ### 1. Contextualización teórica
                        - Explica qué representa la frontera eficiente en la teoría de Markowitz y por qué el ratio de Sharpe ({sharpe_ratios[max_sharpe_idx]:.3f}) es clave para seleccionar carteras óptimas.
                        - Describe cómo la tasa libre de riesgo ({risk_free_rate*100:.2f}%) afecta a la Línea de Mercado de Capitales y la selección de la cartera óptima.
                        - Analiza la importancia de la división entrenamiento-prueba (70-30) en la robustez de la optimización.

                        ### 2. Análisis de la cartera óptima
                        - Desglosa la composición del portafolio con máximo Sharpe:
                        - Identifica los 3 principales activos por peso y su contribución al riesgo/retorno
                        - Detecta activos excluidos (peso <1%) y posibles razones (ej: alta correlación)
                        - Compara la distribución real vs. la esperada según capitalización bursátil
                        - Evalúa el desempeño fuera de muestra:
                        - Explica la diferencia entre el Sharpe teórico ({sharpe_ratios[max_sharpe_idx]:.3f}) y el rendimiento real en prueba
                        - Analiza por qué algunas carteras de menor riesgo pudieron superar a la óptima en el período de prueba

                        ### 3. Interpretación de visualizaciones clave
                        #### ¿Qué es la Frontera Eficiente?
                        - La Frontera Eficiente muestra la relación riesgo-rendimiento de carteras optimizadas:
                        - Cada punto = cartera diferente
                        - Color = Ratio de Sharpe (más brillante = mejor)
                        - Estrella roja = Cartera con máximo Ratio de Sharpe
                        - Línea punteada = Línea de Mercado de Capitales (conecta tasa libre de riesgo ({risk_free_rate*100:.2f}%) con cartera óptima)

                        #### ¿Cómo interpretar el gráfico de composición?
                        - Muestra distribución de activos en la frontera eficiente:
                        - Columnas = Carteras ordenadas por riesgo (izq=bajo, der=alto)
                        - Colores = Activos diferentes
                        - Altura de segmentos = Peso del activo
                        - Triángulo rojo = Cartera con máximo Sharpe

                        #### ¿Qué representa el gráfico de rendimiento acumulado?
                        - Analiza desempeño histórico en prueba:
                        - Líneas grises = Otras carteras
                        - Línea roja = Cartera óptima
                        - Permite validar robustez de la optimización

                        ### 4. Análisis comparativo
                        - Compara métricas clave:
                        │ Métrica               │ Entrenamiento │ Prueba     │
                        │-----------------------+---------------+------------│
                        │ Rentabilidad Anual    │ {returns_annual[max_sharpe_idx]:.2%} │ {cumulative_returns.iloc[-1, max_sharpe_idx]:.2%} │
                        │ Riesgo Anualizado     │ {risks[max_sharpe_idx]:.2%}  │ -          │
                        │ Ratio de Sharpe       │ {sharpe_ratios[max_sharpe_idx]:.3f}  │ -          │
                        - Explica las posibles divergencias y su implicación en sobreajuste

                        ### Formato requerido
                        - Usa analogías financieras intuitivas (ej: "La frontera eficiente es como un mapa de riesgo-rendimiento")
                        - Destaca 3 insights accionables (ej: "Rotar un X% a activos defensivos si la tasa libre supera el Y%")
                        - Incluye cálculos demostrativos (ej: "Al aumentar la tasa en 1%, el Sharpe óptimo cae un Z%")
                        - Resalta patrones visuales clave del gráfico de rendimiento acumulado (ej: "Clúster de caídas en Q3 2022")
                        - Propone 2 estrategias alternativas basadas en la composición óptima
                        '''

                    if st.session_state.first == False and st.session_state.first_min_var_max_rent == False:
                        st.session_state.question = portfolio_prompt_efficient_frontier
                        st.session_state.first = True
                        st.session_state.first_min_var_max_rent = True

                except Exception as e:
                    st.error(f"Error calculando el portafolio: {str(e)}")

        # mostrar_chat()


# -------------------------------------------------------------------------------
# 10. Ejecución principal
# -------------------------------------------------------------------------------
def main():
    """
    Dependiendo de si el usuario ya hizo una pregunta (st.session_state.question),
    mostramos la pantalla inicial o la interfaz de chat.
    """
    mostrar_pantalla_inicial()

if __name__ == "__main__":
    main()
