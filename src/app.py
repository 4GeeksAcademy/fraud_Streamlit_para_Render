import joblib
import streamlit as st
import os
from sklearn.ensemble import RandomForestClassifier

# Ruta al archivo del modelo
ruta_modelo = "/workspaces/fraud_para_Render/models/modelo_RandomForest_optimizado.pkl"

# Verificar la existencia del archivo del modelo
if not os.path.exists(ruta_modelo):
    raise FileNotFoundError(f"El archivo {ruta_modelo} no existe")

# Cargar el modelo
model = joblib.load(ruta_modelo)

# Verificar si el modelo está cargado correctamente
if not isinstance(model, RandomForestClassifier):
    raise TypeError("El archivo cargado no es un modelo RandomForest")

# Diccionario de clases
class_dict = {
    "0": "No Fraude",
    "1": "Fraude",
}

# Título de la aplicación
st.title("MODELO MACHINE LEARNING PARA LA PREDICCION DE FRAUDE FINANCIERO")

# Menú de navegación
menu = st.sidebar.selectbox("Selecciona una opción", ["Predicción de Fraude", "Reseña sobre Fraudes Financieros"])

if menu == "Predicción de Fraude":
    st.header("Predicción de Fraude en Transacciones Bancarias")

    # Formularios para ingresar los valores de manera estructurada
    with st.form("Formulario de Datos"):
        st.subheader("Por favor, complete la siguiente información")
        income = st.number_input("Ingresos", min_value=0.0, max_value=1000000.0, step=1000.0)
        name_email_similarity = st.slider("Similitud entre Nombre y Email", min_value=0.0, max_value=1.0, step=0.01)
        prev_address_months_count = st.number_input("Meses en la Dirección Anterior", min_value=0, max_value=240, step=1)
        current_address_months_count = st.number_input("Meses en la Dirección Actual", min_value=0, max_value=240, step=1)
        customer_age = st.number_input("Edad del Cliente", min_value=18, max_value=100, step=1)
        days_since_request = st.number_input("Días desde la Solicitud", min_value=0.0, max_value=365.0, step=1.0)
        intended_balcon_amount = st.number_input("Monto del Saldo Intencionado", min_value=0.0, max_value=1000000.0, step=1000.0)
        zip_count_4w = st.number_input("Conteo de Códigos Postales en 4 Semanas", min_value=0, max_value=50, step=1)
        velocity_6h = st.number_input("Velocidad de Transacción en 6 Horas", min_value=0.0, max_value=1000.0, step=1.0)
        velocity_24h = st.number_input("Velocidad de Transacción en 24 Horas", min_value=0.0, max_value=1000.0, step=1.0)
        velocity_4w = st.number_input("Velocidad de Transacción en 4 Semanas", min_value=0.0, max_value=1000.0, step=1.0)
        bank_branch_count_8w = st.number_input("Número de Sucursales Bancarias en 8 Semanas", min_value=0, max_value=20, step=1)
        date_of_birth_distinct_emails_4w = st.number_input("Correos Electrónicos Distintos en 4 Semanas", min_value=0, max_value=10, step=1)
        credit_risk_score = st.number_input("Puntuación de Riesgo Crediticio", min_value=0, max_value=1000, step=1)
        email_is_free = st.selectbox("Email es Gratuito", [0, 1])
        phone_home_valid = st.selectbox("Teléfono de Casa Válido", [0, 1])
        phone_mobile_valid = st.selectbox("Teléfono Móvil Válido", [0, 1])
        bank_months_count = st.number_input("Meses con el Banco", min_value=0, max_value=240, step=1)
        has_other_cards = st.selectbox("Tiene Otras Tarjetas", [0, 1])
        proposed_credit_limit = st.number_input("Límite de Crédito Propuesto", min_value=0.0, max_value=1000000.0, step=1000.0)
        foreign_request = st.selectbox("Solicitud Extranjera", [0, 1])
        session_length_in_minutes = st.number_input("Duración de la Sesión en Minutos", min_value=0.0, max_value=1440.0, step=1.0)
        device_distinct_emails_8w = st.number_input("Emails Distintos en 8 Semanas", min_value=0, max_value=10, step=1)
        device_fraud_count = st.number_input("Conteo de Fraudes en el Dispositivo", min_value=0, max_value=10, step=1)
        month = st.slider("Mes", min_value=1, max_value=12, step=1)

        # Botón de predicción
        submit_button = st.form_submit_button(label="Predecir")

    if submit_button:
        data = [[
            income, name_email_similarity, prev_address_months_count, current_address_months_count, customer_age,
            days_since_request, intended_balcon_amount, zip_count_4w, velocity_6h, velocity_24h, velocity_4w,
            bank_branch_count_8w, date_of_birth_distinct_emails_4w, credit_risk_score, email_is_free, phone_home_valid,
            phone_mobile_valid, bank_months_count, has_other_cards, proposed_credit_limit, foreign_request,
            session_length_in_minutes, device_distinct_emails_8w, device_fraud_count, month
        ]]
        
        prediction = str(model.predict(data)[0])
        pred_class = class_dict[prediction]
        st.write("Predicción:", pred_class)

elif menu == "Reseña sobre Fraudes Financieros":
    st.header("Reseña sobre Fraudes Financieros")

    st.write("""
    Los fraudes financieros son actividades ilícitas que buscan obtener ganancias económicas mediante engaños y estafas. 
    Existen diferentes tipos de fraudes financieros, entre ellos:

    1. **Fraude con Tarjetas de Crédito**: Clonación o robo de datos de tarjetas para realizar compras no autorizadas.
    2. **Phishing**: Intento de obtener información confidencial como contraseñas o detalles bancarios mediante correos electrónicos falsos o sitios web fraudulentos.
    3. **Fraude en Transferencias Bancarias**: Manipulación o redirección de transferencias electrónicas hacia cuentas no autorizadas.
    4. **Fraude en Préstamos**: Obtención de préstamos con identidades falsas o documentos falsificados.
    5. **Fraude en Inversiones**: Promesas de altos rendimientos con esquemas Ponzi o inversiones inexistentes.

    Las medidas de prevención y detección incluyen el monitoreo continuo de transacciones, implementación de tecnologías avanzadas de seguridad y educación a los usuarios sobre prácticas seguras.
    """)
