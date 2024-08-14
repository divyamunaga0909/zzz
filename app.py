import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import cv2
import statistics
import sklearn
import pymysql
import base64
from fpdf import FPDF  # Import FPDF for PDF generation
from datetime import datetime

# Function to convert 12-hour time format to 24-hour format
def convert_time_to_24hr(time_str):
    return datetime.strptime(time_str, "%I:%M %p").strftime("%H:%M:%S")

# Print the version of Scikit-learn being used
print(f"Scikit-learn version: {sklearn.__version__}")

# Load the pre-trained machine learning models
rf_model = joblib.load('rf_model.joblib')
xgb_model = joblib.load('xgb_model.joblib')
cat_model = joblib.load('cat_model.joblib')

# Load the scaler used for data normalization; handle any errors if the file is corrupted or empty
try:
    scaler = joblib.load('scaler.joblib')
except EOFError:
    st.error("Error: The scaler.joblib file is empty or corrupted.")
    scaler = None

# Load the label encoder used to convert predictions back to readable labels
label_encoder = joblib.load('label_encoder.joblib')

# Load the voting classifier, which is an ensemble of the models loaded above
voting_clf = joblib.load('voting_clf.joblib')

# Database connection setup using pymysql
def connect_db():
    return pymysql.connect(
        host="localhost",
        user="root",
        password="2004",
        database="predictza"
    )

# Function to create a new user account by inserting user data into the database
def create_account(name, email, number, age, gender, password):
    try:
        connection = connect_db()
        with connection.cursor() as cursor:
            query = "INSERT INTO users (name, email, number, age, gender, password) VALUES (%s, %s, %s, %s, %s, %s)"
            cursor.execute(query, (name, email, number, age, gender, password))
            connection.commit()
            st.success("Account created successfully!")
    except Exception as e:
        st.error(f"Error creating account: {e}")
    finally:
        connection.close()

# Function to check user login credentials by querying the database
def check_login(email, password):
    try:
        connection = connect_db()
        with connection.cursor() as cursor:
            query = "SELECT * FROM users WHERE email=%s AND password=%s"
            cursor.execute(query, (email, password))
            user = cursor.fetchone()
            if user:
                st.success("Login successful!")
                return user
            else:
                st.error("Invalid credentials. Please try again.")
    except Exception as e:
        st.error(f"Error logging in: {e}")
    finally:
        connection.close()

# Function to extract metadata from an ECG image
def extract_ecg_metadata(image_path):
    # Read the ECG image using OpenCV
    ecg_image = cv2.imread(image_path)
    
    # Resize the image to a standard size
    resized_image = cv2.resize(ecg_image, (2213, 1572))
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Convert the image to a binary image using adaptive thresholding
    binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Detect edges using the Canny edge detection algorithm
    edges = cv2.Canny(binary_image, 50, 150)
    
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize lists to store various features of the ECG image
    st_elevation_values = []
    pathological_q_waves_values = []
    t_wave_inversions_values = []
    abnormal_qrs_complexes_values = []

    # Iterate through each contour to extract relevant features based on contour dimensions
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > 20:
            st_elevation_values.append(h)
        if w > 10 and h < 5:
            pathological_q_waves_values.append(h)
        if h < 10 and w > 15:
            abnormal_qrs_complexes_values.append(h)
        if h < 10 and w < 10:
            t_wave_inversions_values.append(h)

    # Function to calculate statistics (max, mean, median) from a list of values
    def calculate_stats(values):
        return {
            'max': max(values) if values else 0,
            'mean': statistics.mean(values) if values else 0,
            'median': statistics.median(values) if values else 0
        }

    # Create a dictionary to hold the calculated metadata
    metadata = {
        'Max ST Elevation (Height)': calculate_stats(st_elevation_values)['max'],
        'Mean ST Elevation (Height)': calculate_stats(st_elevation_values)['mean'],
        'Median ST Elevation (Height)': calculate_stats(st_elevation_values)['median'],
        'Max Pathological Q Wave (Height)': calculate_stats(pathological_q_waves_values)['max'],
        'Mean Pathological Q Wave (Height)': calculate_stats(pathological_q_waves_values)['mean'],
        'Median Pathological Q Wave (Height)': calculate_stats(pathological_q_waves_values)['median'],
        'Max T Wave Inversion (Height)': calculate_stats(t_wave_inversions_values)['max'],
        'Mean T Wave Inversion (Height)': calculate_stats(t_wave_inversions_values)['mean'],
        'Median T Wave Inversion (Height)': calculate_stats(t_wave_inversions_values)['median'],
        'Max Abnormal QRS Complex (Height)': calculate_stats(abnormal_qrs_complexes_values)['max'],
        'Mean Abnormal QRS Complex (Height)': calculate_stats(abnormal_qrs_complexes_values)['mean'],
        'Median Abnormal QRS Complex (Height)': calculate_stats(abnormal_qrs_complexes_values)['median']
    }

    return metadata

# Function to predict heart disease type based on ECG image metadata
def predict_disease(image_path, model, scaler, label_encoder):
    # Extract metadata from the ECG image
    metadata = extract_ecg_metadata(image_path)
    metadata_df = pd.DataFrame([metadata])

    st.write("Extracted metadata:", metadata)

    # Scale the metadata using the loaded scaler and make a prediction using the model
    if scaler:
        metadata_scaled = scaler.transform(metadata_df)
        st.write("Scaled metadata:", metadata_scaled)
        prediction_index = model.predict(metadata_scaled)[0]
        predicted_class = label_encoder.inverse_transform([prediction_index])[0]
        return predicted_class
    else:
        return "Error: Scaler is not available."

# Function to return a list of precautions based on the predicted disease type
def get_precautions(disease_type):
    precautions = {
    "myocardial": [
        "1. Take prescribed medications as directed.",
        "2. Avoid heavy physical exertion.",
        "3. Monitor heart rate and report any irregularities.",
        "4. Follow up with a cardiologist regularly."
    ],
    "historyofmi": [
        "1. Maintain a healthy lifestyle with a balanced diet.",
        "2. Exercise regularly but avoid strenuous activities.",
        "3. Regular check-ups with a healthcare provider.",
        "4. Keep track of any new symptoms."
    ],
    "abnormal": [
        "1. Follow up with a healthcare provider for further evaluation.",
        "2. Monitor for any changes in symptoms.",
        "3. Maintain a healthy lifestyle."
    ],
    "normal": [
        "1. Maintain a healthy diet rich in fruits, vegetables, and whole grains.",
        "2. Engage in regular physical activity, such as walking or cycling.",
        "3. Avoid smoking and limit alcohol intake.",
        "4. Manage stress through relaxation techniques.",
        "5. Monitor blood pressure and visit your healthcare provider regularly."
    ]
}
    return precautions.get(disease_type, [])

from fpdf import FPDF

def generate_pdf_report(name, age, gender, disease_type, precautions, metadata, scaled_metadata):
    pdf = FPDF()
    pdf.add_page()

    # Main Heading
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 10, txt="PREDICTZA", ln=True, align="C")
    pdf.ln(10)  # Line break

    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt="Heart Disease Prediction Report", ln=True, align="C")
    pdf.ln(10)  # Line break

    # Patient Information
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="Patient Information:", ln=True, align="L", border=1)
    pdf.ln(5)

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt=f"Name: {name}", ln=True, align="L", border=1)
    pdf.cell(0, 10, txt=f"Age: {age}", ln=True, align="L", border=1)
    pdf.cell(0, 10, txt=f"Gender: {gender}", ln=True, align="L", border=1)
    pdf.cell(0, 10, txt=f"Predicted Disease Type: {disease_type}", ln=True, align="L", border=1)
    pdf.ln(10)  # Line break

    # Metadata Section
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="ECG Metadata:", ln=True, align="L", border=1)
    pdf.ln(5)

    pdf.set_font("Arial", size=12)
    for key, value in metadata.items():
        pdf.cell(0, 10, txt=f"{key}: {value}", ln=True, align="L", border=1)
    pdf.ln(10)  # Line break

    # Scaled Metadata Section
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="Scaled Metadata:", ln=True, align="L", border=1)
    pdf.ln(5)

    pdf.set_font("Arial", size=12)
    for i, column_name in enumerate(metadata.keys()):
        pdf.cell(0, 10, txt=f"{column_name}: {scaled_metadata[0][i]:.2f}", ln=True, align="L", border=1)
    pdf.ln(10)  # Line break

    # Precautions
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="Precautions:", ln=True, align="L", border=1)
    pdf.ln(5)

    pdf.set_font("Arial", size=12)
    if precautions:
        for precaution in precautions:
            pdf.multi_cell(0, 10, txt=precaution, border=1, align="L")
    else:
        pdf.cell(0, 10, txt="No precautions available.", ln=True, align="L", border=1)

    # Save PDF to a file
    pdf_filename = f"{name}_heart_disease_report.pdf"
    pdf.output(pdf_filename)
    return pdf_filename

# Function to download the PDF report
def download_pdf_report(pdf_filename):
    with open(pdf_filename, "rb") as pdf_file:
        b64_pdf = base64.b64encode(pdf_file.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="{pdf_filename}">Download Report as PDF</a>'
    st.markdown(href, unsafe_allow_html=True)

def update_user_report(user_email, predicted_disease):
    try:
        connection = connect_db()
        with connection.cursor() as cursor:
            query = "UPDATE users SET report=%s WHERE email=%s"
            cursor.execute(query, (f"Predicted Disease: {predicted_disease}", user_email))
            connection.commit()
    except Exception as e:
        st.error(f"Error updating user report: {e}")
    finally:
        connection.close()

# Function to book an appointment in the database
def book_appointment(user_info, appointment_details):
    try:
        connection = connect_db()
        with connection.cursor() as cursor:
            query = "INSERT INTO appointments (user_email, Name, date, time, number) VALUES (%s, %s, %s, %s, %s)"
            cursor.execute(query, (
                user_info["email"], 
                appointment_details["Name"], 
                appointment_details["date"], 
                appointment_details["time"], 
                appointment_details["number"]
            ))
            connection.commit()
            st.success(f"Appointment booked successfully for {appointment_details['date']} at {appointment_details['time']}!")
    except Exception as e:
        st.error(f"Error booking appointment: {e}")
    finally:
        connection.close()
    
# Streamlit app
def main():
    st.title("Heart Disease Prediction App")

    menu = ["Login", "Sign Up", "ECG Classification"]
    choice = st.sidebar.selectbox("Select an option", menu)

    if choice == "Sign Up":
        st.subheader("Create a New Account")
        name = st.text_input("Name")
        email = st.text_input("Email")
        number = st.text_input("Phone Number")
        age = st.number_input("Age", min_value=0)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        password = st.text_input("Password", type="password")
        if st.button("Create Account"):
            create_account(name, email, number, age, gender, password)

    elif choice == "Login":
        st.subheader("Login")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            user_info = check_login(email, password)
            if user_info:
                st.session_state.user_info = {
                    "id": user_info[0],
                    "name": user_info[1],
                    "email": user_info[2],
                    "phone": user_info[3],
                    "age": user_info[4],
                    "gender": user_info[5]
                }

    elif choice == "ECG Classification":
        if 'user_info' not in st.session_state:
            st.error("You must be logged in to access this page.")
            st.stop()

        st.title("ECG Image Classification")
        uploaded_file = st.file_uploader("Choose an ECG image...", type="jpg")
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded ECG Image.', use_column_width=True)

            temp_image_path = 'temp_ecg_image.jpg'
            with open(temp_image_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())

            st.write("Classifying...")
            predicted_disease = predict_disease(temp_image_path, voting_clf, scaler, label_encoder)
            st.write(f"Predicted disease type is: {predicted_disease}")

            precautions = get_precautions(predicted_disease)

            # Extract metadata and scaled metadata
            metadata = extract_ecg_metadata(temp_image_path)
            metadata_df = pd.DataFrame([metadata])
            if scaler:
                scaled_metadata = scaler.transform(metadata_df)
            else:
                scaled_metadata = []

            # Generate and download PDF report
            if st.button("Generate Report"):
                pdf_filename = generate_pdf_report(
                    st.session_state.user_info["name"],
                    st.session_state.user_info["age"],
                    st.session_state.user_info["gender"],
                    predicted_disease,
                    precautions,
                    metadata,
                    scaled_metadata
                )
                st.success(f"Report generated: {pdf_filename}")
                download_pdf_report(pdf_filename)

                # Update user record with the report and predicted disease
                update_user_report(st.session_state.user_info["email"], predicted_disease)

        # Adding date and time input for appointment booking
        st.subheader("Book an Appointment")
        selected_date = st.date_input("Select appointment date", datetime.today())
        selected_time = st.time_input("Select appointment time", datetime.now().time())

        if st.button("Book an appointment"):
            appointment_details = {
                "Name": st.session_state.user_info["name"],  # Example user's name
                "date": selected_date.strftime("%Y-%m-%d"),
                "time": selected_time.strftime("%H:%M:%S"),
                "number": st.session_state.user_info["phone"]  # Example phone number
            }
            book_appointment(st.session_state.user_info, appointment_details)

if __name__ == "__main__":
    main()
