import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="insurance_cost_prediction_model.tflite")
interpreter.allocate_tensors()

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Ambil info tensor
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# UI Judul
st.title("🏥 Prediksi Biaya Asuransi Kesehatan")

st.markdown("Masukkan informasi pasien di bawah ini untuk memprediksi biaya asuransi:")

# Input dari pengguna
age = st.slider("Usia", 18, 64, 30)
bmi = st.slider("BMI", 15.0, 50.0, 25.0)
children = st.number_input("Jumlah Anak", min_value=0, max_value=5, value=0)
sex = st.selectbox("Jenis Kelamin", ['Perempuan', 'Laki-laki'])
smoker = st.selectbox("Perokok?", ['Tidak', 'Ya'])
region = st.selectbox("Region", ['northeast', 'northwest', 'southeast', 'southwest'])

# Fitur turunan
bmi_age_interaction = age * bmi
is_obese = 1 if bmi > 30 else 0

# One-hot encoding manual sesuai training
region_cols = ['region_northwest', 'region_southeast', 'region_southwest']
region_vector = [0, 0, 0]
if region != 'northeast':
    region_vector[region_cols.index(f'region_{region}')] = 1

# sex_male dan smoker_yes
sex_male = 1 if sex == 'Laki-laki' else 0
smoker_yes = 1 if smoker == 'Ya' else 0

# Gabungkan semua fitur ke DataFrame
input_data = pd.DataFrame([{
    'age': age,
    'bmi': bmi,
    'children': children,
    'bmi_age_interaction': bmi_age_interaction,
    'is_obese': is_obese,
    'sex_male': sex_male,
    'smoker_yes': smoker_yes,
    'region_northwest': region_vector[0],
    'region_southeast': region_vector[1],
    'region_southwest': region_vector[2]
}])

# Tombol prediksi
if st.button("Prediksi Biaya Asuransi"):
    # Normalisasi data
    input_scaled = scaler.transform(input_data).astype(np.float32)

    # Ubah bentuk sesuai input TensorFlow Lite
    input_scaled = input_scaled.reshape(1, -1)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_scaled)

    # Jalankan prediksi
    interpreter.invoke()

    # Ambil hasil prediksi
    predicted_cost = interpreter.get_tensor(output_details[0]['index'])[0][0]

    # Tampilkan hasil
    st.subheader("💰 Hasil Prediksi:")
    st.success(f"Biaya Asuransi yang Diprediksi: **${predicted_cost:.2f}**")
