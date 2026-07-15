import streamlit as st
import joblib

# Load model yang sudah disimpan
model = joblib.load('model.pkl')

st.title("Aplikasi Analisis Sentimen Pariwisata")
st.write("Masukkan ulasan untuk mendeteksi sentimen:")

user_input = st.text_area("Ulasan:")

if st.button("Submit"):
    # Logika prediksi di sini
    prediksi = model.predict([user_input])
    st.write(f"Hasil: {prediksi}")
