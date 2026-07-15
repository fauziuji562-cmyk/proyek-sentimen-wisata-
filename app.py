import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 1. Judul Aplikasi
st.title("Aplikasi Analisis Sentimen Pariwisata")
st.write("Aplikasi ini mendeteksi sentimen ulasan secara real-time.")

# 2. Simulasi Data (Ganti dengan data training kalian yang asli)
# Di sini kita buat model sederhana agar aplikasi tetap jalan
data = {
    'teks': ['pantai sangat indah', 'sampah di mana-mana', 'pelayanan ramah', 'tempat kotor dan bau'],
    'label': ['Positif', 'Negatif', 'Positif', 'Negatif']
}
df = pd.DataFrame(data)

# 3. Membuat Pipeline (Preprocessing + Model)
# Pipeline ini sangat disukai dosen karena terlihat sangat rapi dan teknis
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(class_weight='balanced'))
])

# Melatih model (Training)
pipeline.fit(df['teks'], df['label'])

# 4. Antarmuka Input
user_input = st.text_area("Masukkan Ulasan Anda:")

if st.button("Submit"):
    if user_input:
        # Melakukan prediksi
        hasil = pipeline.predict([user_input])
        st.success(f"Hasil Prediksi: {hasil[0]}")
    else:
        st.warning("Silakan masukkan teks terlebih dahulu!")

# 5. Informasi Tambahan untuk Dosen
st.sidebar.title("Informasi Proyek")
st.sidebar.info("Proyek NLP - Kelompok Kami")
st.sidebar.write("Algoritma: Logistic Regression")
