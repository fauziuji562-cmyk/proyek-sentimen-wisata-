import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Konfigurasi tampilan halaman agar lebih "luas" dan rapi
st.set_page_config(page_title="Analisis Sentimen Wisata", page_icon="🏖️")

st.title("🏖️ Analisis Sentimen Pariwisata")
st.write("Aplikasi untuk memonitor kepuasan pelanggan secara real-time.")

# Logika untuk Penjelasan otomatis
def get_penjelasan(teks, sentimen):
    if sentimen == "Negatif":
        if "kotor" in teks.lower():
            return "Karena pengunjung dan masyarakat sekitar kurang menjaga kebersihan area wisata."
        elif "mahal" in teks.lower():
            return "Karena harga tiket atau makanan dinilai tidak sebanding dengan fasilitas yang didapat."
        return "Karena adanya ketidakpuasan terhadap layanan atau fasilitas yang tersedia."
    else:
        return "Karena pengalaman pelanggan sangat baik dan sesuai dengan ekspektasi."

# Training Model Sederhana
data = {'teks': ['pantai indah', 'tempat kotor', 'harga mahal', 'pelayanan ramah'],
        'label': ['Positif', 'Negatif', 'Negatif', 'Positif']}
df = pd.DataFrame(data)
pipeline = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression())])
pipeline.fit(df['teks'], df['label'])

# Input Pengguna
user_input = st.text_area("Tulis ulasan Anda di sini:", placeholder="Contoh: tempatnya kotor...")

if st.button("Analisis Sentimen"):
    if user_input:
        prediksi = pipeline.predict([user_input])[0]
        penjelasan = get_penjelasan(user_input, prediksi)
        
        # Logika Warna (Hijau untuk Positif, Merah untuk Negatif)
        if prediksi == "Positif":
            st.success(f"Hasil Prediksi: {prediksi}")
        else:
            st.error(f"Hasil Prediksi: {prediksi}")
            
        st.info(f"💡 **Analisis Pakar:** {penjelasan}")
    else:
        st.warning("Silakan masukkan teks!")

# Sidebar agar terlihat profesional
st.sidebar.header("Tentang Proyek")
st.sidebar.write("Proyek NLP - Kelompok Kami")
st.sidebar.write("Algoritma: Logistic Regression")
