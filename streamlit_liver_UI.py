import pickle
import numpy as np
import streamlit as st
from io import BytesIO
from xhtml2pdf import pisa

# Fungsi untuk membuat PDF
def generate_pdf(content):
    pdf_buffer = BytesIO()
    pisa_status = pisa.CreatePDF(
        src=BytesIO(content.encode("utf-8")),
        dest=pdf_buffer
    )
    pdf_buffer.seek(0)
    if pisa_status.err:
        raise ValueError("Gagal membuat PDF.")
    return pdf_buffer

# Load model dan scaler
liver_model = pickle.load(open('liver_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

# Judul aplikasi
st.title("🩺 **Prediksi Penyakit Liver**")
st.subheader("Aplikasi ini menggunakan data mining untuk mendeteksi kemungkinan penyakit liver.")
st.markdown("---")

# Inisialisasi session state
if 'liv_diagnosis' not in st.session_state:
    st.session_state['liv_diagnosis'] = ''

# Form Input
with st.form("form_diagnosis"):
    st.header("📝 Masukkan Data Pasien")
    col1, col2 = st.columns(2)

    with col1:
        Age = st.text_input("Umur")
        Total_Bilirubin = st.text_input("Total Bilirubin")
        Alkaline_Phosphotase = st.text_input("Alkaline Phosphotase")
        Aspartate_Aminotransferase = st.text_input("Aspartate Aminotransferase")
        Albumin = st.text_input("Albumin")

    with col2:
        Gender = st.selectbox("Gender", options=["Laki-laki", "Perempuan"], index=0)
        Direct_Bilirubin = st.text_input("Direct Bilirubin")
        Alamine_Aminotransferase = st.text_input("Alamine Aminotransferase)
        Total_Protiens = st.text_input("Total Protein")
        Albumin_and_Globulin_Ratio = st.text_input("Albumin & Globulin Ratio")

    # Tombol untuk prediksi
    submitted = st.form_submit_button("🔍 Prediksi")
    if submitted:
        try:
            # Konversi input hanya untuk model, biarkan Gender dalam bentuk teks untuk PDF
            Gender_numeric = 1 if Gender == "Laki-laki" else 0
            input_data = np.array([[ 
                float(Age), Gender_numeric, float(Total_Bilirubin), float(Direct_Bilirubin), 
                float(Alkaline_Phosphotase), float(Alamine_Aminotransferase), 
                float(Aspartate_Aminotransferase), float(Total_Protiens), 
                float(Albumin), float(Albumin_and_Globulin_Ratio)
            ]])
            std_data = scaler.transform(input_data)
            liv_prediction = liver_model.predict(std_data)

            if liv_prediction[0] == 1:
                st.session_state['liv_diagnosis'] = 'Pasien terkena penyakit liver.'
            else:
                st.session_state['liv_diagnosis'] = 'Pasien tidak terkena penyakit liver.'

            st.success(st.session_state['liv_diagnosis'])
        except ValueError:
            st.error("Pastikan semua input berupa angka yang valid!")

# Tombol untuk mengunduh hasil
st.markdown("---")
if st.button("⬇️ Unduh Hasil Sebagai PDF"):
    if st.session_state['liv_diagnosis'] == '':
        st.error("Lakukan prediksi terlebih dahulu sebelum mengunduh PDF!")
    else:
        # Menampilkan hasil PDF dengan format desimal pada kolom tertentu
        html_content = f"""
        <h1>Data Mining Prediksi Liver</h1>
        <p><strong>Umur:</strong> {Age}</p>
        <p><strong>Gender:</strong> {Gender}</p>
        <p><strong>Total Bilirubin:</strong> {Total_Bilirubin}</p>
        <p><strong>Direct Bilirubin:</strong> {Direct_Bilirubin}</p>
        <p><strong>Alkaline Phosphotase:</strong> {Alkaline_Phosphotase}</p>
        <p><strong>Alamine Aminotransferase:</strong> {Alamine_Aminotransferase}</p>
        <p><strong>Aspartate Aminotransferase:</strong> {Aspartate_Aminotransferase}</p>
        <p><strong>Total Protein:</strong> {Total_Protiens}</p>
        <p><strong>Albumin:</strong> {Albumin}</p>
        <p><strong>Albumin and Globulin Ratio:</strong> {Albumin_and_Globulin_Ratio}</p>
        <h2>Hasil Diagnosis:</h2>
        <p>{st.session_state['liv_diagnosis']}</p>
        """
        try:
            pdf_file = generate_pdf(html_content)
            st.download_button(
                label="📄 Unduh PDF",
                data=pdf_file,
                file_name="Hasil_Prediksi_Liver.pdf",
                mime="application/pdf"
            )
        except ValueError as e:
            st.error(f"Gagal membuat PDF: {e}")
