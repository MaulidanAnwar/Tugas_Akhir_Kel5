import pickle
import numpy as np
import streamlit as st
from io import BytesIO
from xhtml2pdf import pisa

def load_assets():
    liver_model = pickle.load(open('liver_model.sav', 'rb'))
    scaler = pickle.load(open('scaler.sav', 'rb'))
    return liver_model, scaler

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

st.set_page_config(
    page_title="Prediksi Penyakit Liver", 
    page_icon="🩺", 
    layout="centered",
)

background_url = "https://raw.githubusercontent.com/MaulidanAnwar/Final-Project-Group-5/929c4288ff7618d043646f5d6ffb5dcecdcb945d/healthcare-accessories-with-modern-devices-green-background.jpg"

st.markdown(
    f"""
    <style>
        .stApp {{
            background-image: url("{background_url}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-color: white; /* Warna dasar untuk fallback */
        }}
        .stAlert {{
            background-color: #007BFF !important;  /* Warna biru solid */
            color: white !important;  /* Warna font putih */
            border-radius: 8px !important;  /* Membulatkan sudut */
            padding: 20px !important;  /* Menambah ruang di dalam elemen */
        }}
        .stAlert > div[role="alert"] {{
            background-color: #007BFF !important;  /* Warna biru solid */
            color: white !important;  /* Warna font putih */
        }}
    </style>
    """,
    unsafe_allow_html=True
)


if 'page' not in st.session_state:
    st.session_state['page'] = 1

    
if 'liv_diagnosis' not in st.session_state:
    st.session_state['liv_diagnosis'] = ''

def page_welcome():
    st.markdown(
        """
        <h1 style="color: darkblue;">🩺 <b>Prediksi Penyakit Liver</b></h1>
        """,
        unsafe_allow_html=True
    )
    st.subheader("Selamat datang! Aplikasi ini menggunakan data mining untuk mendeteksi kemungkinan penyakit liver.")
    st.markdown("---")
    if st.button("Mulai", key="start_button"):
        st.session_state['page'] = 2
        st.rerun()


def page_formulir(liver_model, scaler):
    st.header("📝 Masukkan Data Pasien")
    
    Nama = st.text_input("Nama Pasien")
    
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
        Alamine_Aminotransferase = st.text_input("Alamine Aminotransferase")
        Total_Protiens = st.text_input("Total Protiens")
        Albumin_and_Globulin_Ratio = st.text_input("Albumin & Globulin Ratio")

    if st.button("🔍 Prediksi"):
        try:
            Gender_numeric = 1 if Gender == "Laki-laki" else 0
            input_data = np.array([[ 
                float(Age), Gender_numeric, float(Total_Bilirubin), float(Direct_Bilirubin), 
                float(Alkaline_Phosphotase), float(Alamine_Aminotransferase), 
                float(Aspartate_Aminotransferase), float(Total_Protiens), 
                float(Albumin), float(Albumin_and_Globulin_Ratio)
            ]])
            std_data = scaler.transform(input_data)
            prediction = liver_model.predict(std_data)
            st.session_state['liv_diagnosis'] = (
                'Pasien terkena penyakit liver.' if prediction[0] == 1 else 'Pasien tidak terkena penyakit liver.'
            )
            st.success(st.session_state['liv_diagnosis'])
        except ValueError:
            st.error("Pastikan semua input berupa angka yang valid!")

    if st.session_state['liv_diagnosis']:
        user_inputs = [Nama, Age, Gender, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase, 
                       Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens, 
                       Albumin, Albumin_and_Globulin_Ratio]
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    }}
                 h1 {{
                    text-align: center;
                    color: #2C3E50;
                    font-size: 18px;
                    }}
                 h2 {{
                    color: #34495E;
                    font-size: 16px;
                    }}
                p {{
                    font-size: 16px;
                    line-height: 1.6;
                    }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 20px;
                    }}
                table, th, td {{
                    border: 1px solid #ddd;
                    }}
                th, td {{
                    padding: 8px;
                    text-align: left;
                    font-size: 16px;
                    }}
                th {{
                    background-color: #f2f2f2;
                    font-size: 18px;
                    }}
            </style>
        </head>
        <body>
            <h1>Data Mining Prediksi Liver</h1>
            <table>
                <tr><th>Nama</th><td>{user_inputs[0]}</td></tr>
                <tr><th>Umur</th><td>{user_inputs[1]}</td></tr>
                <tr><th>Gender</th><td>{user_inputs[2]}</td></tr>
                <tr><th>Total Bilirubin</th><td>{user_inputs[3]}</td></tr>
                <tr><th>Direct Bilirubin</th><td>{user_inputs[4]}</td></tr>
                <tr><th>Alkaline Phosphotase</th><td>{user_inputs[5]}</td></tr>
                <tr><th>Alamine Aminotransferase</th><td>{user_inputs[6]}</td></tr>
                <tr><th>Aspartate Aminotransferase</th><td>{user_inputs[7]}</td></tr>
                <tr><th>Total Protiens</th><td>{user_inputs[8]}</td></tr>
                <tr><th>Albumin</th><td>{user_inputs[9]}</td></tr>
                <tr><th>Albumin & Globulin Ratio</th><td>{user_inputs[10]}</td></tr>
            </table>
            <h2>Hasil Diagnosis:</h2>
            <p>{st.session_state['liv_diagnosis']}</p>
        </body>
        </html>
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
    
def main():
    liver_model, scaler = load_assets()
    if st.session_state['page'] == 1:
        page_welcome()
    elif st.session_state['page'] == 2:
        page_formulir(liver_model, scaler)

if __name__ == "__main__":
    main()
