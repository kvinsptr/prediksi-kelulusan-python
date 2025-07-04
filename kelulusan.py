import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Konfigurasi halaman ---
st.set_page_config(page_title="🎓 Prediksi Kelulusan Mahasiswa", layout="wide", page_icon="🎓")

st.markdown("""
    <div style='text-align: center;'>
        <h1>🎓 Prediksi Kelulusan Mahasiswa</h1>
        <p style='font-size:18px;'>Menggunakan model <strong>Machine Learning</strong> berbasis performa akademik dan data pribadi mahasiswa</p>
    </div>
""", unsafe_allow_html=True)

# --- Load Data ---
data = pd.read_excel("Kelulusan Train.xlsx")
data.columns = data.columns.str.strip()
data = data.dropna(subset=["IPK"])

# --- Encoding Data ---
data_encoded = data.copy()
le_gender = LabelEncoder()
le_status_nikah = LabelEncoder()
le_kelulusan = LabelEncoder()

data_encoded['JENIS KELAMIN'] = le_gender.fit_transform(data_encoded['JENIS KELAMIN'])
data_encoded['STATUS NIKAH'] = le_status_nikah.fit_transform(data_encoded['STATUS NIKAH'])
data_encoded['STATUS KELULUSAN'] = le_kelulusan.fit_transform(data_encoded['STATUS KELULUSAN'])

fitur = ['UMUR', 'JENIS KELAMIN', 'STATUS NIKAH',
         'IPS 1', 'IPS 2', 'IPS 3', 'IPS 4',
         'IPS 5', 'IPS 6', 'IPS 7', 'IPS 8', 'IPK']

X = data_encoded[fitur]
y = data_encoded['STATUS KELULUSAN']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Model Training ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)

# --- Sidebar ---
st.sidebar.header("🔍 Navigasi")
section = st.sidebar.radio("Pilih Halaman", ["🏠 Tentang Aplikasi", "📁 Dataset", "📊 Visualisasi", "🧠 Prediksi", "📈 Evaluasi Model"])
st.sidebar.markdown("---")
st.sidebar.metric("🎯 Akurasi Model", f"{akurasi * 100:.2f}%")

# --- Tentang Aplikasi ---
if section == "🏠 Tentang Aplikasi":
    st.subheader("📘 Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini menggunakan teknologi **Artificial Intelligence (AI)**, khususnya algoritma *Random Forest Classifier*, 
    untuk memprediksi **kelulusan mahasiswa** berdasarkan data akademik seperti IPS dan IPK serta data pribadi seperti umur, jenis kelamin, dan status pernikahan.

    **Fitur Aplikasi:**
    - Menampilkan dataset mahasiswa
    - Visualisasi data kelulusan
    - Prediksi status kelulusan: *Tepat* atau *Terlambat*
    - Evaluasi kinerja model (akurasi, confusion matrix, dll)
    - Analisis korelasi antar fitur

    🛠️ Dikembangkan dengan Python & Streamlit.
    """)

# --- Dataset View ---
elif section == "📁 Dataset":
    st.subheader("📁 Dataset Awal")
    st.dataframe(data.head(), use_container_width=True)
    st.markdown(f"<div style='margin-top:10px;'>Jumlah Data: <strong>{data.shape[0]}</strong> baris</div>", unsafe_allow_html=True)

# --- Visualisasi ---
elif section == "📊 Visualisasi":
    st.subheader("📊 Visualisasi Kelulusan Mahasiswa")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📌 Distribusi Status Kelulusan")
        fig1, ax1 = plt.subplots()
        sns.countplot(x="STATUS KELULUSAN", data=data, ax=ax1, palette="Set2")
        ax1.set_xlabel("Status Kelulusan")
        ax1.set_ylabel("Jumlah")
        st.pyplot(fig1)

    with col2:
        st.markdown("#### 📌 Sebaran IPK Berdasarkan Kelulusan")
        fig2, ax2 = plt.subplots()
        sns.boxplot(x="STATUS KELULUSAN", y="IPK", data=data, palette="Set3")
        ax2.set_xlabel("Status Kelulusan")
        ax2.set_ylabel("IPK")
        st.pyplot(fig2)

# --- Prediksi ---
elif section == "🧠 Prediksi":
    st.subheader("🧠 Formulir Prediksi Kelulusan")
    st.markdown("Silakan isi data mahasiswa untuk memprediksi kelulusan:")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            umur = st.number_input("Umur", 17, 50, 22)
            jenis_kelamin = st.selectbox("Jenis Kelamin", le_gender.classes_)
            status_nikah = st.selectbox("Status Nikah", le_status_nikah.classes_)
        with col2:
            ips1 = st.slider("IPS 1", 0.0, 4.0, 3.0, 0.01)
            ips2 = st.slider("IPS 2", 0.0, 4.0, 3.0, 0.01)
            ips3 = st.slider("IPS 3", 0.0, 4.0, 3.0, 0.01)
            ips4 = st.slider("IPS 4", 0.0, 4.0, 3.0, 0.01)
        with col3:
            ips5 = st.slider("IPS 5", 0.0, 4.0, 3.0, 0.01)
            ips6 = st.slider("IPS 6", 0.0, 4.0, 3.0, 0.01)
            ips7 = st.slider("IPS 7", 0.0, 4.0, 3.0, 0.01)
            ips8 = st.slider("IPS 8", 0.0, 4.0, 3.0, 0.01)
            ipk = st.slider("IPK", 0.0, 4.0, 3.0, 0.01)

        submitted = st.form_submit_button("🔍 Prediksi Kelulusan")
        if submitted:
            input_data = np.array([[
                umur,
                le_gender.transform([jenis_kelamin])[0],
                le_status_nikah.transform([status_nikah])[0],
                ips1, ips2, ips3, ips4, ips5, ips6, ips7, ips8,
                ipk
            ]])
            input_scaled = scaler.transform(input_data)
            hasil = model.predict(input_scaled)
            label_hasil = le_kelulusan.inverse_transform(hasil)
            hasil_prediksi = label_hasil[0]

            if hasil_prediksi.lower() == "tepat":
                st.success(f"📢 Prediksi Kelulusan Mahasiswa: **{hasil_prediksi}**")
                st.markdown("✅ Mahasiswa diprediksi akan **lulus tepat waktu**. Ini menunjukkan performa akademik yang stabil dan konsisten. 💪")
            elif hasil_prediksi.lower() == "terlambat":
                st.error(f"📢 Prediksi Kelulusan Mahasiswa: **{hasil_prediksi}**")
                st.markdown("⚠️ Mahasiswa diprediksi akan **lulus terlambat**. Diperlukan perhatian terhadap studi dan manajemen waktu.")
            else:
                st.info(f"📢 Prediksi Kelulusan Mahasiswa: **{hasil_prediksi}**")

# --- Evaluasi Model ---
elif section == "📈 Evaluasi Model":
    st.subheader("📈 Evaluasi Kinerja Model")

    st.markdown(f"**🎯 Akurasi Model:** `{akurasi * 100:.2f}%`")
    if akurasi == 1.0:
        st.warning("⚠️ Akurasi 100% terdeteksi. Ini bisa jadi indikasi overfitting.")

    st.markdown("#### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig3, ax3 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    st.pyplot(fig3)

    st.markdown("#### Classification Report")
    report = classification_report(y_test, y_pred, target_names=le_kelulusan.classes_, output_dict=False)
    st.text(report)

    st.markdown("#### Feature Importance")
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"Fitur": fitur, "Pentingnya": importances})
    feat_df = feat_df.sort_values(by="Pentingnya", ascending=False)
    fig4, ax4 = plt.subplots()
    sns.barplot(data=feat_df, x="Pentingnya", y="Fitur", palette="viridis")
    st.pyplot(fig4)

    st.markdown("#### Korelasi Antar Fitur (Heatmap)")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    corr_matrix = data_encoded[fitur].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                fmt=".2f", square=True, linewidths=.5, cbar_kws={"shrink": .75}, ax=ax_corr)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    st.pyplot(fig_corr)
