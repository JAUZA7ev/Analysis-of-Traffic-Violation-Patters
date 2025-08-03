import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import folium
from streamlit_folium import folium_static

st.set_page_config(page_title="Analisis Pola Pelanggaran Lalu Lintas", layout="wide")
st.title("🚦 Klasterisasi Pelanggaran Lalu Lintas di Pekalongan")

uploaded_file = st.file_uploader("📤 Upload Data Pelanggaran (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Dataset")
    st.dataframe(df.head())

    st.subheader("📌 Pilih Fitur untuk Klasterisasi")
    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    selected_features = st.multiselect(
        "Pilih 2 fitur numerik (wajib mengandung koordinat jika ingin tampilkan peta)",
        numeric_columns,
        default=["Latitude", "Longitude"]
    )

    if len(selected_features) != 2:
        st.warning("Pilih tepat 2 fitur numerik untuk klasterisasi.")
    else:
        X = df[selected_features]

        # Silhouette Score untuk mencari jumlah klaster terbaik
        st.subheader("📊 Evaluasi Silhouette Score (Optimal K)")
        silhouette_scores = []
        k_range = range(2, 8)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
            score = silhouette_score(X, kmeans.labels_)
            silhouette_scores.append(score)

        fig1, ax1 = plt.subplots()
        sns.lineplot(x=list(k_range), y=silhouette_scores, marker='o', ax=ax1)
        ax1.set_xlabel("Jumlah Klaster (K)")
        ax1.set_ylabel("Silhouette Score")
        ax1.set_title("Evaluasi K dengan Silhouette Score")
        st.pyplot(fig1)

        # Slider jumlah klaster
        k = st.slider("Pilih jumlah klaster", min_value=2, max_value=7, value=3)
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X)

        st.subheader("🧠 Data dengan Klaster")
        st.dataframe(df[[*selected_features, 'Cluster']].head())

        st.subheader("🗺️ Peta Klasterisasi Pelanggaran")
        center_lat = df['Latitude'].mean()
        center_lon = df['Longitude'].mean()

        m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'cadetblue']
        for _, row in df.iterrows():
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=6,
                color=colors[int(row['Cluster']) % len(colors)],
                fill=True,
                fill_opacity=0.7,
                popup=f"{row['Nama_Pelanggar']} - {row['Jenis_Pelanggaran']} - Cluster {row['Cluster']}"
            ).add_to(m)

        folium_static(m)

        st.subheader("📋 Statistik per Klaster")
        st.dataframe(df.groupby('Cluster')[selected_features].mean().reset_index())