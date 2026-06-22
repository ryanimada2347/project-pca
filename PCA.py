import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="PCA: Kompresi Citra & Deteksi Kemiripan Wajah",
    page_icon="🧠",
    layout="wide"
)

IMG_SIZE = (100, 100)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ============================================
# FUNGSI-FUNGSI INTI (DIPAKAI DI KEDUA TAB)
# ============================================
def preprocess(uploaded_file, target_size=IMG_SIZE):
    """Membaca file upload, deteksi wajah, crop, resize, grayscale."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Gambar tidak dapat dibaca.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    detected = False
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        gray = gray[y:y+h, x:x+w]
        detected = True

    gray = cv2.resize(gray, target_size)
    return gray, detected


def bangun_ruang_pca(data_matrix):
    """Membangun mean face dan eigenfaces dari matriks data dataset."""
    mean_face = np.mean(data_matrix, axis=0)
    centered_data = data_matrix - mean_face

    A = centered_data
    L = np.dot(A, A.T)
    eigenvalues, eigenvectors_small = np.linalg.eigh(L)

    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors_small = eigenvectors_small[:, idx]
    eigenvalues = np.clip(eigenvalues, a_min=1e-10, a_max=None)

    eigenfaces = np.dot(A.T, eigenvectors_small)
    eigenfaces = eigenfaces / np.linalg.norm(eigenfaces, axis=0)

    return mean_face, centered_data, eigenfaces, eigenvalues


def kompresi_dan_rekonstruksi(image_vector, mean_face, eigenfaces, k):
    """Kompresi gambar dengan k komponen PCA, lalu rekonstruksi kembali."""
    k = min(k, eigenfaces.shape[1])
    pca_space_k = eigenfaces[:, :k]
    centered = image_vector - mean_face

    weights_k = np.dot(centered, pca_space_k)
    reconstructed = mean_face + np.dot(weights_k, pca_space_k.T)
    reconstructed = np.clip(reconstructed, 0, 255)

    return reconstructed, weights_k


def hitung_psnr(original, reconstructed):
    """Peak Signal-to-Noise Ratio (dB)."""
    mse = np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


def project_to_pca(face_vector, mean_face, pca_space):
    centered = face_vector - mean_face
    return np.dot(centered, pca_space)


# ============================================
# SESSION STATE INIT
# ============================================
if "pca_built" not in st.session_state:
    st.session_state.pca_built = False
if "log_perbandingan" not in st.session_state:
    st.session_state.log_perbandingan = []


# ============================================
# UI UTAMA
# ============================================
st.title("🧠 Sistem PCA: Kompresi Citra & Deteksi Kemiripan Wajah")

tab1, tab2 = st.tabs(["4.1 Kompresi Citra", "4.2 Deteksi Kemiripan Wajah"])

# ============================================
# SIDEBAR: UPLOAD DATASET (DIPAKAI KEDUA TAB)
# ============================================
st.sidebar.header("Dataset Wajah (Training PCA)")
dataset_files = st.sidebar.file_uploader(
    "Upload minimal 5-10 gambar wajah",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="dataset_uploader"
)

if dataset_files:
    cache_key = tuple(sorted(f.name for f in dataset_files))

    if st.session_state.get("dataset_cache_key") != cache_key:
        with st.spinner("Memproses dataset dan membangun ruang PCA..."):
            vectors = []
            filenames = []
            detection_status = []

            for f in dataset_files:
                try:
                    face, detected = preprocess(f)
                    vectors.append(face.flatten().astype(np.float64))
                    filenames.append(f.name)
                    detection_status.append(detected)
                except Exception as e:
                    st.sidebar.warning(f"Gagal memproses {f.name}: {e}")

            if len(vectors) < 2:
                st.sidebar.error("Dataset minimal 2 gambar valid.")
                st.stop()

            data_matrix = np.array(vectors)
            mean_face, centered_data, eigenfaces, eigenvalues = bangun_ruang_pca(data_matrix)

            explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
            cumulative_variance = np.cumsum(explained_variance_ratio)
            K = np.argmax(cumulative_variance >= 0.95) + 1
            K = max(K, 2)
            K = min(K, eigenfaces.shape[1])
            pca_space = eigenfaces[:, :K]

            weights_dataset = np.dot(centered_data, pca_space)
            pairwise_distances = []
            N = data_matrix.shape[0]
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(weights_dataset[i] - weights_dataset[j])
                    pairwise_distances.append(d)
            threshold_auto = min(pairwise_distances) if pairwise_distances else 0.0

            st.session_state.dataset_cache_key = cache_key
            st.session_state.data_matrix = data_matrix
            st.session_state.mean_face = mean_face
            st.session_state.eigenfaces = eigenfaces
            st.session_state.pca_space = pca_space
            st.session_state.K = K
            st.session_state.cumulative_variance = cumulative_variance
            st.session_state.threshold_auto = threshold_auto
            st.session_state.filenames = filenames
            st.session_state.detection_status = detection_status
            st.session_state.pca_built = True

    st.sidebar.success(f"Dataset siap: {len(dataset_files)} gambar, K={st.session_state.K} komponen PCA")
else:
    st.sidebar.info("Upload dataset wajah untuk mulai.")


# ============================================
# TAB 1: KOMPRESI CITRA
# ============================================
with tab1:
    st.header("4.1 Hasil Kompresi Citra")

    if not st.session_state.pca_built:
        st.warning("Upload dataset wajah di sidebar terlebih dahulu.")
    else:
        img_uji = st.file_uploader(
            "Upload satu gambar wajah untuk dikompresi",
            type=["jpg", "jpeg", "png"],
            key="kompresi_uploader"
        )

        if img_uji:
            face_asli, detected = preprocess(img_uji)
            vector_asli = face_asli.flatten().astype(np.float64)

            max_k = st.session_state.eigenfaces.shape[1]
            daftar_k_default = [k for k in [5, 20, 50, 100] if k <= max_k]
            if not daftar_k_default:
                daftar_k_default = [max_k]

            daftar_k = st.multiselect(
                "Pilih nilai k yang akan diuji",
                options=list(range(1, max_k + 1)),
                default=daftar_k_default
            )

            if daftar_k:
                ukuran_asli_kb = vector_asli.size * vector_asli.itemsize / 1024

                hasil_tabel = []
                citra_rekonstruksi = {}

                for k in sorted(daftar_k):
                    reconstructed, weights_k = kompresi_dan_rekonstruksi(
                        vector_asli, st.session_state.mean_face, st.session_state.eigenfaces, k
                    )
                    citra_rekonstruksi[k] = reconstructed.reshape(IMG_SIZE)

                    ukuran_kompresi_kb = weights_k.size * weights_k.itemsize / 1024
                    rasio_tersimpan = (ukuran_kompresi_kb / ukuran_asli_kb) * 100
                    psnr = hitung_psnr(vector_asli, reconstructed)

                    if psnr > 35:
                        kualitas = "Sangat Baik"
                    elif psnr > 25:
                        kualitas = "Baik"
                    elif psnr > 15:
                        kualitas = "Cukup"
                    else:
                        kualitas = "Buruk"

                    hasil_tabel.append({
                        "Nilai k": k,
                        "Rasio Data Tersimpan (%)": round(rasio_tersimpan, 2),
                        "Ukuran File (KB)": round(ukuran_kompresi_kb, 4),
                        "PSNR (dB)": round(psnr, 2),
                        "Kualitas Visual": kualitas
                    })

                df_hasil = pd.DataFrame(hasil_tabel)

                st.write(f"Ukuran citra asli: **{ukuran_asli_kb:.4f} KB** ({IMG_SIZE[0]}x{IMG_SIZE[1]} pixel, grayscale)")
                st.dataframe(df_hasil, use_container_width=True)

                st.subheader("Perbandingan Visual: Citra Asli vs Hasil Kompresi")
                n_total = len(daftar_k) + 1
                fig, axes = plt.subplots(1, n_total, figsize=(3.5 * n_total, 3.5))

                axes[0].imshow(face_asli, cmap="gray")
                axes[0].set_title("Citra Asli", fontweight="bold")
                axes[0].axis("off")

                for i, k in enumerate(sorted(daftar_k)):
                    axes[i + 1].imshow(citra_rekonstruksi[k], cmap="gray")
                    psnr_val = df_hasil[df_hasil["Nilai k"] == k]["PSNR (dB)"].values[0]
                    axes[i + 1].set_title(f"k={k}\nPSNR={psnr_val:.2f}dB")
                    axes[i + 1].axis("off")

                plt.tight_layout()
                st.pyplot(fig)

                st.subheader("Grafik Hubungan Nilai k")
                col1, col2 = st.columns(2)
                with col1:
                    fig1, ax1 = plt.subplots(figsize=(5, 3.5))
                    ax1.plot(df_hasil["Nilai k"], df_hasil["PSNR (dB)"], marker="o", color="green")
                    ax1.set_xlabel("Nilai k")
                    ax1.set_ylabel("PSNR (dB)")
                    ax1.set_title("Kualitas Visual vs k")
                    ax1.grid(alpha=0.3)
                    st.pyplot(fig1)

                with col2:
                    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
                    ax2.plot(df_hasil["Nilai k"], df_hasil["Ukuran File (KB)"], marker="o", color="blue")
                    ax2.set_xlabel("Nilai k")
                    ax2.set_ylabel("Ukuran File (KB)")
                    ax2.set_title("Ukuran Data vs k")
                    ax2.grid(alpha=0.3)
                    st.pyplot(fig2)


# ============================================
# TAB 2: DETEKSI KEMIRIPAN WAJAH
# ============================================
with tab2:
    st.header("4.2 Hasil Deteksi Kemiripan Wajah")

    if not st.session_state.pca_built:
        st.warning("Upload dataset wajah di sidebar terlebih dahulu.")
    else:
        st.write(f"Threshold otomatis dari dataset: **{st.session_state.threshold_auto:.4f}**")

        pakai_manual = st.checkbox("Override threshold manual")
        if pakai_manual:
            threshold = st.slider("Threshold manual", 0.0, 5000.0, float(st.session_state.threshold_auto))
        else:
            threshold = st.session_state.threshold_auto

        col1, col2 = st.columns(2)
        with col1:
            img1 = st.file_uploader("Upload wajah_A", type=["jpg", "jpeg", "png"], key="wajahA")
        with col2:
            img2 = st.file_uploader("Upload wajah_B", type=["jpg", "jpeg", "png"], key="wajahB")

        nama_pasangan = st.text_input("Nama pasangan (untuk catatan tabel)", value=f"Pasangan {len(st.session_state.log_perbandingan) + 1}")

        if img1 and img2:
            face1, detected1 = preprocess(img1)
            face2, detected2 = preprocess(img2)

            z_A = project_to_pca(face1.flatten().astype(np.float64), st.session_state.mean_face, st.session_state.pca_space)
            z_B = project_to_pca(face2.flatten().astype(np.float64), st.session_state.mean_face, st.session_state.pca_space)

            distance = np.linalg.norm(z_A - z_B)
            is_match = distance < threshold
            hasil_text = "WAJAH SAMA (Match)" if is_match else "WAJAH BERBEDA (Not Match)"

            c1, c2 = st.columns(2)
            with c1:
                st.image(face1, caption=f"wajah_A ({'terdeteksi' if detected1 else 'fallback'})")
            with c2:
                st.image(face2, caption=f"wajah_B ({'terdeteksi' if detected2 else 'fallback'})")

            st.write(f"Jarak Euclidean: **{distance:.4f}**")
            st.write(f"Threshold: **{threshold:.4f}**")

            if is_match:
                st.success(hasil_text)
            else:
                st.error(hasil_text)

            if st.button("Simpan ke tabel log"):
                st.session_state.log_perbandingan.append({
                    "Waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Pasangan Wajah": nama_pasangan,
                    "Jarak Euclidean": round(distance, 4),
                    "Threshold (otomatis)": round(threshold, 4),
                    "Hasil": hasil_text
                })
                st.success("Tersimpan ke tabel log di bawah.")

        st.subheader("Tabel Log Hasil Pengujian (untuk laporan 4.2)")
        if st.session_state.log_perbandingan:
            df_log = pd.DataFrame(st.session_state.log_perbandingan)
            st.dataframe(df_log, use_container_width=True)

            csv = df_log.to_csv(index=False).encode("utf-8")
            st.download_button("Download tabel sebagai CSV", csv, "log_perbandingan_wajah.csv", "text/csv")

            if st.button("Hapus semua log"):
                st.session_state.log_perbandingan = []
                st.rerun()
        else:
            st.info("Belum ada hasil yang disimpan. Bandingkan sepasang wajah lalu klik 'Simpan ke tabel log'.")