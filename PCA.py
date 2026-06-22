import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="Pengenalan Wajah PCA",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Sistem Pengenalan Wajah Menggunakan PCA")

IMG_SIZE = (100, 100)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


def preprocess(uploaded_file):
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    img = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    if len(faces) > 0:
        x, y, w, h = max(
            faces,
            key=lambda f: f[2] * f[3]
        )
        gray = gray[y:y+h, x:x+w]

    gray = cv2.resize(
        gray,
        IMG_SIZE
    )

    return gray


st.header("1. Upload Dataset Wajah")

dataset_files = st.file_uploader(
    "Upload minimal 5 gambar wajah",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if dataset_files:

    faces = []

    for file in dataset_files:
        try:
            img = preprocess(file)
            faces.append(
                img.flatten()
            )
        except:
            st.warning(
                f"Gagal membaca {file.name}"
            )

    if len(faces) < 2:
        st.error(
            "Dataset minimal 2 gambar."
        )
        st.stop()

    data_matrix = np.array(faces)

    st.success(
        f"Dataset berhasil dimuat: "
        f"{len(faces)} gambar"
    )

    mean_face = np.mean(
        data_matrix,
        axis=0
    )

    centered_data = (
        data_matrix -
        mean_face
    )

    pca = PCA(
        n_components=0.95
    )

    weights_dataset = pca.fit_transform(
        centered_data
    )

    K = pca.n_components_

    st.write(
        f"Jumlah komponen PCA (K): {K}"
    )

    fig, ax = plt.subplots(
        figsize=(3, 3)
    )

    ax.imshow(
        mean_face.reshape(
            IMG_SIZE
        ),
        cmap="gray"
    )

    ax.set_title(
        "Mean Face"
    )

    ax.axis("off")

    st.pyplot(fig)

    st.header(
        "2. Upload Gambar Pembanding"
    )

    col1, col2 = st.columns(2)

    with col1:
        img1 = st.file_uploader(
            "Upload wajah_A",
            type=[
                "jpg",
                "jpeg",
                "png"
            ]
        )

    with col2:
        img2 = st.file_uploader(
            "Upload wajah_B",
            type=[
                "jpg",
                "jpeg",
                "png"
            ]
        )

    if img1 and img2:

        face1 = preprocess(img1)
        face2 = preprocess(img2)

        vec1 = (
            face1
            .flatten()
            - mean_face
        )

        vec2 = (
            face2
            .flatten()
            - mean_face
        )

        z1 = pca.transform(
            vec1.reshape(
                1,
                -1
            )
        )

        z2 = pca.transform(
            vec2.reshape(
                1,
                -1
            )
        )

        distance = np.linalg.norm(
            z1 - z2
        )

        pairwise = []

        for i in range(
            len(weights_dataset)
        ):
            for j in range(
                i + 1,
                len(weights_dataset)
            ):
                d = np.linalg.norm(
                    weights_dataset[i]
                    -
                    weights_dataset[j]
                )
                pairwise.append(d)

        threshold = min(
            pairwise
        )

        st.subheader(
            "Hasil Perbandingan"
        )

        c1, c2 = st.columns(2)

        with c1:
            st.image(
                face1,
                caption="wajah_A",
                use_container_width=True
            )

        with c2:
            st.image(
                face2,
                caption="wajah_B",
                use_container_width=True
            )

        st.write(
            f"Jarak Euclidean: "
            f"{distance:.4f}"
        )

        st.write(
            f"Threshold: "
            f"{threshold:.4f}"
        )

        if distance < threshold:
            st.success(
                "WAJAH SAMA (Match)"
            )
        else:
            st.error(
                "WAJAH BERBEDA (Not Match)"
            )