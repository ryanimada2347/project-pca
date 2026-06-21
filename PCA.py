import streamlit as st

st.title("Kalkulator Sederhana")

angka1 = st.number_input("Masukkan angka pertama")
angka2 = st.number_input("Masukkan angka kedua")

hasil = angka1 + angka2

st.write("Hasil:", hasil)