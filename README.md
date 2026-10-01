Analisis Sentimen Ulasan Aplikasi Stockbit

Analisis sentimen berbahasa Indonesia terhadap 10.000 ulasan aplikasi Stockbit di Google Play Store. Ulasan di-scrape, dibersihkan, diberi label sentimen (positive/negative) dengan pendekatan lexicon, lalu dipakai untuk melatih dan membandingkan beberapa model klasifikasi. Model terbaik, SVM dengan fitur TF-IDF, disimpan untuk memprediksi sentimen teks baru.

Struktur Proyek
.
scraping.ipynb                 scraping ulasan dari Google Play Store
model.ipynb                    preprocessing, pelabelan, pelatihan, dan evaluasi model
testing.ipynb                  prediksi teks baru memakai model tersimpan
scrapped_data_stockbit.csv     dataset hasil scraping
model_svm_stockbit.pkl         model SVM terlatih
tfidf_vectorizer_stockbit.pkl  TF-IDF vectorizer terlatih
requirements.txt
README.md
Alur Kerja
1. Scraping

Menggunakan google-play-scraper dengan parameter aplikasi com.stockbit.android, bahasa id, negara id, urutan MOST_RELEVANT, dan jumlah 10.000 ulasan. Hasilnya disimpan di scrapped_data_stockbit.csv (11 kolom).

2. Preprocessing

Kolom yang dipakai: userName, score, at, content, thumbsUpCount. Tahapan pembersihan teks:

Tahap	Keterangan
Cleaning	Hapus mention, hashtag, RT, URL, angka, tanda baca, dan baris baru
Case folding	Ubah ke huruf kecil
Normalisasi slang	Ganti istilah pasar saham, misalnya cuan menjadi untung, boncos menjadi rugi, porto menjadi portofolio
Tokenizing	word_tokenize dari NLTK
Stopword removal	Stopword Indonesia dan Inggris, ditambah saham, emiten, idx, ihsg, reksadana, rupiah, rp

Fungsi stemming Sastrawi ada di notebook tetapi tidak dipakai pada pipeline akhir.

3. Pelabelan

Label dibuat dari lexicon kata positif dan negatif berbahasa Indonesia (dari repositori angelmetanosaa/dataset). Bobot kata dijumlahkan per ulasan: skor >= 0 menjadi positive, skor < 0 menjadi negative.

Distribusi label: positive 6.253 (62,5%), negative 3.747 (37,5%).

4. Pemodelan
Model	Fitur	Split	Akurasi Train	Akurasi Test
Random Forest (200 pohon)	TF-IDF	70/30	99,44%	87,00%
SVM (SVC)	TF-IDF	80/20	98,16%	89,70%
Random Forest	Word2Vec	80/20	99,95%	72,00%

Konfigurasi TF-IDF: max_features=1000, min_df=2, max_df=0.95, ngram_range=(1,2).

SVM dengan TF-IDF dipilih karena akurasi test tertinggi dan selisih train-test paling kecil. Model dan vectorizer disimpan dengan pickle.

Instalasi

Python 3.11, dengan koneksi internet untuk scraping, unduhan data NLTK, dan pengambilan lexicon.

bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt

Unduh resource NLTK:

python
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')

Urutan menjalankan notebook: scraping.ipynb, model.ipynb, lalu testing.ipynb. Scraping bisa dilewati jika memakai scrapped_data_stockbit.csv yang sudah ada.

Contoh Penggunaan
python
import pickle

with open('model_svm_stockbit.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf_vectorizer_stockbit.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

texts = [
    "Aplikasi stockbit mantap, fiturnya lengkap bikin cuan terus!",
    "Analisis teknikalnya sering error, jadi boncos deh porto saya.",
]

preds = model.predict(vectorizer.transform(texts))
for text, label in zip(texts, preds):
    print(f"{text} -> {label}")

Keluaran:

Aplikasi stockbit mantap, fiturnya lengkap bikin cuan terus! -> positive
Analisis teknikalnya sering error, jadi boncos deh porto saya. -> negative

Model dilatih pada teks yang sudah di-preprocessing, jadi teks baru sebaiknya melewati tahapan yang sama sebelum vectorizer.transform(). File .pkl dibuat dengan scikit-learn==1.7.2; gunakan versi yang sama agar bisa dimuat.

Keterbatasan
Label berasal dari lexicon, bukan anotasi manusia. Akurasi menunjukkan seberapa baik model meniru lexicon, bukan sentimen yang sebenarnya.
Hanya dua kelas. Ulasan dengan skor 0 ikut masuk kelas positive.
Selisih akurasi train dan test pada SVM sekitar 8 poin, menandakan overfitting ringan.
Evaluasi baru berupa akurasi; precision, recall, F1, dan confusion matrix belum ada.
Data diambil dengan urutan MOST_RELEVANT, sehingga belum tentu mewakili seluruh ulasan.
Teknologi

Python, pandas, NumPy, scikit-learn, NLTK, Sastrawi, gensim, google-play-scraper, matplotlib, seaborn, WordCloud.
