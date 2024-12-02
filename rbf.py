# Import library yang diperlukan
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

# Load dataset spam_ham_dataset dari CSV
df = pd.read_csv('spam_ham_dataset.csv')  # Pastikan file 'spam_ham_dataset.csv' ada di direktori yang benar

# Konversi target kelas "label" menjadi numerik (spam = 1, ham = 0)
df['label_num'] = df['label'].map({'spam': 1, 'ham': 0})

# Tampilkan dataframe
print("DATASET SPAM HAM".center(100, "="))
print(df.head())  # Tampilkan beberapa baris awal untuk memastikan data

# Memisahkan label dan parameter
x = df['text']  # Kolom 'text' sebagai fitur
y = df['label_num']  # Kolom 'label_num' sebagai label

# Tampilkan data variabel (fitur)
print("DATA VARIABEL".center(100, "="))
print(x)

# Tampilkan data kelas (target)
print("DATA KELAS".center(100, "="))
print(y)

# Pemisahan data menjadi training dan testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=100)

# Tampilkan data variabel dan kelas untuk training dan testing
print("DATA VARIABEL TRAINING".center(100, "="))
print(x_train)

print("DATA VARIABEL TESTING".center(100, "="))
print(x_test)

print("DATA KELAS TRAINING".center(100, "="))
print(y_train)

print("DATA KELAS TESTING".center(100, "="))
print(y_test)

# Karena SVM hanya bekerja dengan fitur numerik, kita perlu mengubah teks menjadi fitur numerik.
# Menggunakan CountVectorizer untuk mengubah teks menjadi vektor frekuensi kata
vectorizer = CountVectorizer()

# Transformasi teks menjadi fitur numerik
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

# Pemodelan SVM
model = SVC()
model.fit(x_train_vec, y_train)

# Prediksi menggunakan data testing
y_pred = model.predict(x_test_vec)

# Hasil prediksi SVM
print("HASIL PREDIKSI SVM".center(100, "="))
print(y_pred)

# Evaluasi menggunakan confusion matrix dan akurasi
print("HASIL CONFUSION MATRIX".center(100, "="))
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualisasi confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix SVM')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Evaluasi akurasi
accuracy = accuracy_score(y_test, y_pred)
print("AKURASI SVM".center(100, "="))
print("Akurasi:", accuracy)

# Tambahkan evaluasi untuk Precision, Recall, dan F1-Score
print("HASIL EVALUASI METRIK".center(100, "="))
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# Visualisasi akurasi dalam bentuk pie chart
plt.figure(figsize=(6, 6))
plt.pie(
    [accuracy, 1 - accuracy],
    labels=['Akurasi', 'Kesalahan'],
    autopct='%1.1f%%',
    startangle=140,
    colors=['#66c2a5', '#fc8d62']
)
plt.title('Akurasi SVM dalam Memprediksi Spam vs Ham')
plt.show()
