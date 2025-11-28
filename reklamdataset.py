
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Veri seti
df = pd.read_csv("C:/Users/oguzhan/PycharmProjects/artikprofosyonel/Social_Network_Ads.csv")

print("--- Veri Kümesi İlk 5 Satır ---")
print(df.head())


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
# Veri seti
df = pd.read_csv('Social_Network_Ads.csv')

# X: Bağımsız Değişkenlerim (Age, EstimatedSalary) - 0. ve 1. indeks
# y: Hedef değişken (Purchased) - Son indeks (2)
X = df.iloc[:, [0, 1]].values   # Sadece Age ve EstimatedSalary
y = df.iloc[:, -1].values       # Purchased (Satın Alma)

# EĞİTİM VE TEST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# StandardScaler
sc = StandardScaler()

# X sütunları (Age ve EstimatedSalary)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print("--- Özellik Ölçeklendirme Tamamlandı ---")

# SVM
classifier = SVC(kernel = 'rbf', random_state = 0)

# MODELİ EĞİTİYORUM
classifier.fit(X_train, y_train)

print("--- SVM Modeli Eğitildi ---")

# Test
y_pred = classifier.predict(X_test)

# Karışıklık Matrisi
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("\n--- Değerlendirme Metrikleri ---")
print(f"Karışıklık Matrisi:\n{cm}")
print(f"Model Doğruluk Skoru: {accuracy * 100:.2f}%")
# Karışıklık Matrisi

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

plt.figure(figsize=(8, 6))
ax = sns.heatmap(
    cm,
    annot=True,        # Sayıları hücrelerin içine yaz
    fmt='d',           # Sayıları tam sayı olarak biçimlendir
    cmap='Blues',
    cbar=False,
    linewidths=2,
    linecolor='white', # Çizgi
    annot_kws={"size": 16, "color": "black"}
)

# Eksen etiketleri
plt.xticks(ticks=[0.5, 1.5], labels=['Tahmin: Satın Almadı (0)', 'Tahmin: Satın Aldı (1)'], fontsize=12)
plt.yticks(ticks=[0.5, 1.5], labels=['Gerçek: Satın Almadı (0)', 'Gerçek: Satın Aldı (1)'], rotation=90, va='center', fontsize=12)

# Başlık
plt.title(f'SVM Karışıklık Matrisi (Doğruluk: {accuracy:.4f})', fontsize=16)
plt.xlabel('Tahmin Edilen Sınıf', fontsize=14)
plt.ylabel('Gerçek Sınıf', fontsize=14)

# GRAFIK
for i in range(2):
    for j in range(2):
        # HÜCRERENK
        plt.text(j + 0.5, i + 0.5, str(cm[i, j]),
                 ha='center', va='center', fontsize=16, color='black')


plt.tight_layout()
plt.show()