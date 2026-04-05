import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Veri Yükleme
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Hamilelik', 'Glikoz', 'KanBasinci', 'CiltKalinligi', 'Insuilin', 'BMI', 'Soyagaci', 'Yas', 'Sonuc']
data = pd.read_csv(url, names=columns)

# 2. Veri Analizi ve Görselleştirme (Sunum İçin Kritik!)
print("--- Veri Seti Özeti ---")
print(data.describe())

plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='RdYlGn', fmt='.2f')
plt.title("Özellikler Arasındaki Korelasyon Matrisi")
plt.show()

# 3. Veri Ön İşleme
X = data.drop('Sonuc', axis=1)
y = data['Sonuc']

# Verileri ölçeklendirme (Modellerin daha iyi öğrenmesi için)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Çoklu Model Eğitimi ve Karşılaştırma
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(kernel='linear')
}

results = {}

print("\n--- Model Performansları ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"{name} Doğruluk Skoru: %{acc*100:.2f}")

# 5. En İyi Modeli Detaylandırma (Random Forest Seçildi)
best_model = models["Random Forest"]
y_pred = best_model.predict(X_test)

# Karışıklık Matrisi Görselleştirme
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek Değer')
plt.title('Hata Matrisi (Confusion Matrix)')
plt.show()

# 6. Özellik Önemi (Hocanın En Çok Seveceği Kısım)
importances = pd.Series(best_model.feature_importances_, index=columns[:-1])
importances.nlargest(5).plot(kind='barh')
plt.title("Diyabeti Belirleyen En Önemli 5 Faktör")
plt.show()
