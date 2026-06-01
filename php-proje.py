import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# ==========================================
# 1. VERİ YÜKLEME
# ==========================================
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Hamilelik', 'Glikoz', 'KanBasinci', 'CiltKalinligi', 'Insuilin', 'BMI', 'Soyagaci', 'Yas', 'Sonuc']
data = pd.read_csv(url, names=columns)

# ==========================================
# 2. VERİ ANALİZİ VE GÖRSELLEŞTİRME (EDA)
# ==========================================
print("--- Veri Seti Özeti ---")
print(data.describe())

# Korelasyon Matrisi Görselleştirme
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='RdYlGn', fmt='.2f')
plt.title("Özellikler Arasındaki Korelasyon Matrisi")
plt.show()

# ==========================================
# 3. VERİ ÖN İŞLEME
# ==========================================
X = data.drop('Sonuc', axis=1)
y = data['Sonuc']

# Verileri ölçeklendirme (Modellerin daha iyi öğrenmesi için)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Veriyi Eğitim ve Test seti olarak bölme (%80 Eğitim, %20 Test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ==========================================
# 4. ÇOKLU MODEL EĞİTİMİ VE DEĞERLENDİRME
# ==========================================
# Ödev dökümanına uygun olarak 3 farklı algoritma tanımlandı
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine (SVM)": SVC(kernel='linear')
}

print("\n--- MODEL PERFORMANS SONUÇLARI ---")

# Her modeli eğitiyor ve hocanın istediği tüm metrikleri hesaplıyoruz
for name, model in models.items():
    # Modeli eğit
    model.fit(X_train, y_train)
    # Tahmin yap
    preds = model.predict(X_test)
    
    # Metriklerin hesaplanması
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='weighted')
    rec = recall_score(y_test, preds, average='weighted')
    f1 = f1_score(y_test, preds, average='weighted')
    
    # Sonuçların hocanın istediği formatta yazdırılması
    print(f"\nModel: {name}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("-" * 30)

# ==========================================
# 5. EN İYİ MODEL DETAYLANDIRILMASI (Random Forest)
# ==========================================
best_model = models["Random Forest"]
y_pred = best_model.predict(X_test)

# Karışıklık Matrisi Görselleştirme
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek Değer')
plt.title('Hata Matrisi (Confusion Matrix) - Random Forest')
plt.show()

# Özellik Önemi Grafiği (Feature Importance)
importances = pd.Series(best_model.feature_importances_, index=columns[:-1])
plt.figure(figsize=(8, 5))
importances.nlargest(5).plot(kind='barh', color='skyblue')
plt.title("Diyabeti Belirleyen En Önemli 5 Faktör")
plt.xlabel("Önem Skoru")
plt.show()
