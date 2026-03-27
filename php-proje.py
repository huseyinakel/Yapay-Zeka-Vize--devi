import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Veriyi Yükleme [cite: 25]
df = pd.read_csv('WA_Fn-UseC-Telco-Customer-Churn.csv')

# 2. Genel Tanıtım ve Özelliklerin Açıklanması [cite: 25, 26]
print("--- Veri Seti İlk 5 Satır ---")
print(df.head())
print("\n--- Veri Seti Bilgileri ---")
print(df.info())

# 3. Eksik Veri Analizi [cite: 27]
print("\n--- Eksik Veri Sayısı ---")
print(df.isnull().sum())

# 4. Veri Görselleştirme (Churn Dağılımı) 
plt.figure(figsize=(8, 5))
sns.countplot(x='Churn', data=df)
plt.title('Müşteri Kaybı (Churn) Dağılımı')
plt.show()

# 5. Aykırı Değer Analizi (Aylık Ücretler için Boxplot) [cite: 28, 30]
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['MonthlyCharges'])
plt.title('Aylık Ücretler Aykırı Değer Analizi')
plt.show()

print("\nVize aşaması analizleri başarıyla tamamlandı!")