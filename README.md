# Yapay Zeka Temelleri - Vize/Final Projesi

Bu proje, Yapay Zeka Temelleri dersi kapsamında uçtan uca bir makine öğrenmesi sürecini (veri ön işleme, EDA, modelleme ve değerlendirme) deneyimlemek amacıyla geliştirilmiştir.

## 📌 Projenin Amacı
Bu projenin amacı, hastaların çeşitli tıbbi ölçümlerini (Glikoz, Kan Basıncı, BMI vb.) analiz ederek **Diyabet (Hastalık Teşhisi)** tahmini yapmaktır. Proje kapsamında eksik ve aykırı değer analizleri yapılmış, veri görselleştirme adımları tamamlanmış ve farklı makine öğrenmesi modelleriyle sınıflandırma performansı test edilmiştir.

## 📊 Veri Seti Bilgileri
* **Veri Seti Adı:** Pima Indians Diabetes Dataset
* **Veri Seti Kaynağı:** [Kaggle - Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
* **Veri Seti Doğrudan Linki (CSV):** [GitHub / jbrownlee - pima-indians-diabetes.data.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)
* **Özellikler (Features):**
  * `Hamilelik`: Hamile kalma sayısı.
  * `Glikoz`: Plazma glikoz konsantrasyonu.
  * `KanBasinci`: Diastolik kan basıncı (mm Hg).
  * `CiltKalinligi`: Triceps cilt kıvrım kalınlığı (mm).
  * `Insuilin`: 2 saatlik serum insülini (mu U/ml).
  * `BMI`: Vücut kitle indeksi (kilo / boy^2).
  * `Soyagaci`: Diyabet soy ağacı işlevi.
  * `Yas`: Yaş (yıl).
  * `Sonuc` (Hedef Değişken): Diyabet durumu (0: Sağlıklı, 1: Diyabet Hastası).

## 🛠️ Yapılan İşlemler (EDA & Ön İşleme)
* `data.describe()` fonksiyonu kullanılarak verilerin merkezi eğilim ve dağılım ölçüleri incelenmiştir.
* Özelliklerin birbiriyle olan ilişkisini anlamak için **Seaborn Korelasyon Matrisi (Heatmap)** çizdirilmiştir.
* Modellerin daha dengeli ve performanslı öğrenmesi adına `StandardScaler` kullanılarak veriler ölçeklendirilmiştir.
* Veri seti %80 Eğitim ve %20 Test seti olacak şekilde bölünmüştür.

## 🤖 Kullanılan Modeller ve Sonuçlar
Projenin final aşamasında veri seti **Lojistik Regresyon**, **Rastgele Orman (Random Forest)** ve **Destek Vektör Makineleri (SVM)** algoritmaları ile eğitilmiştir. Modellerin başarısı ödev dökümanında zorunlu tutulan sınıflandırma metrikleriyle ölçülmüştür:

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| Lojistik Regresyon | 0.7532 | 0.7544 | 0.7532 | 0.7537 |
| Random Forest | 0.7208 | 0.7208 | 0.7208 | 0.7208 |
| Support Vector Machine (SVM) | 0.7597 | 0.7591 | 0.7597 | 0.7593 |

*(Not: Yukarıdaki değerler kod çalıştırıldığında elde edilen net test sonuçlarıdır.)*

## 📈 Sonuçların Değerlendirilmesi
* Yapılan testler ve metrik analizleri sonucunda **Support Vector Machine (SVM)** algoritması, %75.97 doğruluk (Accuracy) ve dengeli F1-Skoru ile en başarılı model olmuştur.
* **Random Forest** modelinin "Feature Importance" (Özellik Önemi) çıktısına göre, diyabet teşhisinde en belirleyici ilk iki faktörün **Glikoz seviyesi** ve **BMI (Vücut Kitle İndeksi)** olduğu tespit edilmiştir.
* Hata Matrisi (Confusion Matrix) incelendiğinde modellerin doğru ve yanlış tahmin dağılımları başarıyla raporlanmıştır.
