# reklamverisitahmin
Social_Network_Ads : https://www.kaggle.com/datasets/ayeshaimran123/social-network-ads/data verisi kullanılarak 
Müşterilerin yaş ve tahmini maaş bilgileri alınmış , reklamı izledikten sonra ürünü satın alıp almadıkları incelenmiştir. Bu veriler kullanılarak bir makine öğrenimi modeli geliştirdim .
SVM, tam olarak bu durumlar için tasarlanmış bir model olacaktı .

En Geniş Marjı bulur. Yani, iki sınıf arasında en güvenli ve en büyük boşluğu bırakan karar çizgisini benimsedim.

RBF Kernel (Çekirdek Fonksiyonu) kullanarak, verilerimin doğrusal olmayan karmaşık ilişkilerini bile yakalamayı başardım.
Metrik,Sonuç
Doğruluk (Accuracy),%93.00
Yanlış Pozitif (FP) Hatası,4 / 100
Yanlış Negatif (FN) Hatası,3 / 100 
Geliştirdiğim model, reklam bütçesinin boşa gitme riskini (Yanlış Pozitif) sadece %4 ile sınırlarken, satın alma potansiyeli yüksek olan müşterilerimi yakalama (Doğru Pozitif) konusunda yüksek bir başarı gösterdi.
