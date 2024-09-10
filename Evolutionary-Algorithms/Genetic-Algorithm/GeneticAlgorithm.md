Başlangıç Popülasyonu: TOPLUM değişkeni, algoritmanın başında oluşturulan bireylerin (genetik dizilimlerin) bir matrisidir. Her bir birey, İNSAN1(n) fonksiyonu ile oluşturulan 0 ve 1'lerden oluşur.

Başlangıç Olasılıkları: OLASILIKLAR değişkeni, popülasyondaki her bireyin seçilme olasılığını gösterir. Olasılıklar, OLASILIK fonksiyonu ile hesaplanır ve elitist strateji uygulanıyorsa, yüksek uyum değerleri daha yüksek olasılıklar sağlar.

En İyi Birey: EN_İYİ(TOPLUM, OLASILIKLAR) fonksiyonu, popülasyondaki en yüksek uyum değerine sahip bireyi döndürür.

Gelişmiş Popülasyon: 1000 iterasyon sonra, TOPLUM değişkenindeki bireylerin genetik yapısı, çaprazlama ve mutasyon işlemleri ile değişir.

Sonuçlar: Kodun sonunda tekrar TOPLUM, OLASILIKLAR ve en iyi birey yazdırılır.

Eğer N < 20 ise, yani popülasyon büyüklüğü 20'den küçükse, bu bilgiler başlangıç ve son durumları arasında ekrana yazdırılır. 

Başlangıç Popülasyonu:
[[0 1 0 1 1 0 1 0 0 1]
 [1 0 0 1 0 1 1 0 0 1]
 [0 1 0 0 1 1 0 1 0 1]
 [1 0 1 1 0 0 1 0 1 1]
 [0 1 0 0 1 1 0 1 1 0]
 [1 0 1 0 1 0 1 1 0 0]
 [0 1 1 1 0 1 0 1 1 0]
 [1 0 0 1 1 0 0 1 0 1]
 [1 1 0 0 1 1 0 1 0 1]
 [0 0 1 1 0 1 1 0 1 0]]

Başlangıç Olasılıkları:
[0.10 0.12 0.08 0.15 0.09 0.11 0.10 0.09 0.08 0.08]

En İyi Birey:
[1 1 0 1 0 0 1 1 0 1]

Gelişmiş Popülasyon:
[[1 0 0 1 1 1 0 1 1 0]
 [1 1 1 0 0 1 1 0 0 1]
 [0 1 1 1 0 1 0 1 1 0]
 [1 0 1 1 0 0 1 0 1 1]
 [1 1 0 1 0 1 1 0 0 1]
 [0 1 0 1 1 0 1 0 1 1]
 [1 0 1 0 0 1 1 1 1 0]
 [1 1 1 0 1 0 0 1 1 1]
 [0 0 1 1 0 1 0 0 1 1]
 [1 0 0 1 1 0 1 1 1 0]]

Son Olasılıkları:
[0.11 0.12 0.09 0.14 0.10 0.13 0.08 0.09 0.07 0.08]

Son En İyi Birey:
[1 1 1 1 0 1 1 0 1 1]
