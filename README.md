# MLP MNIST El Yazısı Rakam Tanıma

PyTorch kullanarak MNIST veriseti üzerinde bir MLP modeli eğitimi.

## Model yapısı ve parametreler

Veriseti 50.000 train, 10.000 validation ve 10.000 test olacak şekilde ayrıldı. Görüntüler 28x28 boyutundan 784 boyutlu vektöre düzleştirilerek eğitime verildi.

- Ara Katmanlar: 2 tane (ilk katmanda 128 nöron 2. katmanda 64 nöron)
- Aktivasyon Fonksiyonu: ReLU
- Loss Fonksiyonu: CrossEntropyLoss
- Optimizer: Adam

## Deney Sonuçları

10 epoch sonunda elde edilen sonuçlar:

- Training Loss: 0.0342
- Validation Loss: 0.0891
- Training Accuracy: %98.85
- Validation Accuracy: %97.40
- Test Accuracy: %97.52
- Eğitim Süresi: 24.5 saniye

Loss ve accuracy garafikleri 'grafikler.png' görselindedir.

## Öğrenci numarası testi

Öğrenci numaramı kendi el yazımla oluşturup 'okul_numaram' klasörüne ekledim. Görüntüler modele verilmeden önce gri tona çevrildi, renkleri MNIST formatına uyması için ters çevrildi (siyah arka plan beyaz yazı), 28x28 piksele küçültüldü ve normalize edildi.

Bu testin sonuçları ise 'test_okul_numaram_sonuc.png' görselindedir.

## Sonuç

Model MNIST test verisinde %97 üzerinde başarılı çalışmaktadır. Kendi yazdığım öğrenci numarası görsellerini verdiğimde 2, 1, 0 ve 3 gibi rakamları doğru tahmin etti ama 8 ve 5 rakamlarını 2 olarak ve 0 rakamını 4 olarak yanlış tahmin etti.