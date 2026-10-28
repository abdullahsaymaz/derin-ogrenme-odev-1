import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import time
import os
from PIL import Image, ImageOps

# cihaz seç
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Cihaz:", device)

# veriyi indir
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_all = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 50k train 10k validation ayırma
train_data, val_data = random_split(train_all, [50000, 10000])

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# MLP classı 2 ara katman
class OrnekMLP(nn.Module):
    def __init__(self):
        super(OrnekMLP, self).__init__()
        self.katman1 = nn.Linear(28 * 28, 128)
        self.katman2 = nn.Linear(128, 64)
        self.cikis = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.katman1(x))
        x = self.relu(self.katman2(x))
        x = self.cikis(x)
        return x


model = OrnekMLP().to(device)

olc = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10

train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []

baslangic = time.time()

print("Eğitim başladı")

for epoch in range(epochs):
    model.train()
    egitim_kayip = 0.0
    dogru = 0
    toplam = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = olc(outputs, labels)
        loss.backward()
        optimizer.step()
        
        egitim_kayip += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        toplam += labels.size(0)
        dogru += (predicted == labels).sum().item()
        
    tloss = egitim_kayip / len(train_loader)
    tacc = 100 * dogru / toplam
    train_loss_list.append(tloss)
    train_acc_list.append(tacc)
    
    # doğrulama
    model.eval()
    val_kayip = 0.0
    val_dogru = 0
    val_toplam = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = olc(outputs, labels)
            val_kayip += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            val_toplam += labels.size(0)
            val_dogru += (predicted == labels).sum().item()
            
    vloss = val_kayip / len(val_loader)
    vacc = 100 * val_dogru / val_toplam
    val_loss_list.append(vloss)
    val_acc_list.append(vacc)
    
    print(f"Epoch {epoch+1} bitti -> Train Loss: {tloss:.4f}, Train Acc: {tacc:.2f}% | Val Loss: {vloss:.4f}, Val Acc: {vacc:.2f}%")

bitis = time.time()
print("Egitim suresi:", round(bitis - baslangic, 2), "saniye")

# Test verisi sonuç
model.eval()
test_dogru = 0
test_toplam = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_toplam += labels.size(0)
        test_dogru += (predicted == labels).sum().item()

print("Test Accuracy: %", 100 * test_dogru / test_toplam)

# Grafik çizimi
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss_list, label='Train Loss')
plt.plot(val_loss_list, label='Val Loss')
plt.title('Loss Grafigi')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc_list, label='Train Acc')
plt.plot(val_acc_list, label='Val Acc')
plt.title('Accuracy Grafigi')
plt.legend()

plt.savefig('grafikler.png')
print("grafikler.png kaydedildi")

# El yazım öğrenci numaramla test etme
klasor = "./okul_numaram"
if os.path.exists(klasor):
    dosyalar = sorted(os.listdir(klasor))
    if len(dosyalar) > 0:
        plt.figure(figsize=(12, 3))
        for i, dosya in enumerate(dosyalar):
            yol = os.path.join(klasor, dosya)
            img = Image.open(yol).convert('L') # griye çevirir
            
            # beyaz kağıda siyah yazdıysak ters çeviriyoruz mnist gibi olsun diye
            img = ImageOps.invert(img)
            img = img.resize((28, 28))
            
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                cikti = model(img_tensor)
                _, tahmin = torch.max(cikti, 1)
                
            plt.subplot(1, len(dosyalar), i+1)
            plt.imshow(img, cmap='gray')
            plt.title(f"Tahmin: {tahmin.item()}")
            plt.axis('off')
            
        plt.savefig('test_okul_numaram_sonuc.png')
        print("test_okul_numaram_sonuc.png kaydedildi")