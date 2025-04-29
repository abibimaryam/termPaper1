from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.optim import AdamW
from tqdm import tqdm
from main import model_transformer

print(model_transformer)

device = 'cuda'
seed = 42
batch_size = 64
epochs = 50


# Загрузка тренировочного набора
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True)

labels = [label for _, label in trainset]
# Загрузка тестового набора
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True)


random_idx = np.random.randint(0, len(trainset), size=9)


# Отображение изображений
fig, axes = plt.subplots(3, 3, figsize=(16, 12))

for i, ax in enumerate(axes.ravel()):
    img, label = trainset[random_idx[i]]  # Получаем изображение и метку
    ax.imshow(img)
    ax.set_title(trainset.classes[label])  # Название класса
    ax.axis("off") 

plt.show()



print(f"Train Data: {len(trainset)}")
print(f"Test Data: {len(testset)}")


normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                                 std=[0.2673, 0.2564, 0.2761]) 

train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    normalize,
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

class CIFAR100Dataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]  # Получаем изображение и метку

        if self.transform:
            img = self.transform(img)

        return img, label

train_data = CIFAR100Dataset(trainset, transform=train_transforms)
test_data = CIFAR100Dataset(testset, transform=test_transforms)

train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True )
test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)

print(len(train_data), len(train_loader))
print(len(test_data), len(test_loader))






criterion = nn.CrossEntropyLoss()
# # optimizer
# optimizer = optim.Adam(model_transformer.parameters(), lr=lr)
# # scheduler
# scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

# Использование AdamW оптимизатора
optimizer = AdamW(model_transformer.parameters(), lr=1e-4, weight_decay=1e-5)

# Планировщик для learning rate
scheduler = StepLR(optimizer, step_size=10, gamma=0.7)

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    model_transformer.train()
    for data, label in tqdm(train_loader):
        data = data.to(device)
        label = label.to(device)

        output = model_transformer(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)
    model_transformer.eval()
    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in test_loader:
            data = data.to(device)
            label = label.to(device)

            val_output = model_transformer(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(test_loader)
            epoch_val_loss += val_loss / len(test_loader)
    
    # Открыть файл для записи (если файл не существует, он будет создан)
    with open('my-transformer-param.txt', 'a') as f:
         f.write(
            f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - test_loss : {epoch_val_loss:.4f} - test_acc: {epoch_val_accuracy:.4f}\n"
         )

    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - test_loss : {epoch_val_loss:.4f} - test_acc: {epoch_val_accuracy:.4f}\n"
    )
