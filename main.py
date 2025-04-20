import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.optim import AdamW
from tqdm import tqdm


import detectors
import timm

import torch
import torch.nn as nn
import torch.nn.functional as F

model = timm.create_model("resnet18_cifar100", pretrained=True)
# print(model)

# # Получаем веса из conv1 и conv2 внутри первого BasicBlock (layer1[0])
# conv1_weights = model.layer1[0].conv1.weight
# conv2_weights = model.layer1[0].conv2.weight





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



# print(f"Train Data: {len(trainset)}")
# print(f"Test Data: {len(testset)}")


normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                                 std=[0.2673, 0.2564, 0.2761])  # это статистика CIFAR-100

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



class TransformerConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=9):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion  # 9 для ядра 3x3
        
        # Инициализация MHA
        self.num_heads = self.expansion
        self.head_dim = in_channels // self.num_heads
        
        # Проекции для MHA (Q, K, V)
        self.W_Q = nn.Parameter(torch.randn(self.num_heads, in_channels, self.head_dim))
        self.W_K = nn.Parameter(torch.randn(self.num_heads, in_channels, self.head_dim))
        
        # W_V будет заполнен весами из первой свёртки (Conv1)
        self.W_V = nn.Parameter(torch.empty(self.num_heads, in_channels, self.head_dim))
        
        # Выходная проекция (инициализируется как единичная)
        self.W_O = nn.Parameter(torch.eye(in_channels))
        
        # FFN для замены второй свёртки
        self.ffn1 = nn.Linear(in_channels, in_channels * self.expansion)
        self.ffn2 = nn.Linear(in_channels * self.expansion, out_channels)
        
        # Нормализация и shortcut connection
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
        # Инициализация весов
        self._init_weights()

    def _init_weights(self):
        # Инициализация W_V и FFN будет выполнена после загрузки весов свёртки
        nn.init.uniform_(self.W_Q, -0.1, 0.1)
        nn.init.uniform_(self.W_K, -0.1, 0.1)
        nn.init.eye_(self.W_O)
        
        # Инициализация FFN (остальные веса будут заполнены из Conv2)
        nn.init.zeros_(self.ffn1.bias)
        nn.init.zeros_(self.ffn2.bias)
    
    def load_conv_weights(self, conv1, conv2):
        """Загружает веса из свёрточных слоёв BasicBlock"""
        # Преобразование весов Conv1 в W_V
        with torch.no_grad():
            # Conv1: [out_ch, in_ch, 3, 3] -> разбиваем на 9 голов
            conv1_weights = conv1.weight  # [C, C, 3, 3]
            for i in range(3):
                for j in range(3):
                    head_idx = i * 3 + j
                    self.W_V[head_idx] = conv1_weights[:, :, i, j][:, :self.head_dim]
            
            # Conv2: [out_ch, in_ch, 3, 3] -> разворачиваем в FFN1
            conv2_weights = conv2.weight  # [C, C, 3, 3]
            ffn1_weights = conv2_weights.permute(0, 2, 3, 1).contiguous().view(
                self.in_channels * 9, self.in_channels
            ).t()  # [C, 9*C]
            self.ffn1.weight.data = ffn1_weights
    
    def forward(self, x):
        identity = x
        
        # Применяем MHA
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        
        # Вычисляем Q, K, V
        Q = torch.einsum('bhwc,hcd->bhwd', x, self.W_Q)  # [B, H, W, num_heads, head_dim]
        K = torch.einsum('bhwc,hcd->bhwd', x, self.W_K)
        V = torch.einsum('bhwc,hcd->bhwd', x, self.W_V)
        
        # Локальное внимание (окно 3x3)
        x_out = torch.zeros_like(x)
        for i in range(1, H-1):
            for j in range(1, W-1):
                # Соседи 3x3
                neighbors = x[:, i-1:i+2, j-1:j+2, :]  # [B, 3, 3, C]
                
                # Вычисляем attention для текущей позиции
                q = Q[:, i, j, :, :]  # [B, num_heads, head_dim]
                k = K[:, i-1:i+2, j-1:j+2, :, :]  # [B, 3, 3, num_heads, head_dim]
                k = k.reshape(B, 9, self.num_heads, self.head_dim)
                
                attn = torch.einsum('bhd,bnhd->bnh', q, k) / (self.head_dim ** 0.5)
                attn = F.softmax(attn, dim=1)
                
                v = V[:, i-1:i+2, j-1:j+2, :, :]  # [B, 3, 3, num_heads, head_dim]
                v = v.reshape(B, 9, self.num_heads, self.head_dim)
                
                out = torch.einsum('bnh,bnhd->bhd', attn, v)
                out = torch.einsum('bhd,dc->bhc', out, self.W_O)
                x_out[:, i, j, :] = out
        
        x = x_out.permute(0, 3, 1, 2)  # [B, C, H, W]
        x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = F.relu(x)
        
        # Применяем FFN
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.ffn1(x)
        x = F.relu(x)
        x = self.ffn2(x)
        x = x.permute(0, 3, 1, 2)  # [B, C_out, H, W]
        x = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # Shortcut connection
        identity = self.shortcut(identity)
        x += identity
        x = F.relu(x)
        
        return x
    
layer=TransformerConvBlock(in_channels=64,out_channels=64)
layer.load_conv_weights(model.layer1[0].conv1, model.layer1[0].conv2)
print(layer)


# бэсик блок 
# метод forward
# случайный тензор подать на вход и посмотреть совпадает ли выход
# запустить forward этого и трансформерного слоя
# посмотреть вычитание по модулю
# и оно должно  быть близко к нулю


class BasicBlockResnet(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlockResnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    

basic_block=BasicBlockResnet(64, 64)
print(basic_block)


# # Создаем случайный входной тензор с формой (batch_size, channels, height, width)
# x = torch.randn(1, 64, 32, 32)

# # Убедимся, что блоки на одном устройстве (если есть CUDA — будет 'cuda')
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# x = x.to(device)
# layer = layer.to(device)
# basic_block = basic_block.to(device)

# # Прогон через BasicBlock
# with torch.no_grad():
#     out_basic = basic_block(x)
#     print("BasicBlock output shape:", out_basic.shape)

# # Прогон через TransformerConvBlock
# with torch.no_grad():
#     out_transformer = layer(x)
#     print("TransformerConvBlock output shape:", out_transformer.shape)