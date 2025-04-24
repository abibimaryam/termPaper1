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
from sklearn.metrics import accuracy_score

resnet_model = timm.create_model("resnet18_cifar100", pretrained=True)
print(resnet_model)

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
    def __init__(self, in_channels, out_channels, stride=1, num_heads=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels 
        self.num_heads = num_heads

        assert in_channels % num_heads == 0, "in_channels должно делиться на num_heads"
        self.head_dim = in_channels // num_heads

        # Проекции для MHA
        self.W_Q = nn.Parameter(torch.randn(num_heads, in_channels, self.head_dim))
        self.W_K = nn.Parameter(torch.randn(num_heads, in_channels, self.head_dim))
        self.W_V = nn.Parameter(torch.empty(num_heads, in_channels, self.head_dim))
        self.W_O = nn.Parameter(torch.randn(num_heads * self.head_dim, in_channels))

        # FFN
        self.ffn1 = nn.Linear(in_channels, out_channels)
        self.ffn2 = nn.Linear(out_channels, out_channels)

        # Нормализация и shortcut
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
            )

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.W_Q, -0.1, 0.1)
        nn.init.uniform_(self.W_K, -0.1, 0.1)
        nn.init.uniform_(self.W_V, -0.1, 0.1)
        nn.init.xavier_uniform_(self.W_O)
        nn.init.zeros_(self.ffn1.bias)
        nn.init.zeros_(self.ffn2.bias)

    def load_conv_weights(self, conv1, conv2):
        with torch.no_grad():
            conv1_weights = conv1.weight  # [C_out, C_in, 3, 3]
            head_locations = [(i, j) for i in range(3) for j in range(3)]
            selected_heads = head_locations[:self.num_heads]
            
            for head_idx, (i, j) in enumerate(selected_heads):
                patch = conv1_weights[:, :, i, j]  # [C_out, C_in]
                usable_dim = min(self.head_dim, patch.shape[1])
                self.W_V[head_idx, :, :usable_dim] = patch[:, :usable_dim]

            # FFN
            conv2_weights = conv2.weight  # [C_out, C_in, 3, 3]
            ffn1_weights = conv2_weights.mean(dim=[2, 3]) # [C_out, C_in]
            if ffn1_weights.shape[1] != self.ffn1.weight.shape[1] or ffn1_weights.shape[0] != self.ffn1.weight.shape[0]:
                ffn1_weights = F.adaptive_avg_pool2d(ffn1_weights.unsqueeze(0), self.ffn1.weight.shape[:2]).squeeze(0)
            self.ffn1.weight.data = ffn1_weights

    def forward(self, x):
        identity = x
        B, C, H, W = x.shape
        
        # Attention part
        x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  #[B, H, W, C]
        x_attn = torch.zeros_like(x_norm)
        
        # Reshape for attention
        x_reshaped = x_norm.permute(0, 2, 3, 1).reshape(B * H * W, C)  #[B*H*W, C]
        
        # Multi-head attention
        Q = torch.matmul(x_reshaped, self.W_Q.reshape(-1, self.head_dim * self.num_heads))  #[B*H*W, C] * [C , self.head_dim * self.num_heads] = [B*H*W, num_heads * head_dim]
        K = torch.matmul(x_reshaped, self.W_K.reshape(-1, self.head_dim * self.num_heads))
        V = torch.matmul(x_reshaped, self.W_V.reshape(-1, self.head_dim * self.num_heads))
        
        Q = Q.view(B, H, W, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)  # [B, num_heads, H, W, head_dim]
        K = K.view(B, H, W, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        V = V.view(B, H, W, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        
        # Attention scores
        attn_scores = torch.einsum('bnhwd,bmhwd->bnmhw', Q, K) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=2)
        x_attn = torch.einsum('bnmhw,bmhwd->bnhwd', attn_weights, V)
        x_attn = x_attn.permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)  #[B, H, W, num_heads * head_dim]
        x_attn = torch.matmul(x_attn, self.W_O).permute(0, 3, 1, 2)  #[B, H, W, out_channels]
        
        # Residual connection
        x = x_norm + x_attn
        x = F.relu(x)
        
        # FFN part
        x = x.permute(0, 2, 3, 1).reshape(-1, 64)   #[B, H, W, out_channels]
        
        
        # Применяем ffn1 и ffn2
        x = self.ffn1(x)  
        print("После ffn1:", x.shape)
        x = F.relu(x)
        x = self.ffn2(x) 
        
        x = x.reshape(B, H, W, self.out_channels).permute(0, 3, 1, 2) #B, H, W, out_channels]
        
        # Final normalization and residual
        x = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  #[B, out_channels, H, W]
        x += self.shortcut(identity)
        x = F.relu(x)
        
        return x




    # def forward(self, x):
    #     identity = x
    #     B, C, H, W = x.shape
        
    #     # Attention part
    #     x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) #[B,H,W,C]
    #     x_attn = torch.zeros_like(x_norm)
        
    #     # Reshape for attention
    #     x_reshaped = x_norm.permute(0, 2, 3, 1).reshape(B * H * W, C) #[B*H*W, C]
        
    #     # Multi-head attention
    #     Q = torch.matmul(x_reshaped, self.W_Q.reshape(-1, self.head_dim * self.num_heads)) #[B*H*W, C] * [C , self.head_dim * self.num_heads] = [B * H * W, num_heads * head_dim]
    #     K = torch.matmul(x_reshaped, self.W_K.reshape(-1, self.head_dim * self.num_heads))
    #     V = torch.matmul(x_reshaped, self.W_V.reshape(-1, self.head_dim * self.num_heads))
        
    #     Q = Q.view(B, H, W, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)  # [B, num_heads H, W, head_dim]
    #     K = K.view(B, H, W, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
    #     V = V.view(B, H, W, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        
    #     # Attention scores (simplified)
    #     attn_scores = torch.einsum('bnhwd,bmhwd->bnmhw', Q, K) / (self.head_dim ** 0.5)
    #     attn_weights = F.softmax(attn_scores, dim=2)
    #     x_attn = torch.einsum('bnmhw,bmhwd->bnhwd', attn_weights, V)
    #     x_attn = x_attn.permute(0, 2, 3, 1, 4).reshape(B, H, W, -1) #[B, H, W, num_heads * head_dim]
    #     x_attn = torch.matmul(x_attn, self.W_O).permute(0, 3, 1, 2) #  [B, H, W, num_heads * head_dim] *  [num_heads * head_dim, out_channels] = [B, H, W, out_channels] = [B,out_channels,H,W]
        
    #     # Residual connection
    #     x = x_norm + x_attn
    #     x = F.relu(x)
        
    #     # FFN part
    #     x = x.permute(0, 2, 3, 1).reshape(-1, self.out_channels)  #[B, H, W, out_channels]
    #     print("ffn1 weights shape:", self.ffn1.weight.shape)
    #     print("ffn2 weights shape:", self.ffn2.weight.shape)
    #     x = self.ffn2(F.relu(self.ffn1(x)))
    #     print("ffn1 weights shape:", self.ffn1.weight.shape)
    #     print("ffn2 weights shape:", self.ffn2.weight.shape)
    #     x = x.reshape(B, H, W, self.out_channels).permute(0, 3, 1, 2)
        
    #     # Final normalization and residual
    #     x = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) #[B, out_channels, H, W].
    #     x += self.shortcut(identity)
    #     x = F.relu(x)
        
    #     return x


layer=TransformerConvBlock(in_channels=64,out_channels=128,stride=2)
layer.load_conv_weights(resnet_model.layer1[0].conv1, resnet_model.layer1[0].conv2)
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
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        elif self.shortcut is not None:
            identity = self.shortcut(x)

        out += identity
        out = self.relu(out)

        return out
    

basic_block=BasicBlockResnet(64, 128,stride=2)
print(basic_block)



x = torch.randn(1, 64, 32, 32)
x = (x - x.min()) / (x.max() - x.min()) 
x = x * 254 + 1  
print(x)

# Убедимся, что блоки на одном устройстве (если есть CUDA — будет 'cuda')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)
layer = layer.to(device)
basic_block = basic_block.to(device)

# Прогон через BasicBlock
with torch.no_grad():
    out_basic = basic_block(x)
    print("BasicBlock output shape:", out_basic.shape)

# Прогон через TransformerConvBlock
with torch.no_grad():
    out_transformer = layer(x)
    print("TransformerConvBlock output shape:", out_transformer.shape)



# #MAE
# diff = torch.abs(out_basic - out_transformer)
# mean_diff = diff.mean()

# print("Средняя разница по модулю:", mean_diff.item())

# #MSE
# squared_diff = torch.pow(out_basic - out_transformer, 2)
# mean_squared_diff = squared_diff.mean().item()

# print(f"Mean Squared Error: {mean_squared_diff:.6f}")

# Для out_basic
mean_basic = torch.abs(out_basic).mean()
print("Среднее значение по модулю для out_basic:", mean_basic.item())

# Для out_transformer
mean_transformer = torch.abs(out_transformer).mean()
print("Среднее значение по модулю для out_transformer:", mean_transformer.item())

# каждый блок свой слой
# сделать трансформер столько же слоев сколько в резнет 18 (8)
# каждый слой должен соответсвовать своему блоку
# инициализировать каждый слой весами соответсвующего блока
# посчитать accuracy для трансформера и резнет
# взять выход у резнета как есть и сравнить точности 
# сравнить количество параметров

class TransformerModel(nn.Module):
    def __init__(self, resnet):
        super(TransformerModel, self).__init__()
        
        # начальный conv, bn, relu, maxpool
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.act1,
            resnet.maxpool
        )

        self.layer1 = self.make_layer(resnet.layer1,stride0=1,stride1=1)
        self.layer2 = self.make_layer(resnet.layer2,stride0=2,stride1=1)
        self.layer3 = self.make_layer(resnet.layer3,stride0=2,stride1=1)
        self.layer4 = self.make_layer(resnet.layer4,stride0=1,stride1=1)

        # Pooling + FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 100)  # CIFAR-100

    def make_layer(self, resnet_layer,stride0,stride1):
        layers = []
        for idx, block in enumerate(resnet_layer):
            in_c = block.conv1.in_channels
            out_c = block.conv2.out_channels
            if idx==0:
                trans_block = TransformerConvBlock(in_c, out_c, num_heads=8,stride=stride0)
                trans_block.load_conv_weights(block.conv1, block.conv2)
            else:
                trans_block = TransformerConvBlock(in_c, out_c, num_heads=8,stride=stride1)
                trans_block.load_conv_weights(block.conv1, block.conv2)
            layers.append(trans_block)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


    
model_transformer = TransformerModel(resnet_model).to(device)
print(model_transformer)
# model_resnet=resnet_model.to(device)

# # Функция для оценки точности
# def evaluate(model, dataloader):
#     model.eval()
#     all_preds = []
#     all_labels = []
    
#     with torch.no_grad():
#         for inputs, labels in dataloader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
    
#     accuracy = accuracy_score(all_labels, all_preds)
#     return accuracy


# # Оценка модели ResNet
# resnet_accuracy = evaluate(model_resnet, test_loader)
# print(f"ResNet Accuracy: {resnet_accuracy * 100:.2f}%")

# # Оценка модели Transformer
# transformer_accuracy = evaluate(model_transformer, test_loader)
# print(f"Transformer Accuracy: {transformer_accuracy * 100:.2f}%")

# y = torch.randn(1, 3, 32, 32)
# y = (y - y.min()) / (y.max() - y.min()) 
# y = y * 254 + 1  
# y = y.to(device)
# print(y)
# with torch.no_grad():
#     out_transformer = model_transformer(y)
#     print("TransformerConvBlock output shape:", out_transformer.shape)