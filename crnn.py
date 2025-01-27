"""Импорт библиотек"""
import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import Levenshtein


"""Определение алфавита и создание словаря с индексами символов"""
alphabet = 'әғқңөұүіһӘҒҚҢӨҰҮІҺабвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ0123456789,.!?- '
char_to_idx = {char: idx + 1 for idx, char in enumerate(alphabet)}  # 0 зарезервирован для blank, char остается char, idx = idx + 1
idx_to_char = {idx + 1: char for idx, char in enumerate(alphabet)}  # наоборот


"""Создание Датасета"""
class HandwrittenDataset(Dataset):
    def __init__(self, image_folder, tsv_file, transform=None, char_to_idx=None):
        self.image_folder = image_folder
        self.data = pd.read_csv(tsv_file, sep='\t')  # в tsv файле файл - содержимое разделено табуляцией
        self.transform = transform
        self.char_to_idx = char_to_idx

    def __len__(self):
        return len(self.data)

    def encode_label(self, label): # перебирает символы в индексы
        try:
            return [self.char_to_idx[char] 
                    for char in label 
                    if char in self.char_to_idx]
        except TypeError:
            return []

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.data.iloc[idx, 0]) # берет название файла с tsv, где имя файла на 0 колонне
        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE) # переводит изоюражение в градции серого
        label = self.data.iloc[idx, 1] # из tsv файла берет что изображенно на фото
        label_encoded = self.encode_label(label) # переводит текст в индексы
        
        if self.transform:
            image = self.transform(image) # переводит изображение в тензор
        
        label_encoded = torch.tensor(label_encoded, dtype=torch.long) # переводит текст изображения в тензор
        
        return image, label_encoded
    
    
"""Модель CRNN"""
class CRNN(nn.Module):
    def __init__(self, img_h, num_channels, num_classes): # высота изображения, количество каналов (в градациях серого = 1, в ргб 3), количество классов символов (классификация)
        super(CRNN, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Уменьшаем высоту и ширину в 2 раза
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Уменьшаем высоту и ширину в 2 раза
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # Уменьшаем высоту в 2 раза
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # Уменьшаем высоту в 2 раза
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1))  # Уменьшаем высоту до 1
        ) # преобразование изображения
        
        self.lstm1 = nn.LSTM(256, 256, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.cnn(x)
        
        b, c, h, w = x.size()
        assert h == 1, f"высота должна быть 1, но вышло {h}"  # Изменим вывод ошибки для большей информативности
        x = x.squeeze(2)  # Убираем размер высоты
        x = x.permute(2, 0, 1)  # Переставляем для подачи в LSTM: (W, batch, C)
        
        # Обработка с помощью LSTM
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.fc(x)
        return x



"""Функции декодирования метриков и паддинга текста"""
def decode_output(outputs, idx_to_char):
    pred_texts = []
    for output in outputs:
        pred = []
        previous = None
        for t in output:
            _, max_idx = torch.max(t, dim=0)
            max_idx = max_idx.item()
            if max_idx != 0 and max_idx != previous:
                pred.append(idx_to_char.get(max_idx, ''))
            previous = max_idx
        pred_text = ''.join(pred)
        pred_texts.append(pred_text)
    return pred_texts

def decode_labels(labels, idx_to_char):
    true_texts = []
    for label in labels:
        true_text = ''.join([idx_to_char.get(idx.item(), '') for idx in label])
        true_texts.append(true_text)
    return true_texts

def compute_metrics(pred_texts, true_texts):
    total_edit_distance = 0
    correct_chars = 0
    total_chars = 0

    for pred, true in zip(pred_texts, true_texts):
        total_edit_distance += Levenshtein.distance(pred, true)
        correct_chars += sum(p == t for p, t in zip(pred, true))
        total_chars += len(true)
    
    accuracy = correct_chars / total_chars if total_chars > 0 else 0
    return accuracy, total_edit_distance

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    images, labels = zip(*batch)
    
    # Паддинг для меток (labels)
    labels_padded = pad_sequence([label.clone().detach().long() for label in labels], 
                                 batch_first=True, padding_value=0)
    
    # Преобразуем список изображений в тензор
    images = torch.stack([image.clone().detach().float() for image in images], dim=0)
    
    return images, labels_padded


"""Функция обучения"""
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_edit_distance = 0
    total_correct_chars = 0
    total_chars = 0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))

    for batch_idx, (images, labels_padded) in pbar:
        images = images.to(device)
        labels_padded = labels_padded.to(device)

        optimizer.zero_grad()

        # Предсказания модели
        output = model(images)

        # Рассчитываем loss, используя CTC Loss
        input_lengths = torch.full((output.size(1),), output.size(0), dtype=torch.long).to(device)
        target_lengths = torch.tensor([len(label[label != 0]) for label in labels_padded], dtype=torch.long).to(device)

        loss = criterion(output.log_softmax(2), labels_padded, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Предсказание и расчет метрик
        pred_texts = decode_output(output.permute(1, 0, 2).detach().cpu(), idx_to_char)
        true_texts = decode_labels(labels_padded.detach().cpu(), idx_to_char)

        batch_accuracy, batch_edit_distance = compute_metrics(pred_texts, true_texts)
        total_correct_chars += batch_accuracy * sum(len(t) for t in true_texts)
        total_chars += sum(len(t) for t in true_texts)
        total_edit_distance += batch_edit_distance

        # Обновление прогресс-бара с метриками
        avg_loss = total_loss / (batch_idx + 1)
        avg_accuracy = total_correct_chars / total_chars if total_chars > 0 else 0
        avg_edit_distance = total_edit_distance / (batch_idx + 1)
        pbar.set_description(f'Loss: {avg_loss:.4f} Acc: {avg_accuracy:.4f} Edit Dist: {avg_edit_distance:.4f}')

    avg_loss = total_loss / len(dataloader)
    return avg_loss, total_correct_chars / total_chars, total_edit_distance / total_chars



"""Трансформация изображения в тензор"""
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


"""Инициализация/Запуск"""
# Инициализация устройства
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Инициализация модели
model = CRNN(img_h=32, num_channels=1, num_classes=len(alphabet) + 1).to(device)

# Определение функции потерь и оптимизатора
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Создание датасета и загрузчика данных
image_folder = 'C:/Users/zhnbk/kazhtr/cyrillicset/train'  # Укажите путь к папке с изображениями
tsv_file = 'C:/Users/zhnbk/kazhtr/cyrillicset/train.tsv'  # Укажите путь к .tsv файлу с метками

train_dataset = HandwrittenDataset(image_folder=image_folder, tsv_file=tsv_file, transform=transform, char_to_idx=char_to_idx)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# Обучение модели
num_epochs = 20

for epoch in range(num_epochs):
    train_loss, train_acc, train_edit_dist = train_epoch(model, train_loader, criterion, optimizer, device)
    print(f'Epoch {epoch + 1}/{num_epochs} | Training Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f} | Edit Distance: {train_edit_dist:.4f}')

torch.save(model.state_dict(), 'model-V1.pth')
print('Модель сохранена')