import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import sqlite3
import json
import os
import numpy as np
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from typing import List, Dict, Tuple
from datetime import datetime
import random
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class ProductDataset(Dataset):
    """Датасет для обучения на товарах"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Загружаем изображение
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Применяем трансформации
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label

class EarlyStopping:
    """Early Stopping для предотвращения переобучения"""
    def __init__(self, patience=5, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            print(f"⏰ Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                print("🛑 Early stopping triggered!")
        else:
            self.best_score = val_score
            self.counter = 0
        
        return self.early_stop

class PytorchMerchML:
    def __init__(self, db_path="merchai_v2.db", model_path="pytorch_models/"):
        self.db_path = db_path
        self.model_path = model_path
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.confidence_threshold = 40.0  # ← ПОРОГ УВЕРЕННОСТИ 40%
        
        print(f"🔥 Используем устройство: {self.device}")
        print(f"🎯 Порог уверенности: {self.confidence_threshold}%")
        
        # Создаем папку для моделей
        os.makedirs(model_path, exist_ok=True)
        
        # Трансформации для обучения (с аугментацией)
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Трансформации для валидации (стандартные)
        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Трансформации для предсказания
        self.predict_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.model = None
        self.class_names = []
        self.product_to_label = {}
        self.training_log = []
    
    def get_training_data(self) -> Tuple[List[str], List[int], List[str], Dict]:
        """Получаем данные для обучения из БД"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Получаем все изображения с товарами
        cursor.execute('''
            SELECT ti.filename, p.name, p.id
            FROM training_images ti
            JOIN products p ON ti.product_id = p.id
        ''')
        
        data = cursor.fetchall()
        conn.close()
        
        if not data:
            return [], [], [], {}
        
        # СТАБИЛЬНЫЙ MAPPING КЛАССОВ
        unique_products = sorted(set([row[1] for row in data]))
        product_to_label = {name: idx for idx, name in enumerate(unique_products)}
        
        # Подготавливаем пути к файлам и метки
        image_paths = []
        labels = []
        product_names = []
        
        for filename, product_name, product_id in data:
            file_path = os.path.join("uploads/train", filename)
            
            if os.path.exists(file_path):
                image_paths.append(file_path)
                labels.append(product_to_label[product_name])
                product_names.append(product_name)
        
        self.class_names = unique_products
        self.product_to_label = product_to_label
        
        print(f"🎯 Найдено классов товаров: {len(unique_products)}")
        print(f"📊 Классы (отсортированы): {unique_products}")
        print(f"🖼️  Всего изображений: {len(image_paths)}")
        
        return image_paths, labels, unique_products, product_to_label
    
    def create_model(self, num_classes: int):
        """Создаем модель ResNet18 - ПРОВЕРЕННАЯ АРХИТЕКТУРА!"""
        print(f"🧠 Создаем ResNet18 модель для {num_classes} классов")
        
        # ВОЗВРАЩАЕМСЯ К RESNET18 - ОН ПОКАЗАЛ 91%!
        try:
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
            print("✅ ResNet18 предобученные веса загружены")
        except Exception:
            print("⚠️ Не удалось скачать веса ResNet18, учу с нуля")
            model = resnet18(weights=None)
        
        # Замораживаем ВСЕ слои сначала
        for param in model.parameters():
            param.requires_grad = False
        
        # Заменяем последний слой под наши классы
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        return model.to(self.device)
    
    def unfreeze_layers(self, model, stage: str):
        """Размораживаем слои поэтапно для ResNet18"""
        if stage == "head":
            # Этап 1: Только голова (fc)
            for param in model.fc.parameters():
                param.requires_grad = True
            print("🔓 Разморожена голова ResNet18 (fc)")
            
        elif stage == "layer4":
            # Этап 2: Последний блок + голова
            for param in model.fc.parameters():
                param.requires_grad = True
            for param in model.layer4.parameters():
                param.requires_grad = True
            # BatchNorm слои тоже trainable
            for module in model.layer4.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.track_running_stats = True
                    for param in module.parameters():
                        param.requires_grad = True
            print("🔓 Разморожены layer4 + fc (+ BatchNorm)")
            
        elif stage == "deep":
            # Этап 3: layer3 + layer4 + голова
            for param in model.fc.parameters():
                param.requires_grad = True
            for param in model.layer4.parameters():
                param.requires_grad = True
            for param in model.layer3.parameters():
                param.requires_grad = True
                
            # BatchNorm для layer3 и layer4
            for module in model.layer4.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.track_running_stats = True
                    for param in module.parameters():
                        param.requires_grad = True
            for module in model.layer3.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.track_running_stats = True
                    for param in module.parameters():
                        param.requires_grad = True
            print("🔓 Разморожены layer3 + layer4 + fc (глубокий fine-tuning + BatchNorm)")
    
    def evaluate_model(self, model, val_loader, criterion):
        """Оценка модели на валидации"""
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        val_accuracy = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        return val_accuracy, avg_val_loss
    
    def save_training_log(self, timestamp: str):
        """Сохраняем логи обучения в CSV/JSON"""
        if not self.training_log:
            return
            
        # Сохраняем в CSV для легкого анализа
        df = pd.DataFrame(self.training_log)
        csv_file = os.path.join(self.model_path, f"training_log_{timestamp}.csv")
        df.to_csv(csv_file, index=False)
        
        # Сохраняем в JSON для программной обработки
        json_file = os.path.join(self.model_path, f"training_log_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(self.training_log, f, indent=2)
            
        print(f"📊 Логи обучения сохранены: {csv_file}, {json_file}")
        return csv_file, df
    
    def plot_training_curves(self, df: pd.DataFrame, timestamp: str):
        """Автоматические графики обучения"""
        try:
            if len(df) < 3:
                print("⚠️ Мало данных для графика (< 3 эпох)")
            
            # Настройка стиля matplotlib
            plt.style.use('default')
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Обучение модели ResNet18 + порог 40% - {timestamp}', fontsize=16, fontweight='bold')
            
            # Цвета для этапов
            stage_colors = {'head': '#FF6B6B', 'layer4': '#4ECDC4', 'deep': '#45B7D1'}
            
            # График 1: Train Loss
            for stage in df['stage'].unique():
                stage_data = df[df['stage'] == stage]
                ax1.plot(stage_data['epoch'], stage_data['train_loss'], 
                        color=stage_colors.get(stage, '#666'), 
                        marker='o', linewidth=2, markersize=4,
                        label=f'{stage} stage')
            ax1.set_title('Train Loss', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # График 2: Validation Loss  
            for stage in df['stage'].unique():
                stage_data = df[df['stage'] == stage]
                ax2.plot(stage_data['epoch'], stage_data['val_loss'], 
                        color=stage_colors.get(stage, '#666'), 
                        marker='s', linewidth=2, markersize=4,
                        label=f'{stage} stage')
            ax2.set_title('Validation Loss', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # График 3: Train Accuracy
            for stage in df['stage'].unique():
                stage_data = df[df['stage'] == stage]
                ax3.plot(stage_data['epoch'], stage_data['train_accuracy'], 
                        color=stage_colors.get(stage, '#666'), 
                        marker='o', linewidth=2, markersize=4,
                        label=f'{stage} stage')
            ax3.set_title('Train Accuracy', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Accuracy (%)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # График 4: Validation Accuracy
            for stage in df['stage'].unique():
                stage_data = df[df['stage'] == stage]
                ax4.plot(stage_data['epoch'], stage_data['val_accuracy'], 
                        color=stage_colors.get(stage, '#666'), 
                        marker='s', linewidth=2, markersize=4,
                        label=f'{stage} stage')
            ax4.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Accuracy (%)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Добавляем вертикальные линии для смены этапов
            stage_changes = df.groupby('stage')['epoch'].min().values[1:]
            for ax in [ax1, ax2, ax3, ax4]:
                for change_epoch in stage_changes:
                    ax.axvline(x=change_epoch, color='red', linestyle='--', alpha=0.7, linewidth=1)
            
            plt.tight_layout()
            
            # Сохраняем график
            plot_file = os.path.join(self.model_path, f"training_curves_{timestamp}.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📈 Графики обучения сохранены: {plot_file}")
            return plot_file
            
        except Exception as e:
            print(f"⚠️ Не удалось создать графики: {e}")
            return None
    
    def train_model(self, epochs=15, batch_size=8, learning_rate=0.001, early_stop_patience=5) -> Dict:
        """Обучаем ResNet18 модель - ПРОВЕРЕННАЯ АРХИТЕКТУРА 91%"""
        print("🚀 Начинаем обучение ResNet18 + порог уверенности 40%...")
        
        # Очищаем лог тренировки
        self.training_log = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Получаем данные
        image_paths, labels, class_names, product_to_label = self.get_training_data()
        
        if len(image_paths) == 0:
            return {
                "status": "error",
                "message": "Нет данных для обучения"
            }
        
        if len(set(labels)) < 2:
            return {
                "status": "error", 
                "message": f"Нужно минимум 2 класса товаров, найдено: {len(set(labels))}"
            }
        
        num_classes = len(class_names)
        
        # Создаем модель ResNet18
        self.model = self.create_model(num_classes)
        
        # TRAIN/VAL SPLIT
        full_dataset = ProductDataset(image_paths, labels, self.train_transform)
        
        # Делим на train/val (80/20)
        val_size = int(0.2 * len(full_dataset))
        train_size = len(full_dataset) - val_size
        
        # Фиксируем seed для воспроизводимости
        torch.manual_seed(42)
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        # Для валидации используем трансформации без аугментации
        val_indices = val_dataset.indices
        val_dataset = ProductDataset(
            [image_paths[i] for i in val_indices],
            [labels[i] for i in val_indices], 
            self.val_transform
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"📊 Train: {len(train_dataset)}, Val: {len(val_dataset)} изображений")
        
        # Настраиваем критерий потерь и Early Stopping
        criterion = nn.CrossEntropyLoss()
        early_stopping = EarlyStopping(patience=early_stop_patience, min_delta=0.01)
        
        # ТРЕХЭТАПНОЕ ОБУЧЕНИЕ С EARLY STOPPING
        best_val_acc = 0.0
        total_epochs_trained = 0
        
        # ЭТАП 1: Обучаем только голову (до 5 эпох с early stopping)
        print("\n🎯 ЭТАП 1: Обучение головы ResNet18 (frozen backbone)")
        self.unfreeze_layers(self.model, "head")
        
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        
        stage1_epochs = 0
        for epoch in range(5):
            self.model.train()
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, (images, targets) in enumerate(train_loader):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                epoch_total += targets.size(0)
                epoch_correct += (predicted == targets).sum().item()
            
            # Валидация
            val_accuracy, val_loss = self.evaluate_model(self.model, val_loader, criterion)
            train_accuracy = 100 * epoch_correct / epoch_total
            avg_loss = epoch_loss / len(train_loader)
            
            # ЛОГИРУЕМ МЕТРИКИ
            self.training_log.append({
                "epoch": epoch + 1,
                "stage": "head",
                "train_loss": avg_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
            
            print(f"✅ Эпоха {epoch+1}/5: Train Loss={avg_loss:.4f}, Train Acc={train_accuracy:.2f}%, Val Acc={val_accuracy:.2f}%")
            
            scheduler.step(val_accuracy)
            stage1_epochs = epoch + 1
            total_epochs_trained += 1
            
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
            
            # Early stopping check
            if early_stopping(val_accuracy):
                print(f"🛑 Early stopping на этапе 1 после {stage1_epochs} эпох")
                break
        
        # ЭТАП 2: layer4 + fc (до 5 эпох с early stopping)
        remaining_epochs = epochs - total_epochs_trained
        if remaining_epochs > 0 and not early_stopping.early_stop:
            stage2_epochs_max = min(5, remaining_epochs)
            print(f"\n🔥 ЭТАП 2: Fine-tuning layer4 + fc ResNet18 - до {stage2_epochs_max} эпох")
            self.unfreeze_layers(self.model, "layer4")
            
            # Сбрасываем early stopping для нового этапа
            early_stopping = EarlyStopping(patience=early_stop_patience, min_delta=0.01)
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate * 0.1)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
            
            stage2_epochs = 0
            for epoch in range(stage2_epochs_max):
                self.model.train()
                epoch_loss = 0
                epoch_correct = 0
                epoch_total = 0
                
                for batch_idx, (images, targets) in enumerate(train_loader):
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    epoch_total += targets.size(0)
                    epoch_correct += (predicted == targets).sum().item()
                
                # Валидация
                val_accuracy, val_loss = self.evaluate_model(self.model, val_loader, criterion)
                train_accuracy = 100 * epoch_correct / epoch_total
                avg_loss = epoch_loss / len(train_loader)
                
                current_epoch = total_epochs_trained + epoch + 1
                
                # ЛОГИРУЕМ МЕТРИКИ
                self.training_log.append({
                    "epoch": current_epoch,
                    "stage": "layer4",
                    "train_loss": avg_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
                
                print(f"🔥 Эпоха {current_epoch}/{epochs}: Train Loss={avg_loss:.4f}, Train Acc={train_accuracy:.2f}%, Val Acc={val_accuracy:.2f}%")
                
                scheduler.step(val_accuracy)
                stage2_epochs = epoch + 1
                total_epochs_trained += 1
                
                if val_accuracy > best_val_acc:
                    best_val_acc = val_accuracy
                
                # Early stopping check
                if early_stopping(val_accuracy):
                    print(f"🛑 Early stopping на этапе 2 после {stage2_epochs} эпох")
                    break
        
        # ЭТАП 3: Глубокий fine-tuning layer3 + layer4 + fc (оставшиеся эпохи)
        remaining_epochs = epochs - total_epochs_trained
        if remaining_epochs > 0 and not early_stopping.early_stop:
            print(f"\n💥 ЭТАП 3: Глубокий fine-tuning ResNet18 - до {remaining_epochs} эпох")
            self.unfreeze_layers(self.model, "deep")
            
            # Сбрасываем early stopping для нового этапа
            early_stopping = EarlyStopping(patience=early_stop_patience, min_delta=0.01)
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate * 0.01)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
            
            for epoch in range(remaining_epochs):
                self.model.train()
                epoch_loss = 0
                epoch_correct = 0
                epoch_total = 0
                
                for batch_idx, (images, targets) in enumerate(train_loader):
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    epoch_total += targets.size(0)
                    epoch_correct += (predicted == targets).sum().item()
                
                # Валидация
                val_accuracy, val_loss = self.evaluate_model(self.model, val_loader, criterion)
                train_accuracy = 100 * epoch_correct / epoch_total
                avg_loss = epoch_loss / len(train_loader)
                
                current_epoch = total_epochs_trained + epoch + 1
                
                # ЛОГИРУЕМ МЕТРИКИ
                self.training_log.append({
                    "epoch": current_epoch,
                    "stage": "deep",
                    "train_loss": avg_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
                
                print(f"💥 Эпоха {current_epoch}/{epochs}: Train Loss={avg_loss:.4f}, Train Acc={train_accuracy:.2f}%, Val Acc={val_accuracy:.2f}%")
                
                scheduler.step(val_accuracy)
                total_epochs_trained += 1
                
                if val_accuracy > best_val_acc:
                    best_val_acc = val_accuracy
                
                # Early stopping check
                if early_stopping(val_accuracy):
                    print(f"🛑 Early stopping на этапе 3 после {epoch + 1} эпох")
                    break
        
        # Сохраняем модель с метриками в названии
        model_filename = f"resnet18_threshold40_epoch{total_epochs_trained}_acc{best_val_acc:.1f}_{timestamp}.pth"
        model_file = os.path.join(self.model_path, model_filename)
        classes_file = os.path.join(self.model_path, "class_names.json")
        
        # Сохраняем полный чекпоинт
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_names': class_names,
            'product_to_label': product_to_label,
            'num_classes': num_classes,
            'val_accuracy': best_val_acc,
            'epochs': total_epochs_trained,
            'timestamp': timestamp,
            'training_log': self.training_log,
            'early_stopped': early_stopping.early_stop,
            'architecture': 'ResNet18',
            'confidence_threshold': self.confidence_threshold
        }, model_file)
        
        # Сохраняем имена классов отдельно
        with open(classes_file, 'w', encoding='utf-8') as f:
            json.dump({
                'class_names': class_names,
                'product_to_label': product_to_label,
                'architecture': 'ResNet18',
                'confidence_threshold': self.confidence_threshold
            }, f, ensure_ascii=False, indent=2)
        
        # КРОССПЛАТФОРМЕННАЯ КОПИЯ
        latest_link = os.path.join(self.model_path, "latest_model.pth")
        try:
            shutil.copy2(model_file, latest_link)
            print(f"📎 Копия последней модели ResNet18: {latest_link}")
        except Exception as e:
            print(f"⚠️ Не удалось создать копию latest_model.pth: {e}")
        
        # СОХРАНЯЕМ ЛОГИ ОБУЧЕНИЯ И СТРОИМ ГРАФИКИ
        csv_file, df = self.save_training_log(timestamp)
        plot_file = self.plot_training_curves(df, timestamp)
        
        result = {
            "status": "success",
            "message": f"Модель ResNet18 обучена! Лучшая Val точность: {best_val_acc:.2f}% (порог {self.confidence_threshold}%)",
            "val_accuracy": best_val_acc,
            "epochs": total_epochs_trained,
            "planned_epochs": epochs,
            "early_stopped": early_stopping.early_stop,
            "num_classes": num_classes,
            "total_samples": len(image_paths),
            "class_names": class_names,
            "model_file": model_filename,
            "training_stages": len(set([log["stage"] for log in self.training_log])),
            "plot_file": plot_file,
            "architecture": "ResNet18",
            "confidence_threshold": self.confidence_threshold
        }
        
        print(f"🎉 Обучение ResNet18 завершено! Лучшая валидационная точность: {best_val_acc:.2f}%")
        print(f"💾 Модель сохранена: {model_file}")
        print(f"🎯 Порог уверенности: {self.confidence_threshold}%")
        if early_stopping.early_stop:
            print(f"🛑 Обучение остановлено досрочно после {total_epochs_trained} эпох")
        return result
    
    def load_model(self) -> bool:
        """Загружаем последнюю обученную модель"""
        # Пытаемся загрузить через копию latest
        latest_link = os.path.join(self.model_path, "latest_model.pth")
        model_file = latest_link
        
        if not os.path.exists(model_file):
            # Если нет копии, ищем последнюю модель по дате
            model_files = [f for f in os.listdir(self.model_path) 
                          if f.startswith("resnet18") and f.endswith(".pth")]
            if not model_files:
                print("❌ Обученная модель не найдена")
                return False
            
            # Сортируем по времени изменения
            model_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.model_path, x)), reverse=True)
            model_file = os.path.join(self.model_path, model_files[0])
        
        try:
            # Загружаем чекпоинт
            checkpoint = torch.load(model_file, map_location=self.device)
            
            self.class_names = checkpoint['class_names']
            self.product_to_label = checkpoint.get('product_to_label', {})
            num_classes = checkpoint['num_classes']
            architecture = checkpoint.get('architecture', 'ResNet18')
            self.confidence_threshold = checkpoint.get('confidence_threshold', 40.0)
            
            # Создаем модель
            self.model = self.create_model(num_classes)
            
            # Размораживаем все для инференса
            self.unfreeze_layers(self.model, "deep")
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            val_acc = checkpoint.get('val_accuracy', 'N/A')
            early_stopped = checkpoint.get('early_stopped', False)
            training_stages = len(set([log["stage"] for log in checkpoint.get('training_log', [])]))
            
            print(f"✅ Модель {architecture} загружена: {len(self.class_names)} классов, Val точность: {val_acc}%")
            print(f"📊 Этапов обучения: {training_stages}, Early stopped: {early_stopped}")
            print(f"🎯 Порог уверенности: {self.confidence_threshold}%")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            return False
    
    def predict_image(self, image_path: str, top_k: int = 3) -> Dict:
        """Предсказание для изображения С ПОРОГОМ УВЕРЕННОСТИ 40%"""
        print(f"🔍 Анализируем изображение: {image_path}")
        
        # Загружаем модель если нужно
        if self.model is None:
            if not self.load_model():
                return {
                    "status": "error",
                    "message": "Модель не обучена. Запустите /train/start"
                }
        
        try:
            # Загружаем и предобрабатываем изображение
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.predict_transform(image).unsqueeze(0).to(self.device)
            
            # Предсказание
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                
                # Получаем топ-K предсказаний
                top_probs, top_indices = torch.topk(probabilities, min(top_k, len(self.class_names)))
                
                results = []
                max_confidence = float(top_probs[0].item() * 100)
                
                # 🎯 ПРИМЕНЯЕМ ПОРОГ УВЕРЕННОСТИ 40%
                if max_confidence < self.confidence_threshold:
                    results.append({
                        "product": "Неизвестный товар",
                        "confidence": max_confidence
                    })
                    print(f"⚠️ Максимальная уверенность {max_confidence:.1f}% < {self.confidence_threshold}% - товар не распознан")
                else:
                    for prob, idx in zip(top_probs, top_indices):
                        confidence = float(prob.item() * 100)
                        if confidence >= self.confidence_threshold:  # Показываем только если >= порога
                            results.append({
                                "product": self.class_names[idx.item()],
                                "confidence": confidence
                            })
            
            result = {
                "status": "success", 
                "results": results,
                "confidence_threshold": self.confidence_threshold
            }
            
            print(f"🎯 Результаты предсказания ResNet18 + порог {self.confidence_threshold}%: {results}")
            return result
            
        except Exception as e:
            print(f"❌ Ошибка предсказания: {e}")
            return {
                "status": "error",
                "message": f"Ошибка предсказания: {str(e)}"
            }

# Создаем глобальный экземпляр
ml_model = PytorchMerchML()