"""
MerchAI v2 - Анализ полок магазина
YOLO8 Object Detection + ResNet18 Classification + Business Logic
"""

import os
import uuid
import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
from collections import Counter
import json
from datetime import datetime

class ShelfAnalyzer:
    """Анализатор полок магазина - YOLO + ResNet18 + Бизнес-логика"""
    
    def __init__(self, classifier_model=None):
        """
        Инициализация анализатора
        classifier_model - наша обученная ResNet18 модель
        """
        self.classifier = classifier_model
        self.yolo_model = None
        self.temp_dir = "temp_shelf_analysis"
        
        # Создаем папку для временных файлов
        os.makedirs(self.temp_dir, exist_ok=True)
        
        print("🔍 Инициализация ShelfAnalyzer...")
        
        # Загружаем YOLO модель
        try:
            self.yolo_model = YOLO('yolov8n.pt')  # Nano версия - быстрая
            print("✅ YOLOv8 модель загружена")
        except Exception as e:
            print(f"❌ Ошибка загрузки YOLO: {e}")
            print("💡 Установите: pip install ultralytics")
    
    def detect_objects_on_shelf(self, image_path: str, confidence_threshold: float = 0.3) -> List[Dict]:
        """
        Детекция всех объектов на полке через YOLOv8
        Возвращает список координат найденных объектов
        """
        if self.yolo_model is None:
            raise Exception("YOLO модель не загружена")
        
        print(f"🔍 Анализируем полку: {image_path}")
        
        # Запускаем детекцию
        results = self.yolo_model(image_path, conf=confidence_threshold)
        detections = results[0].boxes
        
        if detections is None or len(detections) == 0:
            print("⚠️ Объекты на полке не найдены")
            return []
        
        objects_found = []
        for i, box in enumerate(detections):
            # Координаты bounding box
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            
            # YOLO классы (нас интересуют bottle=39, cup=41, bowl=51)
            yolo_classes = {39: 'bottle', 41: 'cup', 51: 'bowl', 0: 'person'}
            object_type = yolo_classes.get(class_id, f'object_{class_id}')
            
            objects_found.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': confidence,
                'class_id': class_id,
                'object_type': object_type,
                'crop_id': f'crop_{i}'
            })
        
        print(f"🎯 Найдено объектов на полке: {len(objects_found)}")
        return objects_found
    
    def crop_and_classify_objects(self, shelf_image_path: str, objects: List[Dict]) -> List[Dict]:
        """
        Вырезаем каждый объект и классифицируем через ResNet18
        """
        if self.classifier is None:
            raise Exception("Классификатор (ResNet18) не найден")
        
        print("✂️ Вырезаем и классифицируем объекты...")
        
        shelf_image = Image.open(shelf_image_path)
        classified_objects = []
        
        for i, obj in enumerate(objects):
            bbox = obj['bbox']
            x1, y1, x2, y2 = bbox
            
            # Вырезаем объект с небольшим отступом
            padding = 10
            crop_box = (
                max(0, x1 - padding),
                max(0, y1 - padding), 
                min(shelf_image.width, x2 + padding),
                min(shelf_image.height, y2 + padding)
            )
            
            cropped = shelf_image.crop(crop_box)
            
            # Сохраняем временно
            crop_filename = f"crop_{i}_{uuid.uuid4().hex[:8]}.jpg"
            crop_path = os.path.join(self.temp_dir, crop_filename)
            cropped.save(crop_path, 'JPEG')
            
            try:
                # Классифицируем через ResNet18
                classification_result = self.classifier.predict_image(crop_path, top_k=1)
                
                if (classification_result["status"] == "success" and 
                    classification_result["results"] and 
                    len(classification_result["results"]) > 0):
                    
                    result = classification_result["results"][0]
                    product_name = result["product"]
                    classification_confidence = result["confidence"]
                    
                    # Объединяем данные YOLO + ResNet18
                    classified_objects.append({
                        **obj,  # Данные от YOLO
                        'product_name': product_name,
                        'classification_confidence': classification_confidence,
                        'crop_path': crop_path,
                        'detected': classification_confidence >= self.classifier.confidence_threshold
                    })
                    
                    print(f"📦 Объект {i+1}: {product_name} ({classification_confidence:.1f}%)")
                else:
                    classified_objects.append({
                        **obj,
                        'product_name': 'Неизвестный товар',
                        'classification_confidence': 0.0,
                        'crop_path': crop_path,
                        'detected': False
                    })
                    print(f"❓ Объект {i+1}: Не распознан")
                    
            except Exception as e:
                print(f"❌ Ошибка классификации объекта {i+1}: {e}")
                classified_objects.append({
                    **obj,
                    'product_name': 'Ошибка классификации',
                    'classification_confidence': 0.0,
                    'crop_path': crop_path,
                    'detected': False
                })
        
        return classified_objects
    
    def calculate_shelf_statistics(self, classified_objects: List[Dict]) -> Dict:
        """
        Подсчет статистики по полке
        """
        print("📊 Считаем статистику полки...")
        
        # Фильтруем только распознанные товары
        recognized_products = [
            obj for obj in classified_objects 
            if obj['detected'] and obj['product_name'] != 'Неизвестный товар'
        ]
        
        if len(recognized_products) == 0:
            return {
                "total_detected": len(classified_objects),
                "total_recognized": 0,
                "products": [],
                "message": "Ни один товар не был распознан с достаточной уверенностью"
            }
        
        # Подсчет по товарам
        product_names = [obj['product_name'] for obj in recognized_products]
        product_counts = Counter(product_names)
        total_recognized = len(recognized_products)
        
        # Формируем статистику
        products_stats = []
        for product, count in product_counts.most_common():
            percentage = (count / total_recognized) * 100
            products_stats.append({
                'name': product,
                'count': count,
                'percentage': round(percentage, 1)
            })
        
        return {
            "total_detected": len(classified_objects),
            "total_recognized": total_recognized,
            "recognition_rate": round((total_recognized / len(classified_objects)) * 100, 1),
            "products": products_stats
        }
    
    def generate_business_recommendations(self, stats: Dict, target_brands: List[str] = None, target_percentage: float = 51.0) -> Dict:
        """
        Бизнес-логика: рекомендации для торгового представителя
        """
        if target_brands is None:
            target_brands = ['ruz', 'руз']  # Наши бренды по умолчанию
        
        recommendations = {}
        
        if len(stats["products"]) == 0:
            return {"message": "Нет данных для анализа"}
        
        total_recognized = stats["total_recognized"]
        
        # Анализируем каждый целевой бренд
        for brand in target_brands:
            brand_lower = brand.lower()
            
            # Находим товары нашего бренда
            brand_products = [
                p for p in stats["products"]
                if brand_lower in p["name"].lower()
            ]
            
            if not brand_products:
                recommendations[f"{brand}_analysis"] = {
                    "status": "Товары бренда не найдены на полке",
                    "current_count": 0,
                    "current_percentage": 0.0,
                    "action": f"КРИТИЧНО: Добавить товары {brand} на полку!"
                }
                continue
            
            # Суммируем все товары бренда
            brand_total_count = sum(p["count"] for p in brand_products)
            brand_percentage = sum(p["percentage"] for p in brand_products)
            
            # Расчет рекомендаций
            if brand_percentage >= target_percentage:
                action = f"✅ Отлично! Доля {brand} составляет {brand_percentage:.1f}%"
                need_add = 0
            else:
                # Сколько нужно добавить для достижения целевой доли
                target_count = int((total_recognized * target_percentage / 100)) + 1
                current_other = total_recognized - brand_total_count
                
                # Новое общее количество после добавления наших товаров
                need_add = max(1, target_count - brand_total_count)
                new_total = total_recognized + need_add
                new_brand_percentage = ((brand_total_count + need_add) / new_total) * 100
                
                action = f"📈 Добавить {need_add} товаров {brand} для достижения {target_percentage}%"
            
            recommendations[f"{brand}_analysis"] = {
                "brand": brand,
                "current_count": brand_total_count,
                "current_percentage": round(brand_percentage, 1),
                "target_percentage": target_percentage,
                "need_add": need_add,
                "action": action,
                "products_on_shelf": [p["name"] for p in brand_products]
            }
        
        return recommendations
    
    def create_annotated_image(self, shelf_image_path: str, classified_objects: List[Dict], output_path: str = None) -> str:
        """
        Создает изображение с аннотациями (bounding boxes + подписи)
        """
        if output_path is None:
            output_path = os.path.join(self.temp_dir, f"annotated_shelf_{uuid.uuid4().hex[:8]}.jpg")
        
        image = Image.open(shelf_image_path)
        draw = ImageDraw.Draw(image)
        
        for obj in classified_objects:
            bbox = obj['bbox']
            x1, y1, x2, y2 = bbox
            
            # Цвет рамки в зависимости от распознанности
            if obj['detected']:
                color = 'green'
                text = f"{obj['product_name']} ({obj['classification_confidence']:.0f}%)"
            else:
                color = 'red'
                text = "Не распознан"
            
            # Рисуем bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Подпись
            text_bbox = draw.textbbox((x1, y1-20), text)
            draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], 
                         fill=color, outline=color)
            draw.text((x1, y1-20), text, fill='white')
        
        image.save(output_path, 'JPEG')
        print(f"🖼️ Аннотированное изображение сохранено: {output_path}")
        return output_path
    
    def analyze_shelf_complete(self, shelf_image_path: str, 
                             target_brands: List[str] = None,
                             target_percentage: float = 51.0,
                             save_annotated: bool = True) -> Dict:
        """
        ПОЛНЫЙ АНАЛИЗ ПОЛКИ: детекция → классификация → статистика → рекомендации
        """
        analysis_start_time = datetime.now()
        
        print(f"🚀 Начинаем полный анализ полки: {shelf_image_path}")
        
        try:
            # ШАГ 1: Детекция объектов через YOLO
            objects = self.detect_objects_on_shelf(shelf_image_path)
            if len(objects) == 0:
                return {
                    "status": "error",
                    "message": "На полке не найдено объектов для анализа"
                }
            
            # ШАГ 2: Классификация каждого объекта через ResNet18
            classified_objects = self.crop_and_classify_objects(shelf_image_path, objects)
            
            # ШАГ 3: Статистика
            stats = self.calculate_shelf_statistics(classified_objects)
            
            # ШАГ 4: Бизнес-рекомендации
            recommendations = self.generate_business_recommendations(stats, target_brands, target_percentage)
            
            # ШАГ 5: Аннотированное изображение (опционально)
            annotated_image_path = None
            if save_annotated:
                annotated_image_path = self.create_annotated_image(shelf_image_path, classified_objects)
            
            # Время анализа
            analysis_time = (datetime.now() - analysis_start_time).total_seconds()
            
            result = {
                "status": "success",
                "analysis_timestamp": analysis_start_time.isoformat(),
                "analysis_time_seconds": round(analysis_time, 2),
                "detection_summary": {
                    "total_objects_detected": len(objects),
                    "objects_recognized": stats["total_recognized"],
                    "recognition_rate_percent": stats.get("recognition_rate", 0)
                },
                "shelf_statistics": stats,
                "business_recommendations": recommendations,
                "annotated_image": annotated_image_path,
                "detailed_objects": classified_objects  # Для отладки
            }
            
            print(f"✅ Анализ завершен за {analysis_time:.1f}с")
            print(f"📊 Найдено: {len(objects)} объектов, распознано: {stats['total_recognized']}")
            
            return result
            
        except Exception as e:
            print(f"❌ Ошибка анализа полки: {e}")
            return {
                "status": "error",
                "message": f"Ошибка анализа: {str(e)}",
                "analysis_timestamp": analysis_start_time.isoformat()
            }
        finally:
            # Очистка временных файлов (опционально)
            self.cleanup_temp_files()
    
    def cleanup_temp_files(self):
        """Очистка временных файлов"""
        try:
            for filename in os.listdir(self.temp_dir):
                if filename.startswith('crop_'):
                    file_path = os.path.join(self.temp_dir, filename)
                    os.remove(file_path)
            print("🧹 Временные файлы очищены")
        except Exception as e:
            print(f"⚠️ Не удалось очистить временные файлы: {e}")


# Функция для интеграции с FastAPI
def create_shelf_analyzer_with_model(ml_model):
    """
    Фабричная функция для создания анализатора с нашей ML моделью
    """
    return ShelfAnalyzer(classifier_model=ml_model)


# Пример использования
if __name__ == "__main__":
    # Тестирование (нужна обученная модель)
    try:
        from pytorch_model import ml_model
        
        analyzer = ShelfAnalyzer(classifier_model=ml_model)
        
        # Пример анализа
        test_image = "test_shelf.jpg"  # Путь к тестовому фото полки
        
        if os.path.exists(test_image):
            result = analyzer.analyze_shelf_complete(
                shelf_image_path=test_image,
                target_brands=['ruz', 'pepsi'],
                target_percentage=51.0
            )
            
            print("\n" + "="*50)
            print("РЕЗУЛЬТАТ АНАЛИЗА ПОЛКИ:")
            print("="*50)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"❌ Тестовое изображение {test_image} не найдено")
            
    except ImportError:
        print("⚠️ ML модель не найдена - запустите из основного приложения")