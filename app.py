from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import os
import sqlite3
import uuid
from datetime import datetime
from PIL import Image
import shutil
from typing import List, Optional
import json

# Импорт ML модели
try:
    from pytorch_model import ml_model
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️  ML модель не найдена - будут работать заглушки")

# Импорт анализатора полок
try:
    from shelf_analyzer import create_shelf_analyzer_with_model
    shelf_analyzer = create_shelf_analyzer_with_model(ml_model) if ML_AVAILABLE else None
    SHELF_ANALYZER_AVAILABLE = True
    print("🛒 Анализатор полок подключен")
except ImportError:
    shelf_analyzer = None
    SHELF_ANALYZER_AVAILABLE = False
    print("⚠️  Анализатор полок недоступен - установите ultralytics")

app = FastAPI(title="MerchAI - ML поиск товаров + Анализ полок", version="2.1")

# Настройки
UPLOAD_DIR_TRAIN = "uploads/train"
UPLOAD_DIR_SEARCH = "uploads/search" 
UPLOAD_DIR_SHELF = "uploads/shelf"  # Новая папка для полок
DB_NAME = "merchai_v2.db"
STATIC_DIR = "static"

# Создание директорий
os.makedirs(UPLOAD_DIR_TRAIN, exist_ok=True)
os.makedirs(UPLOAD_DIR_SEARCH, exist_ok=True)
os.makedirs(UPLOAD_DIR_SHELF, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs("ml_models", exist_ok=True)
os.makedirs("temp_shelf_analysis", exist_ok=True)

# Статические файлы
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

def init_database():
    """Инициализация базы данных"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Таблица товаров
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            created_at TEXT NOT NULL
        )
    ''')
    
    # Таблица изображений для обучения
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER,
            filename TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            file_size INTEGER,
            created_at TEXT NOT NULL,
            FOREIGN KEY (product_id) REFERENCES products (id)
        )
    ''')
    
    # Таблица ML признаков
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS image_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER,
            product_id INTEGER,
            features TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (image_id) REFERENCES training_images (id),
            FOREIGN KEY (product_id) REFERENCES products (id)
        )
    ''')
    
    # Новая таблица анализов полок
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS shelf_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            analysis_result TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    ''')
    
    conn.commit()
    conn.close()

def resize_image(image_path: str, size: tuple = (224, 224)):
    """Изменение размера изображения"""
    try:
        with Image.open(image_path) as img:
            # Конвертируем в RGB если нужно
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Ресайзим
            img_resized = img.resize(size, Image.Resampling.LANCZOS)
            img_resized.save(image_path, 'JPEG', quality=85)
            return True
    except Exception as e:
        print(f"❌ Ошибка ресайза: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске"""
    print("🚀 Запуск MerchAI v2.1 - ML + Shelf Analytics")
    init_database()
    print("✅ База данных инициализирована")
    
    if ML_AVAILABLE:
        print("🤖 ML модель подключена")
    else:
        print("⚠️  ML модель недоступна - режим заглушек")
    
    if SHELF_ANALYZER_AVAILABLE:
        print("🛒 Анализатор полок готов")
    else:
        print("⚠️  Анализатор полок недоступен")

@app.get("/")
async def root():
    """Главная страница"""
    return {
        "message": "MerchAI v2.1 - ML поиск товаров + Анализ полок", 
        "ml_available": ML_AVAILABLE,
        "shelf_analyzer_available": SHELF_ANALYZER_AVAILABLE
    }

# ПОЛУЧЕНИЕ СПИСКА ТОВАРОВ
@app.get("/products/list")
async def get_products():
    """Получить список всех товаров"""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM products ORDER BY name")
        products = cursor.fetchall()
        conn.close()
        
        return {
            "products": [{"id": product_id, "name": name} for product_id, name in products]
        }
    except Exception as e:
        return {"products": []}

# ОБУЧЕНИЕ - загрузка товаров
@app.post("/train/upload")
async def upload_training_data(
    product_name: str = Form(...),
    existing_product_id: Optional[str] = Form(None),
    product_description: str = Form(""),
    files: List[UploadFile] = File(...)
):
    """Загрузка товара с фотографиями для обучения"""
    
    try:
        print(f"🔍 DEBUG: Получен запрос - товар: '{product_name}', existing_id: {existing_product_id}, файлов: {len(files) if files else 0}")
        
        if not files:
            raise HTTPException(status_code=400, detail="Нет загруженных файлов")
        
        print(f"🔍 DEBUG: Создаем подключение к БД: {DB_NAME}")
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Определяем product_id
        if existing_product_id:
            # Добавляем к существующему товару
            product_id = int(existing_product_id)
            print(f"🔍 DEBUG: Добавляем фото к существующему товару ID: {product_id}")
            
            # Проверяем что товар существует
            cursor.execute("SELECT name FROM products WHERE id = ?", (product_id,))
            result = cursor.fetchone()
            if not result:
                raise HTTPException(status_code=400, detail="Выбранный товар не существует")
            
            product_name = result[0]  # Используем название из БД
            print(f"🔍 DEBUG: Название товара из БД: {product_name}")
        else:
            # Создаем новый товар
            print(f"🔍 DEBUG: Создаем новый товар в БД")
            cursor.execute(
                "INSERT INTO products (name, created_at) VALUES (?, ?)",
                (product_name, datetime.now().isoformat())
            )
            product_id = cursor.lastrowid
            print(f"🔍 DEBUG: Новый товар создан с ID: {product_id}")
        
        uploaded_files = []
        
        # Обрабатываем каждый файл
        for i, file in enumerate(files):
            print(f"🔍 DEBUG: Обрабатываем файл {i+1}: {file.filename}, тип: {file.content_type}")
            
            if not file.content_type.startswith('image/'):
                print(f"⚠️ DEBUG: Пропускаем файл {file.filename} - не изображение")
                continue
                
            # Генерируем уникальное имя файла
            file_extension = file.filename.split('.')[-1].lower()
            unique_filename = f"{uuid.uuid4()}.{file_extension}"
            file_path = os.path.join(UPLOAD_DIR_TRAIN, unique_filename)
            
            print(f"🔍 DEBUG: Сохраняем файл: {file_path}")
            
            # Сохраняем файл
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            print(f"🔍 DEBUG: Файл сохранен, размер: {len(content)} байт")
            
            # Ресайзим изображение
            print(f"🔍 DEBUG: Начинаем ресайз изображения")
            if resize_image(file_path):
                file_size = os.path.getsize(file_path)
                print(f"🔍 DEBUG: Ресайз успешен, новый размер: {file_size} байт")
                
                # Сохраняем в БД
                print(f"🔍 DEBUG: Сохраняем запись в training_images")
                cursor.execute(
                    "INSERT INTO training_images (product_id, filename, original_filename, file_size, created_at) VALUES (?, ?, ?, ?, ?)",
                    (product_id, unique_filename, file.filename, file_size, datetime.now().isoformat())
                )
                
                uploaded_files.append({
                    "filename": unique_filename,
                    "original": file.filename,
                    "size": file_size
                })
                print(f"✅ DEBUG: Файл {file.filename} успешно обработан")
            else:
                print(f"❌ DEBUG: Ошибка ресайза для {file.filename}")
                # Удаляем файл если ресайз не удался
                if os.path.exists(file_path):
                    os.remove(file_path)
        
        print(f"🔍 DEBUG: Коммитим транзакцию")
        conn.commit()
        
        action = "добавлено к существующему товару" if existing_product_id else "создан новый товар"
        result = {
            "status": "success",
            "message": f"Загружено {len(uploaded_files)} изображений ({action} '{product_name}')",
            "product_id": product_id,
            "files": uploaded_files
        }
        
        print(f"✅ DEBUG: Успешно завершено: {result}")
        return result
        
    except Exception as e:
        print(f"❌ ПОЛНАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки: {str(e)}")
    
    finally:
        if 'conn' in locals():
            conn.close()
            print(f"🔍 DEBUG: Подключение к БД закрыто")

# ОБУЧЕНИЕ - запуск ML обучения
@app.post("/train/start")
async def start_training():
    """Запуск обучения ML модели"""
    
    if not ML_AVAILABLE:
        return {
            "status": "error", 
            "message": "ML модель недоступна - установите зависимости (tensorflow, scikit-learn)"
        }
    
    try:
        # Проверяем наличие данных
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM training_images")
        image_count = cursor.fetchone()[0]
        conn.close()
        
        if image_count < 2:
            return {
                "status": "error",
                "message": f"Недостаточно данных для обучения. Загружено: {image_count} фото (нужно минимум 2)"
            }
        
        # Запуск обучения
        result = ml_model.train_model(epochs=15, batch_size=8)
        return result
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Ошибка обучения: {str(e)}"
        }

# ПОИСК - загрузка фото для поиска (заглушка)
@app.post("/search/upload") 
async def search_product_stub(file: UploadFile = File(...)):
    """Поиск товара по фото (заглушка)"""
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")
    
    # Сохраняем временный файл
    unique_filename = f"{uuid.uuid4()}.jpg"
    temp_path = os.path.join(UPLOAD_DIR_SEARCH, unique_filename)
    
    with open(temp_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    file_size = len(content)
    
    # Заглушка результата
    return {
        "status": "success",
        "message": "Фото обработано для поиска",
        "filename": unique_filename,
        "file_size": file_size,
        "results": [
            {"product": "Нет обученных товаров", "confidence": 0}
        ]
    }

# ПОИСК - ML предсказание
@app.post("/search/predict")
async def predict_product(file: UploadFile = File(...)):
    """ML предсказание товара по фото"""
    
    if not ML_AVAILABLE:
        return await search_product_stub(file)
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")
    
    # Сохраняем временный файл
    temp_path = f"temp_{uuid.uuid4()}.jpg"
    
    try:
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # ML предсказание
        result = ml_model.predict_image(temp_path)
        return result
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Ошибка предсказания: {str(e)}"
        }
    finally:
        # Удаляем временный файл
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ===============================================
# НОВЫЙ БЛОК: АНАЛИЗ ПОЛОК МАГАЗИНА 🛒
# ===============================================

@app.post("/shelf/analyze")
async def analyze_shelf(file: UploadFile = File(...)):
    """🛒 АНАЛИЗ ПОЛКИ - показать статистику товаров с процентами"""
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")
    
    # Сохраняем файл полки
    unique_filename = f"shelf_{uuid.uuid4()}.jpg"
    shelf_path = os.path.join(UPLOAD_DIR_SHELF, unique_filename)
    
    try:
        with open(shelf_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        print(f"🛒 Анализируем полку: {file.filename}")
        
        # ШАГ 1: YOLO детекция с МУЛЬТИФИЛЬТРАЦИЕЙ
        try:
            from ultralytics import YOLO
            yolo = YOLO('yolov8n.pt')
            
            # 🎯 НАСТРОЙКИ ФИЛЬТРАЦИИ:
            # conf=0.4 - минимальная уверенность (было 0.25)
            # iou=0.5 - фильтр перекрывающихся объектов  
            # classes=[39,41,47] - только bottle, cup, bowl
            results = yolo(shelf_path, conf=0.4, iou=0.5, classes=[39, 41, 47])  # bottle=39, cup=41, bowl=47
            detections = results[0].boxes
            
            if detections is not None and len(detections) > 0:
                # ФИЛЬТРАЦИЯ ПО РАЗМЕРУ ОБЪЕКТОВ
                filtered_boxes = []
                image_area = Image.open(shelf_path).size[0] * Image.open(shelf_path).size[1]
                
                for box in detections:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Размер объекта
                    box_width = x2 - x1
                    box_height = y2 - y1
                    box_area = box_width * box_height
                    
                    # ФИЛЬТРЫ:
                    # 1. Минимальный размер (не слишком маленький)
                    min_area = image_area * 0.005  # 0.5% от общей площади
                    # 2. Максимальный размер (не слишком большой) 
                    max_area = image_area * 0.3    # 30% от общей площади
                    # 3. Разумные пропорции (не слишком узкий/широкий)
                    aspect_ratio = box_width / box_height if box_height > 0 else 1
                    
                    if (box_area > min_area and 
                        box_area < max_area and 
                        0.2 < aspect_ratio < 5.0):  # Разумные пропорции
                        filtered_boxes.append(box)
                        print(f"✅ Объект принят: {box_width:.0f}x{box_height:.0f}, площадь: {box_area:.0f}")
                    else:
                        print(f"❌ Объект отфильтрован: {box_width:.0f}x{box_height:.0f}, площадь: {box_area:.0f}")
                
                detections = filtered_boxes
                print(f"🔍 После фильтрации осталось объектов: {len(detections)}")
            
            
            if len(detections) == 0:
                return {
                    "status": "success",
                    "message": "После фильтрации товары на полке не найдены",
                    "total_objects": 0,
                    "total_recognized": 0,
                    "products": []
                }
            
        except Exception as e:
            return {"status": "error", "message": f"Ошибка детекции: {str(e)}"}
        
        # ШАГ 2: Классификация отфильтрованных объектов
        shelf_image = Image.open(shelf_path)
        recognized_products = []
        
        for i, box in enumerate(detections):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence_yolo = float(box.conf[0].cpu().numpy())
            
            print(f"🔍 Обрабатываем объект {i+1}: YOLO уверенность {confidence_yolo*100:.1f}%")
            
            # Вырезаем объект
            padding = 15
            crop_box = (
                max(0, int(x1) - padding), max(0, int(y1) - padding), 
                min(shelf_image.width, int(x2) + padding), min(shelf_image.height, int(y2) + padding)
            )
            cropped = shelf_image.crop(crop_box)
            crop_path = f"temp_crop_{i}_{uuid.uuid4().hex[:8]}.jpg"
            cropped.save(crop_path)
            
            try:
                # Классифицируем через ResNet18
                if ML_AVAILABLE and ml_model:
                    classification = ml_model.predict_image(crop_path)
                    
                    if (classification["status"] == "success" and 
                        classification["results"] and len(classification["results"]) > 0):
                        
                        result = classification["results"][0]
                        product_name = result["product"]
                        confidence = result["confidence"]
                        
                        # Добавляем только если уверенность выше порога
                        if confidence >= ml_model.confidence_threshold:
                            recognized_products.append(product_name)
                            print(f"✅ Объект {i+1}: {product_name} ({confidence:.1f}%)")
                        else:
                            print(f"⚠️ Объект {i+1}: {product_name} ({confidence:.1f}%) - слишком низкая уверенность")
                    else:
                        print(f"❌ Объект {i+1}: Ошибка классификации")
                else:
                    print(f"❌ Объект {i+1}: ML недоступна")
                    
            except Exception as e:
                print(f"❌ Ошибка классификации объекта {i+1}: {e}")
            finally:
                if os.path.exists(crop_path):
                    os.remove(crop_path)
        
        # ШАГ 3: СТАТИСТИКА ПО ТОВАРАМ С ПРОЦЕНТАМИ
        from collections import Counter
        product_counts = Counter(recognized_products)
        
        total_objects = len(detections)
        total_recognized = len(recognized_products)
        
        # Формируем результат с процентами от общего числа распознанных товаров
        products_stats = []
        if total_recognized > 0:
            for product, count in product_counts.most_common():
                percentage = (count / total_recognized) * 100
                products_stats.append({
                    "name": product,
                    "count": count,
                    "percentage": round(percentage, 1)
                })
        
        result = {
            "status": "success",
            "message": f"На полке найдено {total_objects} объектов, распознано {total_recognized} товаров",
            "total_objects": total_objects,
            "total_recognized": total_recognized,
            "recognition_rate": round((total_recognized / total_objects * 100), 1) if total_objects > 0 else 0,
            "products": products_stats
        }
        
        print(f"🎯 СТАТИСТИКА ПОЛКИ:")
        print(f"   📦 Всего объектов: {total_objects}")
        print(f"   ✅ Распознано товаров: {total_recognized}")
        for product in products_stats:
            print(f"   • {product['name']}: {product['count']} шт ({product['percentage']}%)")
        
        return result
        
    except Exception as e:
        print(f"❌ Ошибка анализа полки: {e}")
        return {"status": "error", "message": f"Ошибка анализа: {str(e)}"}

@app.get("/shelf/annotated/{filename}")
async def get_annotated_image(filename: str):
    """Получить аннотированное изображение полки"""
    file_path = os.path.join("temp_shelf_analysis", filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Аннотированное изображение не найдено")
    
    return FileResponse(file_path, media_type="image/jpeg")

@app.get("/shelf/history")
async def get_shelf_analysis_history():
    """История анализов полок"""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, original_filename, created_at, analysis_result 
            FROM shelf_analyses 
            ORDER BY created_at DESC 
            LIMIT 20
        """)
        
        analyses = []
        for row in cursor.fetchall():
            analysis_id, original_filename, created_at, analysis_result = row
            
            # Парсим результат
            try:
                result_data = json.loads(analysis_result)
                summary = {
                    "total_recognized": result_data.get("detection_summary", {}).get("objects_recognized", 0),
                    "recognition_rate": result_data.get("detection_summary", {}).get("recognition_rate_percent", 0)
                }
            except:
                summary = {"error": "Не удалось парсить результат"}
            
            analyses.append({
                "id": analysis_id,
                "filename": original_filename,
                "created_at": created_at,
                "summary": summary
            })
        
        conn.close()
        return {"analyses": analyses}
        
    except Exception as e:
        return {"analyses": [], "error": str(e)}

# ===============================================

# СТАТУС МОДЕЛИ
@app.get("/model/status")
async def model_status():
    """Статус ML модели"""
    
    # Проверяем наличие обученной модели
    model_exists = os.path.exists("pytorch_models/latest_model.pth")
    
    # Считаем данные для обучения
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM training_images")
    image_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(DISTINCT product_id) FROM training_images") 
    product_count = cursor.fetchone()[0]
    
    # Считаем анализы полок
    cursor.execute("SELECT COUNT(*) FROM shelf_analyses")
    shelf_analyses_count = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        "ml_available": ML_AVAILABLE,
        "model_trained": model_exists,
        "shelf_analyzer_available": SHELF_ANALYZER_AVAILABLE,
        "training_data": {
            "images": image_count,
            "products": product_count
        },
        "shelf_analyses_count": shelf_analyses_count,
        "ready_for_training": image_count >= 2,
        "ready_for_prediction": model_exists and ML_AVAILABLE,
        "ready_for_shelf_analysis": SHELF_ANALYZER_AVAILABLE and model_exists
    }

# СТАТИСТИКА
@app.get("/stats")
async def get_stats():
    """Статистика системы"""
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Общая статистика
    cursor.execute("SELECT COUNT(*) FROM products")
    total_products = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM training_images")
    total_images = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM shelf_analyses")
    total_shelf_analyses = cursor.fetchone()[0]
    
    # По товарам
    cursor.execute("""
        SELECT p.name, COUNT(ti.id) as photo_count 
        FROM products p 
        LEFT JOIN training_images ti ON p.id = ti.product_id 
        GROUP BY p.name
    """)
    products_stats = cursor.fetchall()
    
    conn.close()
    
    return {
        "total_products": total_products,
        "total_images": total_images,
        "total_shelf_analyses": total_shelf_analyses,
        "products": [{"name": name, "photos": count} for name, count in products_stats]
    }

if __name__ == "__main__":
    print("🚀 Запуск MerchAI сервера v2.1...")
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="debug"
    )