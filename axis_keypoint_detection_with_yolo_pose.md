# Определение координат осей графика с помощью YOLO-Pose

## 1. Постановка задачи

**Цель:** автоматически определять **пиксельные координаты начала осей и их концов** на изображениях графиков.

Мы хотим получить:
- `origin (x0, y0)` — начало координат (пересечение осей)
- `x_end (x1, y1)` — конец оси X
- `y_end (x2, y2)` — конец оси Y

Результат должен быть:
- в **пикселях**
- устойчивым к разным стилям графиков (сетка, линии, толщина, масштаб)

---

## 2. Почему ChartReader оказался недостаточным

ChartReader изначально ориентирован на:
- детекцию графических элементов
- эвристический парсинг осей

Проблемы для нашей задачи:
- точки осей представлены как `1×1 bbox`
- heatmap почти нулевой → модель теряет `x_end / y_end`
- сложная отладка преобразований координат

Вывод: **задача “координаты осей” лучше формулируется как keypoint detection**, а не object detection.

---

## 3. Выбранное решение: YOLO-Pose (Keypoint Detection)

Мы используем **YOLO-Pose (Ultralytics YOLO v8)**:
- модель обучается предсказывать **ключевые точки**
- bbox используется только как вспомогательная сущность

### Почему YOLO-Pose подходит идеально
- нативная поддержка keypoints
- нет проблемы `1×1 bbox`
- корректная работа с resize / letterbox
- простой inference и получение численных координат

Мы моделируем **один объект `chart` с 3 ключевыми точками**:
1. `origin`
2. `x_end`
3. `y_end`

---

## 4. Формат и структура датасета

### Исходная структура (COCO-подобная)

```
axis_data/
├── annotations/
│   ├── train.json
│   └── val.json
├── images/
│   ├── train/
│   └── val/
```

Аннотации:
- каждая точка оси — отдельная COCO-аннотация
- bbox вида `[x, y, 1, 1]`
- категории: `origin`, `x_end`, `y_end`

---

## 5. Конвертация датасета в YOLO-Pose формат

### Результат конвертации

После запуска конвертера появляется:

```
axis_data/
├── labels/
│   ├── train/
│   │   ├── image_001.txt
│   │   └── ...
│   └── val/
│       ├── image_101.txt
│       └── ...
```

Картинки **не меняются**.

---

### Формат одного label-файла

```
0 xc yc w h  x0 y0 2  x1 y1 2  x2 y2 2
```

Где:
- `0` — класс `chart`
- `xc yc w h` — bbox (нормализованный)
- `(x0,y0)` — origin
- `(x1,y1)` — x_end
- `(x2,y2)` — y_end
- `2` — точка видима

Все координаты нормализованы в диапазоне `[0..1]`.

---

### data.yaml

```yaml
path: axis_data
train: images/train
val: images/val

names:
  0: chart

kpt_shape: [3, 3]
```

Порядок keypoints **строго фиксирован**:
1. origin
2. x_end
3. y_end

---

## 6. Установка и запуск обучения

### Установка

```bash
python3 -m venv venv
source venv/bin/activate
pip install ultralytics
```

---

### Запуск обучения

```bash
yolo pose train \
  model=yolov8s-pose.pt \
  data=axis_data/data.yaml \
  imgsz=768 \
  epochs=150 \
  batch=8
```

Результаты сохраняются в:
```
runs/pose/train/
```

---

## 7. Валидация и метрики

### Валидация

```bash
yolo pose val \
  model=runs/pose/train/weights/best.pt \
  data=axis_data/data.yaml \
  imgsz=640
```

Результаты:
```
runs/pose/val/
├── val_batch0.jpg   # визуализация GT + pred
├── results.png      # графики
├── results.csv      # метрики
```

Основная метрика:
- `metrics/pose_mAP50`

---

## 8. Предсказания и визуальная проверка

```bash
yolo pose predict \
  model=runs/pose/train/weights/best.pt \
  source=axis_data/images/val \
  save=True
```

Результаты:
```
runs/pose/predict/
```

На изображениях:
- bbox — вспомогательный
- цветные точки — **ключевые точки осей**

---

## 9. Получение численных координат (главное)

### Вариант 1: через `save_txt`

```bash
yolo pose predict \
  model=runs/pose/train/weights/best.pt \
  source=axis_data/images/val \
  save=True \
  save_txt=True
```

Файлы:
```
runs/pose/predict/labels/*.txt
```

Координаты — нормализованные.

Перевод в пиксели:
```
px = x * image_width
py = y * image_height
```

---

### Вариант 2 (рекомендуется): Python API

```python
from ultralytics import YOLO
import cv2

model = YOLO("best.pt")
img = cv2.imread("image.png")
h, w = img.shape[:2]

r = model(img)[0]
kpts = r.keypoints.xy[0]

origin = tuple(map(int, kpts[0]))
x_end  = tuple(map(int, kpts[1]))
y_end  = tuple(map(int, kpts[2]))
```

На выходе — **пиксельные координаты осей**.

---

## 10. Итог

- Задача корректно решается как **keypoint detection**
- YOLO-Pose — простое и стабильное решение
- Датасет легко конвертируется из COCO
- Модель даёт **точные пиксельные координаты**
- Решение масштабируется на большой датасет и продакшен

Это решение полностью покрывает исходную задачу определения координат начала и концов осей графиков.

