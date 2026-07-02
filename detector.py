"""
Детектор ИИ-текста на базе энкодера BAAI/bge-m3.
Поддерживает бинарную классификацию и атрибуцию.
Кэширует эмбеддинги для повторяющихся текстов.
"""
import os
import gc
import numpy as np
import joblib
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "BAAI/bge-m3"
LOCAL_PATH = "bge_m3_local"
MAX_LENGTH = 512

def load_encoder(local_path, model_name):
    """Загружает bge-m3 локально или скачивает, если папки нет."""
    path_to_load = local_path if os.path.exists(local_path) else model_name
    
    tokenizer = AutoTokenizer.from_pretrained(path_to_load)
    model = AutoModel.from_pretrained(
        path_to_load,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    model = model.to(device)
    model.eval()
    return tokenizer, model, device


class AIDetector:
    """
    Детектор ИИ-текста.
    - predict_proba / predict — бинарная классификация (человек/ИИ)
    - predict_closest_model — атрибуция
    """

    def __init__(
        self,
        model_path: str = LOCAL_PATH,
        classifier_path: str = "classifier.joblib",
        attribution_classifier_path: str = None,
        attribution_labels_path: str = None,
        max_length: int = MAX_LENGTH,
        threshold: float = 0.5,
        cache_size: int = 128,
    ):
        self.max_length = max_length
        self.threshold = threshold
        self.cache_size = cache_size
        self._embedding_cache = {}

        # Загружаем bge-m3
        self.tokenizer, self.model, self.device = load_encoder(model_path, MODEL_NAME)

        # Бинарный классификатор (XGBoost)
        if not os.path.exists(classifier_path):
            raise FileNotFoundError(f"Файл классификатора {classifier_path} не найден. Сначала запустите train_model.py")
        self.classifier = joblib.load(classifier_path)

        # Опциональный классификатор атрибуции
        self.attribution_classifier = None
        self.attribution_labels = None
        if attribution_classifier_path and os.path.exists(attribution_classifier_path):
            self.attribution_classifier = joblib.load(attribution_classifier_path)
            if attribution_labels_path and os.path.exists(attribution_labels_path):
                self.attribution_labels = joblib.load(attribution_labels_path)

        status = " | атрибуция включена" if self.attribution_classifier is not None else ""
        print(f"Детектор готов | устройство: {self.device}{status}")

    def _get_embedding(self, text: str) -> np.ndarray:
        """Получение эмбеддинга с кэшированием (CLS + L2-нормализация)."""
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        # Токенизация
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        # Генерация эмбеддинга
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # CLS-пулинг (берем первый токен)
            emb = outputs.last_hidden_state[:, 0]
            # L2-нормализация
            emb = F.normalize(emb, p=2, dim=1)
            emb_np = emb.cpu().numpy()

        # Кэширование
        if len(self._embedding_cache) >= self.cache_size:
            self._embedding_cache.pop(next(iter(self._embedding_cache)))
        self._embedding_cache[text] = emb_np
        
        return emb_np

    def predict_proba(self, text: str) -> float:
        """Вероятность ИИ (0..1)."""
        emb = self._get_embedding(text)
        prob = self.classifier.predict_proba(emb)[0, 1]
        return float(prob)

    def predict(self, text: str, threshold: float = None) -> int:
        """Бинарная метка: 0 — человек, 1 — ИИ."""
        if threshold is None:
            threshold = self.threshold
        return 1 if self.predict_proba(text) >= threshold else 0

    def predict_closest_model(self, text: str) -> dict:
        """
        Возвращает вероятности для каждого класса (human + модели) в порядке убывания.
        """
        if self.attribution_classifier is None or self.attribution_labels is None:
            raise RuntimeError(
                "Классификатор атрибуции не загружен. Передайте "
                "attribution_classifier_path и attribution_labels_path в конструктор."
            )
        emb = self._get_embedding(text)
        probs = self.attribution_classifier.predict_proba(emb)[0]
        labels = self.attribution_labels.classes_
        ranked = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)
        return {label: float(p) for label, p in ranked}

    def __del__(self):
        """Освобождение ресурсов."""
        if hasattr(self, 'model'):
            del self.model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()