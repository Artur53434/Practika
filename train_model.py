#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Обучение бинарного детектора ИИ-текста на датасете LLMTrace_classification
(https://huggingface.co/datasets/iitolstykh/LLMTrace_classification).

Энкодер: BAAI/bge-m3, CLS-пулинг + L2-нормализация.
ВАЖНО: способ получения эмбеддинга здесь должен точно совпадать
с detector.py (тот же MODEL_NAME, тот же пулинг, тот же MAX_LENGTH),
иначе classifier.joblib на инференсе будет получать эмбеддинги другой
природы и выдавать бессмысленные вероятности.

Датасет билингвальный (en/ru), колонки:
- lang         : "eng" | "ru"
- label        : "human" | "ai"
- model        : имя LLM (None для человека)
- data_type    : домен текста (news, article, poetry, ...)
- prompt_type  : тип промпта (create/expand/delete/update, None для человека)
- topic_id     : id группы связанных текстов
- text         : сам текст
- prompt       : промпт, использованный для генерации (None для человека)

Используется не более SAMPLE_SIZE примеров суммарно (по умолчанию 10 000).
Эмбеддинги кэшируются с учётом языка(ов), доменов и размера выборки.
"""

import os
import gc
import hashlib
import shutil
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import snapshot_download
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

# =============================================================================
#  НАСТРОЙКА ПЕРЕМЕННЫХ ОКРУЖЕНИЯ ДЛЯ MPS (управление памятью)
# =============================================================================
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "1.5"
os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"]  = "1.2"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]      = "1"

# =============================================================================
#  НАСТРОЙКИ СКРИПТА
# =============================================================================
DATASET_NAME   = "iitolstykh/LLMTrace_classification"
DATASET_SPLIT  = "train"          # train / validation / test
LANGS          = ["ru", "eng"]           # ["ru"], ["eng"], либо ["ru", "eng"]
DATA_TYPES     = None             # None = все домены, либо список, напр. ["news", "article"]

CLF_PATH       = "classifier.joblib"
METRICS_PATH   = "metrics.txt"
TEST_SIZE      = 0.2
RANDOM_STATE   = 42
SAMPLE_SIZE    = 50000
USE_CACHE      = True
DO_GRID_SEARCH = False
CV_FOLDS       = 5

# Параметры энкодера — ДОЛЖНЫ совпадать с detector.py
MODEL_NAME = "BAAI/bge-m3"
LOCAL_PATH = "bge_m3_local"
MAX_LENGTH = 512
BATCH_SIZE = 16

# =============================================================================
#  ПРОВЕРКА ЦЕЛОСТНОСТИ ЛОКАЛЬНОЙ МОДЕЛИ
# =============================================================================
def ensure_model_integrity(model_name: str, local_dir: str) -> None:
    """Проверяет целостность локальной модели и при необходимости скачивает/докачивает её."""
    required_files = ['config.json', 'tokenizer.json']

    if os.path.exists(local_dir):
        all_ok = True
        for fname in required_files:
            fpath = os.path.join(local_dir, fname)
            if not os.path.isfile(fpath) or os.path.getsize(fpath) == 0:
                print(f"❌ Файл {fname} отсутствует или повреждён (размер 0).")
                all_ok = False
                break
        if all_ok:
            has_weights = any(
                os.path.isfile(os.path.join(local_dir, wf)) and os.path.getsize(os.path.join(local_dir, wf)) > 0
                for wf in ('model.safetensors','pytorch_model.bin')
            )
            if has_weights:
                print("✅ Модель уже загружена и проверена.")
                return
            print("❌ Не найден файл весов (model.safetensors или pytorch_model.bin).")
            all_ok = False

        if not all_ok:
            print("⚠️ Обнаружены повреждённые файлы модели. Удаляем папку и скачиваем заново.")
            shutil.rmtree(local_dir)

    print(f"⏳ Скачиваем модель {model_name} в {local_dir} ...")
    snapshot_download(
        repo_id=model_name,
        local_dir=local_dir,
        ignore_patterns=["*.h5", "*.ot", "*.msgpack"],
    )
    print("✅ Модель успешно скачана.")

# =============================================================================
#  ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ИЗ LLMTrace
# =============================================================================
def load_llmtrace(dataset_name, split, langs, data_types):
    """
    Загружает LLMTrace_classification через datasets.load_dataset,
    фильтрует по языку(ам) и, опционально, по доменам (data_type),
    приводит label к бинарному виду: human -> 0, ai -> 1.
    """
    print(f"Загрузка датасета {dataset_name} (split={split})...")
    ds = load_dataset(dataset_name, split=split)
    df = ds.to_pandas()

    if langs is not None:
        df = df[df["lang"].isin(langs)]

    if data_types is not None:
        df = df[df["data_type"].isin(data_types)]

    df = df[["text", "label", "lang", "data_type", "model", "prompt_type", "topic_id"]].copy()
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].map({"human": 0, "ai": 1})

    if df["label"].isnull().any():
        bad = df["label"].isnull().sum()
        print(f"⚠️ Обнаружено {bad} строк с неизвестным значением label, они будут отброшены.")
        df = df.dropna(subset=["label"])

    df["label"] = df["label"].astype(int)
    df = df.reset_index(drop=True)

    print(f"Загружено строк после фильтрации: {len(df)}")
    print(f"Баланс классов:\n{df['label'].value_counts()}")
    return df

# =============================================================================
#  ФУНКЦИИ ДЛЯ РАБОТЫ С ЭМБЕДДИНГАМИ (bge-m3, CLS + L2-нормализация)
# =============================================================================
def load_encoder(local_path, model_name):
    """Загружает модель bge-m3 (предварительно проверив целостность)."""
    ensure_model_integrity(model_name, local_path)

    print(f"Загрузка модели из {local_path}...")
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    model = AutoModel.from_pretrained(
        local_path,
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

def get_embeddings_batch(texts, tokenizer, model, device, max_length, batch_size):
    """Генерация эмбеддингов с CLS-пулингом и L2-нормализацией (как в detector.py)."""
    class TextDataset(Dataset):
        def __init__(self, texts, tokenizer, max_length):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            encoded = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            return {
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0)
            }

    dataset = TextDataset(texts, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Генерация эмбеддингов"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            sentence_embeddings = outputs.last_hidden_state[:, 0]
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            all_embeddings.append(sentence_embeddings.cpu().numpy())

    return np.vstack(all_embeddings)

def save_embeddings(embeddings, labels, path):
    np.save(path, {"embeddings": embeddings, "labels": labels})

def load_embeddings(path):
    data = np.load(path, allow_pickle=True).item()
    return data["embeddings"], data["labels"]

def get_combined_cache_path(dataset_name, split, langs, data_types, sample_size, task="binary"):
    langs_part = "-".join(sorted(langs)) if langs else "all"
    types_part = "-".join(sorted(data_types)) if data_types else "all"
    base = f"{dataset_name}_{split}_{langs_part}_{types_part}_sample{sample_size if sample_size else 'all'}"
    hash_obj = hashlib.sha256(base.encode('utf-8'))
    cache_dir = "embeddings_cache"
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{task}_{hash_obj.hexdigest()[:16]}.npy")

# =============================================================================
#  ОСНОВНАЯ ФУНКЦИЯ
# =============================================================================
def main():
    # ---------- 1. Загрузка данных ----------
    train_df = load_llmtrace(DATASET_NAME, DATASET_SPLIT, LANGS, DATA_TYPES)

    if SAMPLE_SIZE is not None and len(train_df) > SAMPLE_SIZE:
        train_df, _ = train_test_split(
            train_df, train_size=SAMPLE_SIZE,
            stratify=train_df['label'],
            random_state=RANDOM_STATE
        )
        train_df = train_df.reset_index(drop=True)
        print(f"Используется для обучения: {len(train_df)} строк")

    X_texts = train_df['text'].tolist()
    y = train_df['label'].tolist()

    # ---------- 2. Эмбеддинги (с кэшированием) ----------
    cache_file = None
    embeddings = None

    if USE_CACHE:
        cache_file = get_combined_cache_path(DATASET_NAME, DATASET_SPLIT, LANGS, DATA_TYPES, SAMPLE_SIZE, task="binary")
        if os.path.exists(cache_file):
            print(f"Загрузка эмбеддингов из кэша: {cache_file}")
            embeddings, y_cached = load_embeddings(cache_file)
            if not np.array_equal(y, y_cached):
                print("Метки не совпадают, пересчёт эмбеддингов...")
                embeddings = None

    if embeddings is None:
        print("Генерация эмбеддингов...")
        tokenizer, model, device = load_encoder(LOCAL_PATH, MODEL_NAME)

        embeddings = get_embeddings_batch(
            X_texts, tokenizer, model, device,
            max_length=MAX_LENGTH, batch_size=BATCH_SIZE
        )

        del model, tokenizer
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

        if USE_CACHE and cache_file:
            save_embeddings(embeddings, y, cache_file)

    print(f"Эмбеддинги готовы: {embeddings.shape}")

    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        print("⚠️ В эмбеддингах обнаружены NaN/Inf значения! Проверьте точность модели (float16 на MPS).")

    # ---------- 3. Train/test split ----------
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # ---------- 4. Подбор гиперпараметров (опционально) ----------
    if DO_GRID_SEARCH:
        print("Подбор гиперпараметров через GridSearchCV...")
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
        base_clf = LogisticRegression(max_iter=2000, solver='lbfgs', n_jobs=-1, class_weight=None)
        grid = GridSearchCV(base_clf, param_grid, cv=CV_FOLDS, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_C = grid.best_params_['C']
        print(f"Лучший C: {best_C}")
        clf = grid.best_estimator_
    else:
        clf = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs', n_jobs=-1, class_weight=None)
        clf.fit(X_train, y_train)

    # ---------- 5. Оценка на тестовой выборке ----------
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["human", "ai"])
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nROC-AUC: {roc_auc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nClassification report:")
    print(report)
    print("Confusion matrix:")
    print(cm)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(f"Датасет: {DATASET_NAME} (split={DATASET_SPLIT}, langs={LANGS}, data_types={DATA_TYPES})\n")
        f.write(f"Энкодер: {MODEL_NAME}\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n\n")
        f.write(report)
        f.write("\nConfusion matrix:\n")
        f.write(str(cm))
    print(f"Метрики сохранены в {METRICS_PATH}")

    # ---------- 6. Кросс-валидация ----------
    cv_scores = cross_val_score(clf, embeddings, y, cv=CV_FOLDS, scoring='roc_auc')
    print(f"\nСредний ROC-AUC по {CV_FOLDS}-кратной кросс-валидации: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # ---------- 7. Финальный классификатор на всех данных ----------
    print("\nОбучение финального классификатора на всех данных...")
    final_clf = LogisticRegression(max_iter=2000, C=clf.C, solver='lbfgs', n_jobs=-1, class_weight=None)
    final_clf.fit(embeddings, y)

    joblib.dump(final_clf, CLF_PATH)
    print(f"Классификатор сохранён: {CLF_PATH}")

if __name__ == "__main__":
    main()