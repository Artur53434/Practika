#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Обучение бинарного детектора ИИ-текста на датасете LLMTrace_classification.
Энкодер: BAAI/bge-m3, CLS-пулинг + L2-нормализация.
Классификатор: LightGBM (нелинейный градиентный бустинг).
Дополнительные признаки: стилометрические фичи (с MATTR вместо TTR)
+ перплексити под лёгкой LM (gpt2, 124M).

ИСПРАВЛЕНИЯ:
- Перплексити теперь считается батчами (уважается --perplexity-batch-size)
  с динамическим уменьшением батча при OOM.
- Модель перплексити — лёгкая gpt2 (124M), выбрана из-за проблем со
  скачиванием более тяжёлой bloom-560m.
- TTR заменён на MATTR (moving average TTR) с окном 100 слов,
  чтобы уменьшить зависимость от длины текста.
- MAX_LENGTH увеличен до 1024 (можно менять через --max-length).
- Добавлены параметры для настройки гиперпараметров LightGBM.
- Убраны частые вызовы torch.mps.empty_cache() в цикле перплексити
  (оставлены только при OOM и после завершения).
- Ускорен расчёт перплексити за счёт батчевой обработки.

ОПТИМИЗАЦИИ ДЛЯ M1 PRO 16GB:
- float16 для MPS (экономия ~50% памяти)
- attn_implementation="eager" только на MPS (обходим баги SDPA)
- Обработка датасета через Arrow без to_pandas()
- PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
- Предвыделение массива эмбеддингов
- torch.mps.empty_cache() после каждого батча энкодера
- Кэширование стилометрии и перплексити на диске
"""
import argparse
import copy
import gc
import hashlib
import json
import logging
import os
import re
import shutil
from typing import Optional, Sequence, Tuple, Dict, Any, List
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset, ClassLabel
from huggingface_hub import snapshot_download
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, log_loss,
)
import lightgbm as lgb
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

# =============================================================================
#  ЛОГИРОВАНИЕ
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("train_model.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("train_model")

# =============================================================================
#  НАСТРОЙКА ПЕРЕМЕННЫХ ОКРУЖЕНИЯ ДЛЯ MPS
# =============================================================================
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.7"
os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.5"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# Ограничиваем число потоков torch — на M1 это снижает пиковую память
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
# ВАЖНО: в процессе одновременно загружены две разные копии libomp.dylib —
# одна из PyTorch, другая из LightGBM (homebrew). При параллельном построении
# гистограмм (n_jobs>1) это приводит к сегфолту внутри lib_lightgbm.dylib
# (конфликт thread-pool/barrier между двумя рантаймами OpenMP). Обходной путь:
# разрешаем дублирование рантайма (иначе часто падает ещё раньше, на импорте)
# и ниже принудительно ограничиваем LightGBM одним потоком (n_jobs=1).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# =============================================================================
#  КОНСТАНТЫ
# =============================================================================
MODEL_NAME = "BAAI/bge-m3"
LOCAL_PATH = "bge_m3_local"
MAX_LENGTH = 1024  # увеличено с 512 до 1024
EMBEDDING_DIM = 1024
POOLING = "cls"
# Лёгкая модель для перплексити: gpt2 (124M) — вдвое легче bloom-560m (560M),
# canonical-репозиторий на HF (маленький, один файл весов, не должен давать
# проблем при скачивании). Токенизатор byte-level BPE технически работает
# с любым UTF-8 текстом, включая русский, хотя качество сигнала на русском
# будет несколько ниже, чем на английском (модель англо-центричная).
# Если нужен более качественный сигнал на русском при сохранении лёгкого веса,
# альтернатива: "ai-forever/rugpt3small_based_on_gpt2" (125M, но плохо
# справляется с чисто английскими текстами).
PERPLEXITY_MODEL_NAME = "gpt2"
PERPLEXITY_LOCAL_PATH = "gpt2_local"
PERPLEXITY_MAX_LENGTH = 256  # оставляем 256 для экономии памяти

# =============================================================================
#  ПАРАМЕТРЫ ЭКСПЕРИМЕНТА
# =============================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Обучение бинарного детектора ИИ-текста (bge-m3 + стилометрия + LightGBM)."
    )
    parser.add_argument("--dataset-name", type=str, default="iitolstykh/LLMTrace_classification")
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--langs", type=str, nargs="*", default=["ru", "eng"])
    parser.add_argument("--data-types", type=str, nargs="*", default=None)
    parser.add_argument("--clf-path", type=str, default="classifier.joblib")
    parser.add_argument("--metrics-path", type=str, default="metrics.txt")
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--min-text-length", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size-of-train", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Батч энкодера при генерации эмбеддингов.")
    parser.add_argument("--perplexity-batch-size", type=int, default=8,
                        help="Батч для расчёта перплексити (для gpt2 можно 8-16).")
    parser.add_argument("--perplexity-max-length", type=int, default=PERPLEXITY_MAX_LENGTH,
                        help="Макс. длина для перплексити.")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH,
                        help="Макс. длина входного текста для энкодера bge-m3.")
    parser.add_argument("--lgbm-n-estimators", type=int, default=1000,
                        help="Максимальное число деревьев LightGBM (early stopping остановит раньше).")
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.01,
                        help="Скорость обучения LightGBM.")
    parser.add_argument("--lgbm-max-depth", type=int, default=5,
                        help="Максимальная глубина дерева.")
    parser.add_argument("--lgbm-num-leaves", type=int, default=31,
                        help="Максимальное число листьев.")
    parser.add_argument("--early-stopping-patience", type=int, default=30,
                        help="Сколько итераций без улучшения val log-loss ждать.")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-n-estimators", type=int, default=200,
                        help="Число деревьев LightGBM внутри каждого фолда CV.")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-stylometric", action="store_true",
                        help="Не использовать стилометрические признаки и перплексити.")
    parser.add_argument("--error-analysis-topn", type=int, default=20)
    parser.add_argument("--mattr-window", type=int, default=100,
                        help="Размер окна для MATTR (заменяет TTR).")
    return parser

# =============================================================================
#  ПРОВЕРКА ЦЕЛОСТНОСТИ ЛОКАЛЬНОЙ МОДЕЛИ
# =============================================================================
def ensure_model_integrity(model_name: str, local_dir: str) -> None:
    required_files = ["config.json", "tokenizer.json"]
    if os.path.exists(local_dir):
        all_ok = all(
            os.path.isfile(os.path.join(local_dir, f)) and os.path.getsize(os.path.join(local_dir, f)) > 0
            for f in required_files
        )
        if all_ok:
            has_weights = any(
                os.path.isfile(os.path.join(local_dir, wf)) and os.path.getsize(os.path.join(local_dir, wf)) > 0
                for wf in ("model.safetensors", "pytorch_model.bin")
            )
            if has_weights:
                log.info("Модель %s уже загружена и проверена.", model_name)
                return
        shutil.rmtree(local_dir)

    log.info("Скачиваем модель %s в %s ...", model_name, local_dir)
    snapshot_download(
        repo_id=model_name,
        local_dir=local_dir,
        ignore_patterns=["*.h5", "*.ot", "*.msgpack"],
    )
    log.info("Модель %s успешно скачана.", model_name)

# =============================================================================
#  ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# =============================================================================
def load_llmtrace(
    dataset_name: str,
    split: str,
    langs: Optional[Sequence[str]],
    data_types: Optional[Sequence[str]],
    sample_size: Optional[int],
    min_text_length: int,
    random_state: int,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    log.info("Загрузка датасета %s (split=%s)...", dataset_name, split)
    ds = load_dataset(dataset_name, split=split)
    if langs:
        ds = ds.filter(lambda x: x["lang"] in langs)
    if data_types:
        ds = ds.filter(lambda x: x["data_type"] in data_types)
    ds = ds.filter(lambda x: x["text"] is not None and len(x["text"]) >= min_text_length)

    def map_labels(example: Dict[str, Any]) -> Dict[str, Any]:
        if example["label"] == "human":
            example["label"] = 0
        elif example["label"] == "ai":
            example["label"] = 1
        else:
            example["label"] = -1
        return example

    ds = ds.map(map_labels)
    ds = ds.filter(lambda x: x["label"] != -1)
    ds = ds.cast_column("label", ClassLabel(num_classes=2))

    from collections import Counter
    label_counts = Counter(ds["label"])
    log.info("Загружено строк после фильтрации: %d", len(ds))
    log.info("Баланс классов: %s", dict(label_counts))

    if sample_size is not None and len(ds) > sample_size:
        ds = ds.train_test_split(
            test_size=len(ds) - sample_size,
            seed=random_state,
            stratify_by_column="label",
        )["train"]
        log.info("После стратифицированной выборки: %d строк", len(ds))

    texts: List[str] = ds["text"]
    labels = np.array(ds["label"], dtype=np.int8)
    groups = np.array(ds["topic_id"])
    return texts, labels, groups

# =============================================================================
#  ЗАГРУЗКА ЭНКОДЕРА
# =============================================================================
def load_encoder(local_path: str, model_name: str):
    ensure_model_integrity(model_name, local_path)
    log.info("Загрузка энкодера из %s...", local_path)
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16
        attn_impl = "eager"
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16
        attn_impl = "sdpa"
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        attn_impl = "sdpa"

    model = AutoModel.from_pretrained(
        local_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=attn_impl,
    )
    model = model.to(device)
    model.eval()
    log.info("Энкодер: устройство=%s, точность=%s, attention=%s", device, dtype, attn_impl)
    return tokenizer, model, device

# =============================================================================
#  ЗАГРУЗКА МОДЕЛИ ДЛЯ ПЕРПЛЕКСИТИ (GPT-2)
# =============================================================================
def load_perplexity_model(local_path: str, model_name: str):
    ensure_model_integrity(model_name, local_path)
    log.info("Загрузка модели для перплексити из %s...", local_path)
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()
    log.info("Модель перплексити: устройство=%s, точность=%s", device, dtype)
    return tokenizer, model, device

# =============================================================================
#  ГЕНЕРАЦИЯ ЭМБЕДДИНГОВ
# =============================================================================
class TextDataset(Dataset):
    def __init__(self, texts: Sequence[str], tokenizer, max_length: int):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

def collate_fn_factory(tokenizer):
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [item["attention_mask"] for item in batch],
            batch_first=True,
            padding_value=0,
        )
        return {"input_ids": input_ids, "attention_mask": attention_mask}
    return collate_fn

def get_embeddings_batch_resumable(
    texts: Sequence[str],
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
    batch_size: int,
    checkpoint_path: str,
    checkpoint_every_batches: int = 20,
) -> np.ndarray:
    n = len(texts)
    progress_path = checkpoint_path + ".progress.json"

    if os.path.exists(checkpoint_path) and os.path.exists(progress_path):
        all_embeddings = np.memmap(checkpoint_path, dtype=np.float32, mode="r+", shape=(n, EMBEDDING_DIM))
        with open(progress_path, "r", encoding="utf-8") as f:
            progress = json.load(f)
        start_idx = progress.get("completed_rows", 0)
        log.info("Найден чекпоинт эмбеддингов: продолжаем с строки %d/%d", start_idx, n)
    else:
        all_embeddings = np.memmap(checkpoint_path, dtype=np.float32, mode="w+", shape=(n, EMBEDDING_DIM))
        start_idx = 0

    if start_idx >= n:
        log.info("Эмбеддинги уже полностью посчитаны.")
        result = np.array(all_embeddings)
        _cleanup_checkpoint(checkpoint_path, progress_path)
        return result

    remaining_texts = texts[start_idx:]
    dataset = TextDataset(remaining_texts, tokenizer, max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_factory(tokenizer),
    )

    current_idx = start_idx
    try:
        with torch.inference_mode():
            for batch_num, batch in enumerate(
                tqdm(dataloader, desc="Генерация эмбеддингов", initial=0)
            ):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                sentence_embeddings = outputs.last_hidden_state[:, 0]
                sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
                batch_np = sentence_embeddings.detach().cpu().float().numpy()
                batch_len = batch_np.shape[0]
                all_embeddings[current_idx: current_idx + batch_len] = batch_np
                current_idx += batch_len

                del outputs, sentence_embeddings, batch_np, input_ids, attention_mask
                if device.type == "mps":
                    torch.mps.empty_cache()

                if (batch_num + 1) % checkpoint_every_batches == 0:
                    all_embeddings.flush()
                    with open(progress_path, "w", encoding="utf-8") as f:
                        json.dump({"completed_rows": current_idx}, f)
    except Exception:
        all_embeddings.flush()
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump({"completed_rows": current_idx}, f)
        log.exception("Генерация эмбеддингов прервана. Прогресс сохранён.")
        raise

    all_embeddings.flush()
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump({"completed_rows": current_idx}, f)
    result = np.array(all_embeddings)
    _cleanup_checkpoint(checkpoint_path, progress_path)
    return result

def _cleanup_checkpoint(checkpoint_path: str, progress_path: str) -> None:
    for p in (checkpoint_path, progress_path):
        if os.path.exists(p):
            os.remove(p)

# =============================================================================
#  РАСЧЁТ ПЕРПЛЕКСИТИ (GPT-2) — БАТЧЕВЫЙ С OOM-FALLBACK
# =============================================================================
def _compute_ppl_batch(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    """
    Считает средний log-loss на токен для батча текстов.
    Возвращает массив float длины len(texts).
    В случае OOM уменьшает batch_size рекурсивно.
    """
    if not texts:
        return np.array([])

    # Токенизация с паддингом
    encodings = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    # Проверка, что длина последовательности >= 2
    if input_ids.size(1) < 2:
        # Слишком короткие тексты -> возвращаем 0
        return np.zeros(len(texts), dtype=np.float32)

    try:
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :].contiguous()
            labels = input_ids[:, 1:].contiguous()
            mask = attention_mask[:, 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            token_losses = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            token_losses = token_losses.view(labels.size())

            # Средний log-loss на токен по каждому элементу батча
            seq_loss = (token_losses * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            result = seq_loss.detach().cpu().float().numpy()

            del outputs, logits, labels, mask, token_losses, seq_loss
            if device.type == "mps":
                torch.mps.empty_cache()
            return result
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        # Если OOM, уменьшаем batch_size вдвое и пробуем снова
        if batch_size <= 1:
            log.warning("OOM даже при batch_size=1, возвращаем 0 для этих текстов")
            return np.zeros(len(texts), dtype=np.float32)
        new_batch_size = batch_size // 2
        log.warning("OOM при batch_size=%d, уменьшаем до %d", batch_size, new_batch_size)
        # Рекурсивно обрабатываем по частям
        results = []
        for i in range(0, len(texts), new_batch_size):
            chunk = texts[i:i+new_batch_size]
            chunk_results = _compute_ppl_batch(
                chunk, tokenizer, model, device, max_length, new_batch_size
            )
            results.extend(chunk_results)
        return np.array(results, dtype=np.float32)


def compute_perplexity(
    texts: Sequence[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
    checkpoint_path: str,
) -> np.ndarray:
    """
    Считает средний log-loss на токен для каждого текста под GPT-2.
    Меньше значение = более "человечный" текст (обычно).

    Использует батчевую обработку с динамическим уменьшением размера батча при OOM.
    Сохраняет прогресс в memmap.
    """
    n = len(texts)
    progress_path = checkpoint_path + ".progress.json"

    if os.path.exists(checkpoint_path) and os.path.exists(progress_path):
        mmap = np.memmap(checkpoint_path, dtype=np.float32, mode="r+", shape=(n, 1))
        with open(progress_path, "r", encoding="utf-8") as f:
            progress = json.load(f)
        start_idx = progress.get("completed_rows", 0)
        log.info("Найден чекпоинт перплексити: продолжаем с %d/%d", start_idx, n)
    else:
        mmap = np.memmap(checkpoint_path, dtype=np.float32, mode="w+", shape=(n, 1))
        start_idx = 0

    if start_idx >= n:
        log.info("Перплексити уже полностью посчитана.")
        result = np.array(mmap)
        _cleanup_checkpoint(checkpoint_path, progress_path)
        return result

    log.info("Расчёт перплексити для %d текстов (с %d), batch_size=%d...", n, start_idx, batch_size)

    current_idx = start_idx
    try:
        # Обрабатываем батчами
        for i in tqdm(range(start_idx, n, batch_size), desc="Perplexity (batched)", initial=start_idx//batch_size, total=(n+batch_size-1)//batch_size):
            end = min(i + batch_size, n)
            chunk_texts = list(texts[i:end])
            chunk_results = _compute_ppl_batch(
                chunk_texts, tokenizer, model, device, max_length, batch_size
            )
            # Записываем в memmap
            mmap[i:end, 0] = chunk_results.reshape(-1)
            current_idx = end

            # Освобождаем память после каждого батча (но не слишком часто)
            if device.type == "mps":
                torch.mps.empty_cache()

            # Чекпоинт каждые 10 батчей
            if (end) % (batch_size * 10) < batch_size:
                mmap.flush()
                with open(progress_path, "w", encoding="utf-8") as f:
                    json.dump({"completed_rows": current_idx}, f)
    except Exception:
        mmap.flush()
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump({"completed_rows": current_idx}, f)
        log.exception("Расчёт перплексити прерван. Прогресс сохранён в %s", progress_path)
        raise

    mmap.flush()
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump({"completed_rows": current_idx}, f)
    result = np.array(mmap)
    _cleanup_checkpoint(checkpoint_path, progress_path)
    log.info("Перплексити сохранена в %s", checkpoint_path)
    return result

# =============================================================================
#  РАСЧЁТ СТИЛОМЕТРИЧЕСКИХ ПРИЗНАКОВ (с MATTR вместо TTR)
# =============================================================================
def _compute_stylometric_single(text: str, mattr_window: int = 100) -> List[float]:
    """
    Считает стилометрические признаки для одного текста.
    Заменяет TTR на MATTR (moving average TTR) с окном mattr_window слов.
    """
    words = re.findall(r"\b\w+\b", text.lower())
    num_words = len(words)

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if s]
    sent_lengths = [len(re.findall(r"\b\w+\b", s)) for s in sentences]
    sent_len_var = float(np.var(sent_lengths)) if sent_lengths else 0.0

    # MATTR (Moving Average Type-Token Ratio)
    if num_words >= mattr_window:
        # Скользящее окно по словам
        ttrs = []
        for start in range(num_words - mattr_window + 1):
            window = words[start:start + mattr_window]
            ttr = len(set(window)) / mattr_window
            ttrs.append(ttr)
        mattr = float(np.mean(ttrs))
    elif num_words > 0:
        # Если слов меньше окна, считаем обычный TTR
        mattr = len(set(words)) / num_words
    else:
        mattr = 0.0

    # Повторяемость n-грамм (биграммы)
    if num_words > 1:
        bigrams = [tuple(words[i:i + 2]) for i in range(num_words - 1)]
        unique_bigrams = len(set(bigrams))
        rep_ngram_ratio = 1.0 - (unique_bigrams / len(bigrams))
    else:
        rep_ngram_ratio = 0.0

    punct_count = len(re.findall(r"[^\w\s]", text))
    punct_ratio = punct_count / len(text) if len(text) > 0 else 0.0

    avg_word_len = float(np.mean([len(w) for w in words])) if num_words > 0 else 0.0

    return [sent_len_var, mattr, rep_ngram_ratio, punct_ratio, avg_word_len]


def compute_stylometric_features(texts: Sequence[str], checkpoint_path: str, mattr_window: int = 100) -> np.ndarray:
    """
    Считает стилометрические признаки с memmap-чекпоинтингом.
    При падении продолжаем с последнего сохранённого места.
    """
    n = len(texts)
    n_features = 5
    progress_path = checkpoint_path + ".progress.json"

    if os.path.exists(checkpoint_path) and os.path.exists(progress_path):
        mmap = np.memmap(checkpoint_path, dtype=np.float32, mode="r+", shape=(n, n_features))
        with open(progress_path, "r", encoding="utf-8") as f:
            progress = json.load(f)
        start_idx = progress.get("completed_rows", 0)
        log.info("Найден чекпоинт стилометрии: продолжаем с %d/%d", start_idx, n)
    else:
        mmap = np.memmap(checkpoint_path, dtype=np.float32, mode="w+", shape=(n, n_features))
        start_idx = 0

    if start_idx >= n:
        log.info("Стилометрия уже полностью посчитана.")
        result = np.array(mmap)
        _cleanup_checkpoint(checkpoint_path, progress_path)
        return result

    log.info("Расчёт стилометрии для %d текстов (с %d)...", n, start_idx)

    current_idx = start_idx
    try:
        for i in tqdm(range(start_idx, n), desc="Стилометрия", initial=start_idx, total=n):
            features = _compute_stylometric_single(texts[i], mattr_window)
            mmap[i] = features
            current_idx = i + 1

            if (i + 1) % 1000 == 0:
                mmap.flush()
                with open(progress_path, "w", encoding="utf-8") as f:
                    json.dump({"completed_rows": current_idx}, f)
    except Exception:
        mmap.flush()
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump({"completed_rows": current_idx}, f)
        log.exception("Расчёт стилометрии прерван. Прогресс сохранён.")
        raise

    mmap.flush()
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump({"completed_rows": current_idx}, f)
    result = np.array(mmap)
    _cleanup_checkpoint(checkpoint_path, progress_path)
    log.info("Стилометрия сохранена в %s, форма: %s", checkpoint_path, result.shape)
    return result

# =============================================================================
#  КЭШИРОВАНИЕ
# =============================================================================
def save_full_cache(
    embeddings: np.ndarray, labels: np.ndarray, groups: np.ndarray, texts: Sequence[str],
    emb_path: str, labels_path: str, groups_path: str, texts_path: str,
) -> None:
    np.save(emb_path, embeddings)
    np.save(labels_path, labels)
    np.save(groups_path, groups)
    with open(texts_path, "w", encoding="utf-8") as f:
        json.dump(list(texts), f, ensure_ascii=False)

def load_full_cache(
    emb_path: str, labels_path: str, groups_path: str, texts_path: str, mmap: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    embeddings = np.load(emb_path, mmap_mode="r" if mmap else None)
    labels = np.load(labels_path)
    groups = np.load(groups_path)
    with open(texts_path, "r", encoding="utf-8") as f:
        texts = json.load(f)
    return embeddings, labels, groups, texts

def get_cache_paths(
    dataset_name: str, split: str, langs: Optional[Sequence[str]], data_types: Optional[Sequence[str]],
    sample_size: Optional[int], min_len: int, task: str = "binary",
) -> Tuple[str, str, str, str, str, str, str]:
    langs_part = "-".join(sorted(langs)) if langs else "all"
    types_part = "-".join(sorted(data_types)) if data_types else "all"
    sample_part = f"sample{sample_size}" if sample_size else "all"
    base = f"{dataset_name}_{split}_{langs_part}_{types_part}_{sample_part}_minlen{min_len}"
    hash_obj = hashlib.sha256(base.encode("utf-8"))
    cache_dir = "embeddings_cache"
    os.makedirs(cache_dir, exist_ok=True)
    prefix = os.path.join(cache_dir, f"{task}_{hash_obj.hexdigest()[:16]}")
    return (
        prefix + "_emb.npy",
        prefix + "_labels.npy",
        prefix + "_groups.npy",
        prefix + "_texts.json",
        prefix + "_emb_checkpoint.dat",
        prefix + "_perplexity.npy",
        prefix + "_stylometric.npy",
    )

# =============================================================================
#  ПОИСК ОПТИМАЛЬНОГО ПОРОГА
# =============================================================================
def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray, metric: str = "macro_f1") -> float:
    thresholds = np.linspace(0.01, 0.99, 197)
    best_score = -1.0
    best_threshold = 0.5
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        if metric == "macro_f1":
            score = f1_score(y_true, y_pred, average="macro", zero_division=0)
        elif metric == "f1_ai":
            score = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        elif metric == "f1_human":
            score = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
        else:
            raise ValueError(f"Неизвестная metric: {metric}")
        if score > best_score:
            best_score = score
            best_threshold = t
    return float(best_threshold)

# =============================================================================
#  ОБУЧЕНИЕ LightGBM С EARLY STOPPING
# =============================================================================
def train_lgbm_with_early_stopping(
    X: np.ndarray, y: np.ndarray,
    train_idx: np.ndarray, val_idx: np.ndarray,
    n_estimators: int, learning_rate: float, max_depth: int, num_leaves: int,
    patience: int, random_state: int, desc_prefix: str = "Обучение",
) -> Tuple[lgb.LGBMClassifier, int, List[float]]:
    counts = np.bincount(y[train_idx])
    scale_pos_weight = float(counts[0] / counts[1]) if counts[1] > 0 else 1.0
    log.info("LightGBM scale_pos_weight: %.3f", scale_pos_weight)

    clf = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=1,  # 1 поток: избегаем крэша из-за двух копий libomp.dylib (torch + lightgbm)
        verbose=-1,
    )

    eval_set = [(X[val_idx], y[val_idx])]

    log.info("%s LightGBM (max %d деревьев, early stopping patience=%d)...",
             desc_prefix, n_estimators, patience)

    clf.fit(
        X[train_idx], y[train_idx],
        eval_set=eval_set,
        callbacks=[
            lgb.early_stopping(stopping_rounds=patience),
            lgb.log_evaluation(period=50),
        ],
    )

    best_epoch = clf.best_iteration_
    val_loss_history = clf.evals_result_["valid_0"]["binary_logloss"]

    return clf, best_epoch, val_loss_history

# =============================================================================
#  ГРУППОВАЯ КРОСС-ВАЛИДАЦИЯ
# =============================================================================
def run_group_kfold_cv(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray,
    n_folds: int, n_estimators: int, learning_rate: float, max_depth: int, num_leaves: int,
    random_state: int,
) -> Dict[str, Any]:
    gkf = GroupKFold(n_splits=n_folds)
    fold_metrics = {"roc_auc": [], "f1_macro": []}

    for fold_i, (fold_train_idx, fold_test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        counts = np.bincount(y[fold_train_idx])
        scale_pos_weight = float(counts[0] / counts[1]) if counts[1] > 0 else 1.0

        clf = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=1,  # 1 поток: избегаем крэша из-за двух копий libomp.dylib (torch + lightgbm)
            verbose=-1,
        )

        clf.fit(
            X[fold_train_idx], y[fold_train_idx],
            eval_set=[(X[fold_test_idx], y[fold_test_idx])],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)],
        )

        y_proba = clf.predict_proba(X[fold_test_idx])[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        roc_auc = roc_auc_score(y[fold_test_idx], y_proba)
        f1_macro = f1_score(y[fold_test_idx], y_pred, average="macro", zero_division=0)

        fold_metrics["roc_auc"].append(roc_auc)
        fold_metrics["f1_macro"].append(f1_macro)
        log.info("CV fold %d/%d: ROC-AUC=%.4f, macro-F1=%.4f", fold_i + 1, n_folds, roc_auc, f1_macro)

    summary = {
        "roc_auc_mean": float(np.mean(fold_metrics["roc_auc"])),
        "roc_auc_std": float(np.std(fold_metrics["roc_auc"])),
        "f1_macro_mean": float(np.mean(fold_metrics["f1_macro"])),
        "f1_macro_std": float(np.std(fold_metrics["f1_macro"])),
        "per_fold": fold_metrics,
    }
    return summary

# =============================================================================
#  АНАЛИЗ ОШИБОК
# =============================================================================
def analyze_errors(
    texts_test: Sequence[str], y_test: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray,
    groups_test: np.ndarray, top_n: int, output_path: str,
) -> None:
    lengths = np.array([len(t) for t in texts_test])
    fp_mask = (y_pred == 1) & (y_test == 0)
    fn_mask = (y_pred == 0) & (y_test == 1)
    correct_mask = y_pred == y_test

    def _top_examples(mask: np.ndarray, ascending: bool) -> List[Dict[str, Any]]:
        idxs = np.where(mask)[0]
        order = np.argsort(y_proba[idxs]) if ascending else np.argsort(-y_proba[idxs])
        idxs = idxs[order][:top_n]
        return [
            {
                "text_snippet": texts_test[i][:300],
                "topic_id": str(groups_test[i]),
                "true_label": int(y_test[i]),
                "pred_label": int(y_pred[i]),
                "proba_ai": float(y_proba[i]),
                "text_length": int(lengths[i]),
            }
            for i in idxs
        ]

    fp_examples = _top_examples(fp_mask, ascending=False)
    fn_examples = _top_examples(fn_mask, ascending=True)

    from collections import defaultdict
    topic_errors: Dict[str, int] = defaultdict(int)
    topic_total: Dict[str, int] = defaultdict(int)
    for i in range(len(y_test)):
        topic = str(groups_test[i])
        topic_total[topic] += 1
        if y_pred[i] != y_test[i]:
            topic_errors[topic] += 1

    worst_topics = sorted(
        ((t, topic_errors[t], topic_total[t], topic_errors[t] / topic_total[t]) for t in topic_errors),
        key=lambda x: -x[3],
    )[:top_n]

    length_stats = {
        "errors_mean_length": float(lengths[~correct_mask].mean()) if (~correct_mask).any() else None,
        "errors_median_length": float(np.median(lengths[~correct_mask])) if (~correct_mask).any() else None,
        "correct_mean_length": float(lengths[correct_mask].mean()) if correct_mask.any() else None,
        "correct_median_length": float(np.median(lengths[correct_mask])) if correct_mask.any() else None,
    }

    report = {
        "false_positives": fp_examples,
        "false_negatives": fn_examples,
        "worst_topics_by_error_rate": [
            {"topic_id": t, "errors": e, "total": tot, "error_rate": rate}
            for t, e, tot, rate in worst_topics
        ],
        "length_stats": length_stats,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log.info(
        "Анализ ошибок сохранён в %s (FP: %d, FN: %d)",
        output_path, int(fp_mask.sum()), int(fn_mask.sum()),
    )

# =============================================================================
#  MAIN
# =============================================================================
def main() -> None:
    args = build_arg_parser().parse_args()

    # Обновляем глобальные константы из CLI
    global MAX_LENGTH, PERPLEXITY_MAX_LENGTH
    MAX_LENGTH = args.max_length
    PERPLEXITY_MAX_LENGTH = args.perplexity_max_length

    # ---------- 1. Загрузка данных ----------
    (emb_path, labels_path, groups_path, texts_path, checkpoint_path,
     perplexity_path, stylometric_path) = get_cache_paths(
        args.dataset_name, args.dataset_split, args.langs, args.data_types,
        args.sample_size, args.min_text_length, task="binary",
    )

    use_cache = (not args.no_cache) and os.path.exists(emb_path) and os.path.exists(texts_path)
    if use_cache:
        log.info("Загрузка эмбеддингов из кэша...")
        embeddings, y, groups, texts = load_full_cache(emb_path, labels_path, groups_path, texts_path, mmap=True)
        log.info("Эмбеддинги загружены, форма: %s", embeddings.shape)
    else:
        texts, y, groups = load_llmtrace(
            args.dataset_name, args.dataset_split, args.langs, args.data_types,
            args.sample_size, args.min_text_length, args.random_state,
        )
        log.info("Используется для обучения: %d строк", len(texts))
        log.info("Генерация эмбеддингов...")
        tokenizer, model, device = load_encoder(LOCAL_PATH, MODEL_NAME)
        embeddings = get_embeddings_batch_resumable(
            texts, tokenizer, model, device,
            max_length=MAX_LENGTH, batch_size=args.batch_size,
            checkpoint_path=checkpoint_path,
        )
        save_full_cache(embeddings, y, groups, texts, emb_path, labels_path, groups_path, texts_path)
        log.info("Эмбеддинги сохранены в %s", emb_path)
        del model, tokenizer
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        del embeddings
        gc.collect()
        embeddings, y, groups, texts = load_full_cache(emb_path, labels_path, groups_path, texts_path, mmap=True)

    texts = list(texts)

    # ---------- 2. Стилометрия и перплексити ----------
    if not args.no_stylometric:
        log.info("Расчёт дополнительных признаков (стилометрия + перплексити)...")

        # Стилометрия с MATTR
        stylometric = compute_stylometric_features(texts, checkpoint_path=stylometric_path, mattr_window=args.mattr_window)

        # Перплексити
        if os.path.exists(perplexity_path) and not args.no_cache:
            # Проверяем целостность кэша
            try:
                cached = np.load(perplexity_path)
                if cached.shape[0] == len(texts):
                    log.info("Загрузка перплексити из кэша...")
                    perplexity = cached
                else:
                    log.warning("Кэш перплексити повреждён (размер %d != %d), пересчитываем",
                                cached.shape[0], len(texts))
                    raise ValueError("size mismatch")
            except Exception:
                ppl_tokenizer, ppl_model, ppl_device = load_perplexity_model(
                    PERPLEXITY_LOCAL_PATH, PERPLEXITY_MODEL_NAME,
                )
                perplexity = compute_perplexity(
                    texts, ppl_tokenizer, ppl_model, ppl_device,
                    batch_size=args.perplexity_batch_size,
                    max_length=args.perplexity_max_length,
                    checkpoint_path=perplexity_path,
                )
                del ppl_model, ppl_tokenizer
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
        else:
            ppl_tokenizer, ppl_model, ppl_device = load_perplexity_model(
                PERPLEXITY_LOCAL_PATH, PERPLEXITY_MODEL_NAME,
            )
            perplexity = compute_perplexity(
                texts, ppl_tokenizer, ppl_model, ppl_device,
                batch_size=args.perplexity_batch_size,
                max_length=args.perplexity_max_length,
                checkpoint_path=perplexity_path,
            )
            del ppl_model, ppl_tokenizer
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        # Конкатенация
        X = np.hstack([embeddings, perplexity, stylometric])
        log.info("Итоговая матрица признаков X: %s (эмбеддинги + перплексити + стилометрия)", X.shape)
    else:
        X = embeddings
        log.info("Стилометрические признаки отключены. X = embeddings, форма: %s", X.shape)

    del embeddings
    gc.collect()

    # ---------- 3. Train/val/test split ----------
    gss_test = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
    train_val_idx, test_idx = next(gss_test.split(X, y, groups=groups))
    gss_val = GroupShuffleSplit(n_splits=1, test_size=args.val_size_of_train, random_state=args.random_state)
    train_sub_idx, val_idx_local = next(gss_val.split(
        X[train_val_idx], y[train_val_idx], groups=groups[train_val_idx],
    ))
    train_idx = train_val_idx[train_sub_idx]
    val_idx = train_val_idx[val_idx_local]
    log.info("Train: %d, Val: %d, Test: %d", len(train_idx), len(val_idx), len(test_idx))

    # ---------- 4. GroupKFold CV ----------
    if args.cv_folds and args.cv_folds > 1:
        log.info("Запуск %d-fold GroupKFold CV...", args.cv_folds)
        cv_summary = run_group_kfold_cv(
            X, y, groups, n_folds=args.cv_folds,
            n_estimators=args.cv_n_estimators,
            learning_rate=args.lgbm_learning_rate,
            max_depth=args.lgbm_max_depth,
            num_leaves=args.lgbm_num_leaves,
            random_state=args.random_state,
        )
        log.info(
            "CV результат: ROC-AUC = %.4f ± %.4f, macro-F1 = %.4f ± %.4f",
            cv_summary["roc_auc_mean"], cv_summary["roc_auc_std"],
            cv_summary["f1_macro_mean"], cv_summary["f1_macro_std"],
        )
    else:
        cv_summary = None

    # ---------- 5. Обучение LightGBM с early stopping ----------
    eval_clf, best_epoch, val_loss_history = train_lgbm_with_early_stopping(
        X, y, train_idx, val_idx,
        n_estimators=args.lgbm_n_estimators,
        learning_rate=args.lgbm_learning_rate,
        max_depth=args.lgbm_max_depth,
        num_leaves=args.lgbm_num_leaves,
        patience=args.early_stopping_patience,
        random_state=args.random_state,
        desc_prefix="Обучение (eval)",
    )
    log.info("Early stopping выбрал %d итераций как оптимальные.", best_epoch)

    # ---------- 6. Калибровка вероятностей ----------
    y_proba_val_raw = eval_clf.predict_proba(X[val_idx])[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(y_proba_val_raw, y[val_idx])
    y_proba_val_calibrated = calibrator.predict(y_proba_val_raw)

    best_threshold = find_best_threshold(y[val_idx], y_proba_val_calibrated, metric="macro_f1")
    log.info("Оптимальный порог (макс. macro-F1, калиброванные val-вероятности): %.3f", best_threshold)

    # ---------- 7. Оценка на TEST ----------
    y_proba_test_raw = eval_clf.predict_proba(X[test_idx])[:, 1]
    y_proba_test = calibrator.predict(y_proba_test_raw)
    y_test = y[test_idx]
    y_pred_tuned = (y_proba_test >= best_threshold).astype(int)

    # ---------- 8. Финальное обучение на всех данных ----------
    log.info("Финальное обучение LightGBM на всех данных (%d итераций)...", best_epoch)

    counts = np.bincount(y)
    scale_pos_weight = float(counts[0] / counts[1]) if counts[1] > 0 else 1.0

    final_clf = lgb.LGBMClassifier(
        n_estimators=best_epoch,
        learning_rate=args.lgbm_learning_rate,
        max_depth=args.lgbm_max_depth,
        num_leaves=args.lgbm_num_leaves,
        scale_pos_weight=scale_pos_weight,
        random_state=args.random_state,
        n_jobs=1,  # 1 поток: избегаем крэша из-за двух копий libomp.dylib (torch + lightgbm)
        verbose=-1,
    )
    final_clf.fit(X, y)

    joblib.dump(
        {
            "model": final_clf,
            "threshold": best_threshold,
            "calibrator": calibrator,
            "pooling": POOLING,
            "use_stylometric": not args.no_stylometric,
        },
        args.clf_path,
    )
    log.info("Финальный классификатор сохранён: %s", args.clf_path)

    # ---------- 9. Метрики ----------
    y_pred = y_pred_tuned
    roc_auc = roc_auc_score(y_test, y_proba_test)
    f1 = f1_score(y_test, y_pred)
    precision_ai = precision_score(y_test, y_pred, pos_label=1)
    recall_ai = recall_score(y_test, y_pred, pos_label=1)
    report = classification_report(y_test, y_pred, target_names=["human", "ai"])
    cm = confusion_matrix(y_test, y_pred)

    log.info("ROC-AUC: %.4f", roc_auc)
    log.info("F1-score: %.4f", f1)
    log.info("Precision (ai): %.4f", precision_ai)
    log.info("Recall (ai): %.4f", recall_ai)
    log.info("\n%s", report)
    log.info("Confusion matrix:\n%s", cm)

    with open(args.metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Датасет: {args.dataset_name} (split={args.dataset_split}, langs={args.langs})\n")
        f.write(f"Размер выборки: {len(X)}\n")
        f.write(f"Train/Val/Test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}\n")
        f.write(f"Мин. длина текста: {args.min_text_length}\n")
        f.write(f"Энкодер: {MODEL_NAME} (pooling={POOLING}, max_length={MAX_LENGTH})\n")
        f.write(f"Стилометрические признаки: {'Да' if not args.no_stylometric else 'Нет'}\n")
        f.write(f"MATTR окно: {args.mattr_window}\n")
        f.write(f"Perplexity модель: {PERPLEXITY_MODEL_NAME}, max_length={args.perplexity_max_length}\n")
        f.write(f"Классификатор: LightGBM (n_estimators={best_epoch}, lr={args.lgbm_learning_rate}, max_depth={args.lgbm_max_depth}, num_leaves={args.lgbm_num_leaves})\n")
        f.write(f"История val log-loss (последние 5): {val_loss_history[-5:]}\n")
        if cv_summary is not None:
            f.write(f"\nGroupKFold CV ({args.cv_folds} фолдов):\n")
            f.write(f"  ROC-AUC: {cv_summary['roc_auc_mean']:.4f} ± {cv_summary['roc_auc_std']:.4f}\n")
            f.write(f"  macro-F1: {cv_summary['f1_macro_mean']:.4f} ± {cv_summary['f1_macro_std']:.4f}\n")
        f.write(f"\nROC-AUC (test): {roc_auc:.4f}\n")
        f.write(f"F1-score (test): {f1:.4f}\n")
        f.write(f"Precision (ai, test): {precision_ai:.4f}\n")
        f.write(f"Recall (ai, test): {recall_ai:.4f}\n")
        f.write(f"Оптимальный порог: {best_threshold:.3f}\n")
        f.write(report)
        f.write("\nConfusion matrix (test):\n")
        f.write(str(cm))
    log.info("Метрики сохранены в %s", args.metrics_path)

    # ---------- 10. Анализ ошибок ----------
    texts_test = [texts[i] for i in test_idx]
    analyze_errors(
        texts_test, y_test, y_pred, y_proba_test, groups[test_idx],
        top_n=args.error_analysis_topn, output_path="error_analysis.json",
    )

    # ---------- 11. Очистка ----------
    del X, y, groups
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    log.info("Скрипт завершён, память очищена.")

if __name__ == "__main__":
    main()