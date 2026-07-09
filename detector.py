"""
Детектор ИИ-текста — прослойка между обученной моделью и GUI/сайтом.
Пайплайн ДОЛЖЕН точно совпадать с train_model.py.
"""
import os
import re
import gc
import numpy as np
import joblib
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# =============================================================================
# КОНСТАНТЫ — ДОЛЖНЫ СОВПАДАТЬ С train_model.py
# =============================================================================
MODEL_NAME = "BAAI/bge-m3"
LOCAL_PATH = "bge_m3_local"
MAX_LENGTH = 1024
POOLING = "cls"
PERPLEXITY_MODEL_NAME = "gpt2"
PERPLEXITY_LOCAL_PATH = "gpt2_local"
PERPLEXITY_MAX_LENGTH = 256
MATTR_WINDOW = 100


# =============================================================================
# ВЫБОР УСТРОЙСТВА
# =============================================================================
def _pick_device_dtype():
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    return torch.device("cpu"), torch.float32


# =============================================================================
# ЗАГРУЗКА МОДЕЛЕЙ
# =============================================================================
def load_encoder(local_path: str = LOCAL_PATH, model_name: str = MODEL_NAME):
    path_to_load = local_path if os.path.exists(local_path) else model_name
    tokenizer = AutoTokenizer.from_pretrained(path_to_load)
    device, dtype = _pick_device_dtype()
    model = AutoModel.from_pretrained(
        path_to_load, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    model = model.to(device)
    model.eval()
    return tokenizer, model, device


def load_perplexity_model(local_path: str = PERPLEXITY_LOCAL_PATH,
                          model_name: str = PERPLEXITY_MODEL_NAME):
    path_to_load = local_path if os.path.exists(local_path) else model_name
    tokenizer = AutoTokenizer.from_pretrained(path_to_load)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device, dtype = _pick_device_dtype()
    model = AutoModelForCausalLM.from_pretrained(
        path_to_load, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    model = model.to(device)
    model.eval()
    return tokenizer, model, device


# =============================================================================
# СТИЛОМЕТРИЯ (точная копия логики из train_model.py)
# =============================================================================
def compute_stylometric_single(text: str, mattr_window: int = MATTR_WINDOW) -> np.ndarray:
    words = re.findall(r"\b\w+\b", text.lower())
    num_words = len(words)
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if s]
    sent_lengths = [len(re.findall(r"\b\w+\b", s)) for s in sentences]
    sent_len_var = float(np.var(sent_lengths)) if sent_lengths else 0.0

    if num_words >= mattr_window:
        ttrs = []
        for start in range(num_words - mattr_window + 1):
            window = words[start:start + mattr_window]
            ttrs.append(len(set(window)) / mattr_window)
        mattr = float(np.mean(ttrs))
    elif num_words > 0:
        mattr = len(set(words)) / num_words
    else:
        mattr = 0.0

    if num_words > 1:
        bigrams = [tuple(words[i:i + 2]) for i in range(num_words - 1)]
        rep_ngram_ratio = 1.0 - (len(set(bigrams)) / len(bigrams))
    else:
        rep_ngram_ratio = 0.0

    punct_count = len(re.findall(r"[^\w\s]", text))
    punct_ratio = punct_count / len(text) if len(text) > 0 else 0.0
    avg_word_len = float(np.mean([len(w) for w in words])) if num_words > 0 else 0.0

    return np.array(
        [[sent_len_var, mattr, rep_ngram_ratio, punct_ratio, avg_word_len]],
        dtype=np.float32,
    )


# =============================================================================
# ДЕТЕКТОР
# =============================================================================
class AIDetector:
    def __init__(
        self,
        model_path: str = LOCAL_PATH,
        classifier_path: str = "classifier.joblib",
        threshold: float = None,
        cache_size: int = 64,
    ):
        self.cache_size = cache_size
        self._feature_cache = {}
        if not os.path.exists(classifier_path):
            raise FileNotFoundError(
                f"Файл классификатора {classifier_path} не найден. "
                f"Сначала запустите train_model.py"
            )
        bundle = joblib.load(classifier_path)
        self.model = bundle["model"]
        self.calibrator = bundle.get("calibrator")
        self.use_stylometric = bundle.get("use_stylometric", True)
        self.threshold = threshold if threshold is not None else bundle.get("threshold", 0.5)

        self.tokenizer, self.encoder, self.device = load_encoder(model_path, MODEL_NAME)

        self.ppl_tokenizer = self.ppl_model = self.ppl_device = None
        if self.use_stylometric:
            self.ppl_tokenizer, self.ppl_model, self.ppl_device = load_perplexity_model()

        status = "стилометрия+перплексити включены" if self.use_stylometric else "только эмбеддинги"
        print(f"Детектор готов | устройство: {self.device} | {status} | порог: {self.threshold:.3f}")

    # -------------------------------------------------------------------
    def _embed(self, text: str) -> np.ndarray:
        encoded = self.tokenizer(
            text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        with torch.inference_mode():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            emb = outputs.last_hidden_state[:, 0]
            emb = F.normalize(emb, p=2, dim=1)
        return emb.detach().cpu().float().numpy()

    def _perplexity(self, text: str) -> np.ndarray:
        encoded = self.ppl_tokenizer(
            text, truncation=True, max_length=PERPLEXITY_MAX_LENGTH, return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(self.ppl_device)
        attention_mask = encoded["attention_mask"].to(self.ppl_device)
        if input_ids.size(1) < 2:
            return np.zeros((1, 1), dtype=np.float32)
        with torch.inference_mode():
            outputs = self.ppl_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :].contiguous()
            labels = input_ids[:, 1:].contiguous()
            mask = attention_mask[:, 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            token_losses = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            token_losses = token_losses.view(labels.size())
            seq_loss = (token_losses * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return seq_loss.detach().cpu().float().numpy().reshape(1, 1)

    def _build_features(self, text: str) -> np.ndarray:
        emb = self._embed(text)
        if not self.use_stylometric:
            return emb
        ppl = self._perplexity(text)
        stylo = compute_stylometric_single(text)
        # Порядок важен: embeddings, perplexity, stylometric
        return np.hstack([emb, ppl, stylo])

    def _get_features_cached(self, text: str) -> np.ndarray:
        if text in self._feature_cache:
            return self._feature_cache[text]
        X = self._build_features(text)
        if len(self._feature_cache) >= self.cache_size:
            self._feature_cache.pop(next(iter(self._feature_cache)))
        self._feature_cache[text] = X
        return X

    # -------------------------------------------------------------------
    def predict_proba(self, text: str) -> float:
        X = self._get_features_cached(text)
        raw_proba = self.model.predict_proba(X)[0, 1]
        if self.calibrator is not None:
            raw_proba = float(self.calibrator.predict([raw_proba])[0])
        return float(raw_proba)

    def predict(self, text: str, threshold: float = None) -> int:
        if threshold is None:
            threshold = self.threshold
        return 1 if self.predict_proba(text) >= threshold else 0

    def __del__(self):
        for attr in ("encoder", "ppl_model"):
            if hasattr(self, attr):
                try:
                    delattr(self, attr)
                except Exception:
                    pass
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()