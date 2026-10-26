import os
import re
import logging
import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
MAX_SEQ_LENGTH = 256
CHUNK_SIZE_CHARS = 500
CHUNK_OVERLAP_CHARS = 100

class EmbedderModel:
    """
    Genuine 384-dimensional semantic text embedder using the pretrained
    sentence-transformers/all-MiniLM-L6-v2 model.

    Supports both standard PyTorch SentenceTransformer runtime and official
    ONNX Runtime fallback (ensuring full compatibility across Linux and Windows
    environments where native PyTorch DLLs might be constrained by security policies).
    All embeddings are L2-normalized unit vectors so cosine similarity can be
    computed directly via dot product or pgvector's cosine distance operator.
    """

    def __init__(self):
        self.model_name = MODEL_NAME
        self.dimension = EMBEDDING_DIM
        self.backend = None
        self.device = "cpu"
        self._st_model = None
        self._onnx_session = None
        self._tokenizer = None

        logger.info("Initializing EmbedderModel for %s ...", MODEL_NAME)

        # 1. Try PyTorch SentenceTransformer first
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Attempting to load with sentence-transformers (PyTorch)...")
            self._st_model = SentenceTransformer(MODEL_NAME)
            self.backend = "sentence-transformers"
            self.device = str(getattr(self._st_model, "device", "cpu"))
            logger.info("Successfully loaded %s via sentence-transformers on device: %s", MODEL_NAME, self.device)
            return
        except Exception as e:
            logger.info("SentenceTransformer/PyTorch could not be loaded directly (%s). Trying ONNX Runtime...", e)

        # 2. Try official ONNX Runtime fallback
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer

            logger.info("Downloading/loading official ONNX weights for %s ...", MODEL_NAME)
            model_path = hf_hub_download(repo_id=MODEL_NAME, filename="onnx/model.onnx")
            tok_path = hf_hub_download(repo_id=MODEL_NAME, filename="tokenizer.json")

            self._tokenizer = Tokenizer.from_file(tok_path)
            self._tokenizer.enable_truncation(max_length=MAX_SEQ_LENGTH)
            self._tokenizer.enable_padding(length=MAX_SEQ_LENGTH)

            # Configure ONNX session
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._onnx_session = ort.InferenceSession(model_path, sess_options)
            self.backend = "onnxruntime"
            self.device = ort.get_device()
            logger.info("Successfully loaded %s via ONNX Runtime on device: %s", MODEL_NAME, self.device)
            return
        except Exception as e:
            logger.error("Failed to load %s via ONNX Runtime: %s", MODEL_NAME, e)
            raise RuntimeError(
                f"Failed to load embedding model '{MODEL_NAME}' via SentenceTransformer and ONNX Runtime: {e}"
            )

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Cleans input text without stripping technical terms, keywords, or punctuation
        essential to semantic meaning. Removes null bytes, unprintable control characters,
        and collapses excessive whitespace.
        """
        if not text:
            return ""
        # Remove null characters which cause PostgreSQL errors
        text = text.replace("\x00", " ")
        # Remove non-printable control characters, retaining standard whitespace
        text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
        # Collapse multiple spaces and blank lines
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _chunk_text(self, text: str) -> list[str]:
        """
        Splits oversized text into overlapping chunks so that key skills and sections
        distributed across long documents (like comprehensive resumes) are preserved.
        """
        if len(text) <= 1200:
            return [text]

        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + CHUNK_SIZE_CHARS, text_len)
            # Try to break at a newline or space rather than mid-word
            if end < text_len:
                last_space = text.rfind(" ", start, end)
                if last_space > start + (CHUNK_SIZE_CHARS // 2):
                    end = last_space
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - CHUNK_OVERLAP_CHARS if end < text_len else text_len

        return chunks if chunks else [text]

    def _encode_raw(self, texts: list[str]) -> np.ndarray:
        """
        Encodes a batch of cleaned text strings into raw 384-dimensional embeddings.
        """
        if self.backend == "sentence-transformers":
            return self._st_model.encode(texts, normalize_embeddings=False, batch_size=32)

        # ONNX Runtime inference
        input_ids = []
        attention_masks = []
        token_type_ids = []

        for t in texts:
            encoded = self._tokenizer.encode(t)
            input_ids.append(encoded.ids)
            attention_masks.append(encoded.attention_mask)
            token_type_ids.append(encoded.type_ids)

        inps = {
            "input_ids": np.array(input_ids, dtype=np.int64),
            "attention_mask": np.array(attention_masks, dtype=np.int64),
            "token_type_ids": np.array(token_type_ids, dtype=np.int64),
        }

        # Run model
        last_hidden_state = self._onnx_session.run(None, inps)[0]

        # Mean pooling with attention mask
        mask = np.broadcast_to(
            np.expand_dims(inps["attention_mask"], -1),
            last_hidden_state.shape
        ).astype(float)

        sum_embeddings = np.sum(last_hidden_state * mask, axis=1)
        sum_mask = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask

    def embed(self, text: str) -> list[float]:
        """
        Generates a 384-dimensional L2-normalized embedding for a single text.
        For documents exceeding token limits, uses a chunked mean-pooling strategy.
        Raises ValueError on empty or invalid text.
        """
        cleaned = self.clean_text(text)
        if not cleaned:
            raise ValueError("Input text cannot be empty or whitespace only.")

        chunks = self._chunk_text(cleaned)
        if len(chunks) == 1:
            raw_vec = self._encode_raw([chunks[0]])[0]
        else:
            # Multi-chunk mean pooling
            chunk_vecs = self._encode_raw(chunks)
            raw_vec = np.mean(chunk_vecs, axis=0)

        # L2-normalize
        norm = np.linalg.norm(raw_vec)
        if norm > 0:
            norm_vec = raw_vec / norm
        else:
            norm_vec = raw_vec

        return [round(float(x), 6) for x in norm_vec]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generates 384-dimensional L2-normalized embeddings for a list of texts.
        Raises ValueError if texts is empty or contains empty strings.
        """
        if not texts:
            raise ValueError("Texts list cannot be empty.")

        cleaned_texts = [self.clean_text(t) for t in texts]
        for idx, ct in enumerate(cleaned_texts):
            if not ct:
                raise ValueError(f"Text at index {idx} is empty or whitespace only.")

        # If all texts are short, encode in one batch
        if all(len(t) <= 1200 for t in cleaned_texts):
            raw_vecs = self._encode_raw(cleaned_texts)
            norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            norm_vecs = raw_vecs / norms
            return [[round(float(x), 6) for x in row] for row in norm_vecs]

        # Otherwise encode individually to respect chunking
        return [self.embed(t) for t in cleaned_texts]
