# Universal Model Integration Standard — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a shared SDK with base class hierarchy, manifest-driven configuration, and `deepsafe serve` entrypoint so that adding a new model requires only a `model.yaml` + one Python file with domain logic.

**Architecture:** An installable Python package (`deepsafe_sdk`) at `/sdk/` provides abstract base classes (`ImageModel`, `VideoModel`, `AudioModel`), a manifest parser, weight management, and a universal FastAPI server entrypoint. Each model container copies and pip-installs the SDK, then runs `deepsafe serve --manifest model.yaml`. All 3 existing models are migrated to use the SDK.

**Tech Stack:** Python 3.9, FastAPI, Pydantic, PyTorch, Pillow, OpenCV, PyYAML

---

## File Structure

### New files (SDK):
- `sdk/setup.py` — Package metadata and entry point registration
- `sdk/deepsafe_sdk/__init__.py` — Public API exports
- `sdk/deepsafe_sdk/types.py` — `PredictionResult` Pydantic model
- `sdk/deepsafe_sdk/manifest.py` — `ModelManifest` parser/validator
- `sdk/deepsafe_sdk/weights.py` — Weight download + SHA256 verification
- `sdk/deepsafe_sdk/base.py` — `DeepSafeModel` abstract base class (lazy loading, locking, timeout)
- `sdk/deepsafe_sdk/image.py` — `ImageModel` with image decode utilities
- `sdk/deepsafe_sdk/video.py` — `VideoModel` with frame extraction + face detection utilities
- `sdk/deepsafe_sdk/audio.py` — `AudioModel` with audio decode utilities
- `sdk/deepsafe_sdk/server.py` — FastAPI app factory + `deepsafe` CLI entry point

### New files (tests):
- `sdk/tests/__init__.py`
- `sdk/tests/test_types.py`
- `sdk/tests/test_manifest.py`
- `sdk/tests/test_weights.py`
- `sdk/tests/test_base.py`
- `sdk/tests/test_image.py`
- `sdk/tests/test_video.py`
- `sdk/tests/test_audio.py`
- `sdk/tests/test_server.py`

### New files (model manifests):
- `models/image/npr_deepfakedetection/model.yaml`
- `models/image/npr_deepfakedetection/detector.py`
- `models/image/universalfakedetect/model.yaml`
- `models/image/universalfakedetect/detector.py`
- `models/video/cross_efficient_vit/model.yaml`
- `models/video/cross_efficient_vit/detector.py`

### Modified files:
- `models/image/npr_deepfakedetection/Dockerfile` — Use SDK + `deepsafe serve`
- `models/image/universalfakedetect/Dockerfile` — Use SDK + `deepsafe serve`
- `models/video/cross_efficient_vit/Dockerfile` — Use SDK + `deepsafe serve`
- `docker-compose.yml` — Update build contexts to include SDK
- `docs/adding-a-model.md` — New contributor guide (replaces `docs/INTEGRATING_NEW_MODELS.md`)

### Deleted files:
- `models/image/npr_deepfakedetection/app.py` — Replaced by `detector.py` + SDK
- `models/image/universalfakedetect/app.py` — Replaced by `detector.py` + SDK
- `models/video/cross_efficient_vit/app.py` — Replaced by `detector.py` + SDK
- `docs/INTEGRATING_NEW_MODELS.md` — Replaced by `docs/adding-a-model.md`

---

## Task 1: SDK Package Scaffolding + PredictionResult Type

**Files:**
- Create: `sdk/setup.py`
- Create: `sdk/deepsafe_sdk/__init__.py`
- Create: `sdk/deepsafe_sdk/types.py`
- Create: `sdk/tests/__init__.py`
- Create: `sdk/tests/test_types.py`

- [ ] **Step 1: Write failing test for PredictionResult**

Create `sdk/tests/test_types.py`:

```python
from deepsafe_sdk.types import PredictionResult


def test_prediction_result_fields():
    result = PredictionResult(
        model="test_model",
        probability=0.85,
        prediction=1,
        class_name="fake",
        inference_time=0.123,
    )
    assert result.model == "test_model"
    assert result.probability == 0.85
    assert result.prediction == 1
    assert result.class_name == "fake"
    assert result.inference_time == 0.123


def test_prediction_result_serialization():
    result = PredictionResult(
        model="test_model",
        probability=0.3,
        prediction=0,
        class_name="real",
        inference_time=0.05,
    )
    d = result.model_dump()
    assert d["model"] == "test_model"
    assert d["probability"] == 0.3
    assert d["prediction"] == 0
    assert d["class_name"] == "real"
    # Verify it also serializes with "class" alias for API compat
    d_alias = result.model_dump(by_alias=True)
    assert d_alias["class"] == "real"
```

Create empty `sdk/tests/__init__.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/sidd/Desktop/Personal/Projects/Open_Source_DeepSafe/DeepSafe && python -m pytest sdk/tests/test_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deepsafe_sdk'`

- [ ] **Step 3: Create SDK package structure and implement PredictionResult**

Create `sdk/setup.py`:

```python
from setuptools import setup, find_packages

setup(
    name="deepsafe-sdk",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "deepsafe=deepsafe_sdk.server:cli",
        ],
    },
    python_requires=">=3.9",
)
```

Create `sdk/deepsafe_sdk/__init__.py`:

```python
from deepsafe_sdk.types import PredictionResult

__all__ = ["PredictionResult"]
```

Create `sdk/deepsafe_sdk/types.py`:

```python
from pydantic import BaseModel, Field


class PredictionResult(BaseModel):
    model: str
    probability: float = Field(ge=0.0, le=1.0)
    prediction: int = Field(ge=0, le=1)
    class_name: str = Field(serialization_alias="class")
    inference_time: float
```

- [ ] **Step 4: Install SDK in editable mode and run test**

Run: `cd /Users/sidd/Desktop/Personal/Projects/Open_Source_DeepSafe/DeepSafe && pip install -e sdk/ && python -m pytest sdk/tests/test_types.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add sdk/setup.py sdk/deepsafe_sdk/__init__.py sdk/deepsafe_sdk/types.py sdk/tests/__init__.py sdk/tests/test_types.py
git commit -m "feat(sdk): scaffold SDK package with PredictionResult type"
```

---

## Task 2: Model Manifest Parser

**Files:**
- Create: `sdk/deepsafe_sdk/manifest.py`
- Create: `sdk/tests/test_manifest.py`

- [ ] **Step 1: Write failing tests for manifest parsing**

Create `sdk/tests/test_manifest.py`:

```python
import os
import pytest
import yaml
from deepsafe_sdk.manifest import ModelManifest, load_manifest


def _write_manifest(tmp_path, data):
    path = os.path.join(tmp_path, "model.yaml")
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


def test_load_valid_manifest(tmp_path):
    data = {
        "name": "test_model",
        "version": "1.0.0",
        "media_type": "image",
        "model_class": "detector.TestDetector",
        "port": 5008,
        "weights": [
            {
                "url": "https://example.com/model.pth",
                "path": "weights/model.pth",
                "sha256": "abc123",
            }
        ],
        "environment": {"PRELOAD_MODEL": "false", "MODEL_TIMEOUT": "600"},
        "dependencies": ["timm==0.9.2"],
    }
    path = _write_manifest(str(tmp_path), data)
    manifest = load_manifest(path)
    assert manifest.name == "test_model"
    assert manifest.media_type == "image"
    assert manifest.port == 5008
    assert len(manifest.weights) == 1
    assert manifest.weights[0].sha256 == "abc123"


def test_load_minimal_manifest(tmp_path):
    data = {
        "name": "minimal_model",
        "media_type": "video",
        "model_class": "detector.Detector",
        "port": 7002,
    }
    path = _write_manifest(str(tmp_path), data)
    manifest = load_manifest(path)
    assert manifest.name == "minimal_model"
    assert manifest.weights == []
    assert manifest.environment == {}
    assert manifest.dependencies == []


def test_invalid_media_type(tmp_path):
    data = {
        "name": "bad",
        "media_type": "pdf",
        "model_class": "d.D",
        "port": 5000,
    }
    path = _write_manifest(str(tmp_path), data)
    with pytest.raises(ValueError, match="media_type"):
        load_manifest(path)


def test_missing_required_field(tmp_path):
    data = {"name": "incomplete"}
    path = _write_manifest(str(tmp_path), data)
    with pytest.raises((ValueError, KeyError)):
        load_manifest(path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest sdk/tests/test_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deepsafe_sdk.manifest'`

- [ ] **Step 3: Implement manifest parser**

Create `sdk/deepsafe_sdk/manifest.py`:

```python
from dataclasses import dataclass, field
from typing import List, Dict
import yaml


VALID_MEDIA_TYPES = {"image", "video", "audio"}


@dataclass
class WeightEntry:
    url: str
    path: str
    sha256: str


@dataclass
class ModelManifest:
    name: str
    media_type: str
    model_class: str
    port: int
    version: str = "0.0.0"
    weights: List[WeightEntry] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)


def load_manifest(path: str) -> ModelManifest:
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    for required in ("name", "media_type", "model_class", "port"):
        if required not in data:
            raise ValueError(f"Missing required field: {required}")

    if data["media_type"] not in VALID_MEDIA_TYPES:
        raise ValueError(
            f"Invalid media_type '{data['media_type']}'. "
            f"Must be one of: {VALID_MEDIA_TYPES}"
        )

    weights = [
        WeightEntry(url=w["url"], path=w["path"], sha256=w["sha256"])
        for w in data.get("weights", [])
    ]

    return ModelManifest(
        name=data["name"],
        version=data.get("version", "0.0.0"),
        media_type=data["media_type"],
        model_class=data["model_class"],
        port=data["port"],
        weights=weights,
        environment=data.get("environment", {}),
        dependencies=data.get("dependencies", []),
    )
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest sdk/tests/test_manifest.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Update __init__.py exports and commit**

Update `sdk/deepsafe_sdk/__init__.py` to add:
```python
from deepsafe_sdk.manifest import ModelManifest, load_manifest
```

```bash
git add sdk/deepsafe_sdk/manifest.py sdk/tests/test_manifest.py sdk/deepsafe_sdk/__init__.py
git commit -m "feat(sdk): add model manifest parser with validation"
```

---

## Task 3: Weight Manager

**Files:**
- Create: `sdk/deepsafe_sdk/weights.py`
- Create: `sdk/tests/test_weights.py`

- [ ] **Step 1: Write failing tests for weight management**

Create `sdk/tests/test_weights.py`:

```python
import os
import hashlib
import pytest
from unittest.mock import patch
from deepsafe_sdk.manifest import WeightEntry
from deepsafe_sdk.weights import ensure_weights, compute_sha256


def test_compute_sha256(tmp_path):
    filepath = os.path.join(str(tmp_path), "test.bin")
    content = b"test content for hashing"
    with open(filepath, "wb") as f:
        f.write(content)
    expected = hashlib.sha256(content).hexdigest()
    assert compute_sha256(filepath) == expected


def test_ensure_weights_skips_existing(tmp_path):
    content = b"fake model weights"
    sha = hashlib.sha256(content).hexdigest()
    filepath = os.path.join(str(tmp_path), "model.pth")
    with open(filepath, "wb") as f:
        f.write(content)

    entry = WeightEntry(
        url="https://example.com/model.pth", path="model.pth", sha256=sha
    )

    with patch("deepsafe_sdk.weights.urllib.request.urlretrieve") as mock_dl:
        ensure_weights([entry], str(tmp_path))
        mock_dl.assert_not_called()


def test_ensure_weights_downloads_missing(tmp_path):
    content = b"downloaded weights"
    sha = hashlib.sha256(content).hexdigest()

    entry = WeightEntry(
        url="https://example.com/model.pth", path="model.pth", sha256=sha
    )

    def fake_download(url, dest):
        with open(dest, "wb") as f:
            f.write(content)

    with patch(
        "deepsafe_sdk.weights.urllib.request.urlretrieve",
        side_effect=fake_download,
    ):
        ensure_weights([entry], str(tmp_path))

    filepath = os.path.join(str(tmp_path), "model.pth")
    assert os.path.exists(filepath)


def test_ensure_weights_checksum_mismatch(tmp_path):
    entry = WeightEntry(
        url="https://example.com/model.pth",
        path="model.pth",
        sha256="wrong_hash",
    )

    def fake_download(url, dest):
        with open(dest, "wb") as f:
            f.write(b"some data")

    with patch(
        "deepsafe_sdk.weights.urllib.request.urlretrieve",
        side_effect=fake_download,
    ):
        with pytest.raises(RuntimeError, match="Checksum mismatch"):
            ensure_weights([entry], str(tmp_path))


def test_ensure_weights_skips_empty_url(tmp_path):
    entry = WeightEntry(url="", path="weights/model.pth", sha256="")
    # Should not raise, should not download
    with patch("deepsafe_sdk.weights.urllib.request.urlretrieve") as mock_dl:
        ensure_weights([entry], str(tmp_path))
        mock_dl.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest sdk/tests/test_weights.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deepsafe_sdk.weights'`

- [ ] **Step 3: Implement weight manager**

Create `sdk/deepsafe_sdk/weights.py`:

```python
import hashlib
import logging
import os
import urllib.request
from typing import List

from deepsafe_sdk.manifest import WeightEntry

logger = logging.getLogger(__name__)


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_weights(weights: List[WeightEntry], model_dir: str) -> None:
    for entry in weights:
        if not entry.url:
            logger.info(
                f"Weight entry '{entry.path}' has no URL. "
                f"Skipping download (assumed present from Docker build)."
            )
            continue

        filepath = os.path.join(model_dir, entry.path)

        if os.path.exists(filepath):
            if entry.sha256:
                actual_sha = compute_sha256(filepath)
                if actual_sha == entry.sha256:
                    logger.info(
                        f"Weight file '{entry.path}' exists "
                        f"with correct checksum. Skipping."
                    )
                    continue
                else:
                    logger.warning(
                        f"Weight file '{entry.path}' checksum mismatch "
                        f"(expected {entry.sha256}, got {actual_sha}). "
                        f"Re-downloading."
                    )
            else:
                logger.info(
                    f"Weight file '{entry.path}' exists "
                    f"(no checksum to verify). Skipping."
                )
                continue

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        logger.info(
            f"Downloading weight file '{entry.path}' from {entry.url}..."
        )
        urllib.request.urlretrieve(entry.url, filepath)

        if entry.sha256:
            actual_sha = compute_sha256(filepath)
            if actual_sha != entry.sha256:
                os.remove(filepath)
                raise RuntimeError(
                    f"Checksum mismatch for '{entry.path}': "
                    f"expected {entry.sha256}, got {actual_sha}"
                )

        logger.info(f"Weight file '{entry.path}' ready.")
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest sdk/tests/test_weights.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Update exports and commit**

Update `sdk/deepsafe_sdk/__init__.py` to add:
```python
from deepsafe_sdk.weights import ensure_weights
```

```bash
git add sdk/deepsafe_sdk/weights.py sdk/tests/test_weights.py sdk/deepsafe_sdk/__init__.py
git commit -m "feat(sdk): add weight download and checksum verification"
```

---

## Task 4: DeepSafeModel Abstract Base Class

**Files:**
- Create: `sdk/deepsafe_sdk/base.py`
- Create: `sdk/tests/test_base.py`

- [ ] **Step 1: Write failing tests for base model lifecycle**

Create `sdk/tests/test_base.py`:

```python
import pytest
from deepsafe_sdk.base import DeepSafeModel
from deepsafe_sdk.types import PredictionResult


class FakeModel(DeepSafeModel):
    def __init__(self, name="fake", model_dir="/tmp"):
        super().__init__(name=name, model_dir=model_dir)
        self.load_count = 0

    def load(self):
        self.load_count += 1
        self.model = "loaded"

    def predict(self, input_data: str, threshold: float) -> PredictionResult:
        return self.make_result(probability=0.75, threshold=threshold)


def test_lazy_loading():
    m = FakeModel()
    assert not m.is_loaded
    result = m.safe_predict("data", 0.5)
    assert m.is_loaded
    assert m.load_count == 1
    assert result.probability == 0.75


def test_make_result():
    m = FakeModel(name="test")
    result = m.make_result(probability=0.8, threshold=0.5)
    assert result.prediction == 1
    assert result.class_name == "fake"
    assert result.model == "test"

    result_real = m.make_result(probability=0.3, threshold=0.5)
    assert result_real.prediction == 0
    assert result_real.class_name == "real"


def test_unload():
    m = FakeModel()
    m.safe_predict("data", 0.5)
    assert m.is_loaded
    m.unload()
    assert not m.is_loaded


def test_double_load_is_noop():
    m = FakeModel()
    m.safe_predict("data", 0.5)
    m.safe_predict("data", 0.5)
    assert m.load_count == 1


def test_weights_path():
    m = FakeModel(name="test", model_dir="/app/model")
    assert m.weights_path("weights/NPR.pth") == "/app/model/weights/NPR.pth"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest sdk/tests/test_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deepsafe_sdk.base'`

- [ ] **Step 3: Implement DeepSafeModel base class**

Create `sdk/deepsafe_sdk/base.py`:

```python
import gc
import logging
import os
import threading
import time
from abc import ABC, abstractmethod

from deepsafe_sdk.types import PredictionResult

logger = logging.getLogger(__name__)


class DeepSafeModel(ABC):
    def __init__(self, name: str = "", model_dir: str = ""):
        self._model_name = name
        self._model_dir = model_dir
        self._model = None
        self._lock = threading.Lock()
        self._last_used: float = 0.0

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory. Set self.model."""

    @abstractmethod
    def predict(self, input_data: str, threshold: float) -> PredictionResult:
        """Run inference on base64-encoded input data."""

    def unload(self) -> None:
        with self._lock:
            if self._model is not None:
                logger.info(f"Unloading model '{self._model_name}'.")
                del self._model
                self._model = None
                gc.collect()

    def safe_predict(
        self, input_data: str, threshold: float
    ) -> PredictionResult:
        self._ensure_loaded()
        start = time.time()
        result = self.predict(input_data, threshold)
        if result.inference_time == 0:
            result.inference_time = time.time() - start
        return result

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            self._last_used = time.time()
            return
        with self._lock:
            if self._model is not None:
                self._last_used = time.time()
                return
            logger.info(f"Loading model '{self._model_name}'...")
            self.load()
            self._last_used = time.time()
            logger.info(f"Model '{self._model_name}' loaded.")

    def check_idle_unload(self, timeout: int) -> None:
        if self._model is None:
            return
        if time.time() - self._last_used > timeout:
            self.unload()

    def weights_path(self, relative_path: str) -> str:
        return os.path.join(self._model_dir, relative_path)

    def make_result(
        self, probability: float, threshold: float
    ) -> PredictionResult:
        prediction = 1 if probability >= threshold else 0
        return PredictionResult(
            model=self._model_name,
            probability=probability,
            prediction=prediction,
            class_name="fake" if prediction == 1 else "real",
            inference_time=0.0,
        )
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest sdk/tests/test_base.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Update exports and commit**

Update `sdk/deepsafe_sdk/__init__.py` to add:
```python
from deepsafe_sdk.base import DeepSafeModel
```

```bash
git add sdk/deepsafe_sdk/base.py sdk/tests/test_base.py sdk/deepsafe_sdk/__init__.py
git commit -m "feat(sdk): add DeepSafeModel abstract base with lifecycle management"
```

---

## Task 5: ImageModel Base Class

**Files:**
- Create: `sdk/deepsafe_sdk/image.py`
- Create: `sdk/tests/test_image.py`

- [ ] **Step 1: Write failing tests for ImageModel utilities**

Create `sdk/tests/test_image.py`:

```python
import base64
import io
import pytest
from PIL import Image
from deepsafe_sdk.image import ImageModel
from deepsafe_sdk.types import PredictionResult


class FakeImageModel(ImageModel):
    def load(self):
        self.model = "loaded"

    def predict(self, input_data: str, threshold: float) -> PredictionResult:
        image = self.decode_image(input_data)
        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"
        return self.make_result(probability=0.9, threshold=threshold)


def _make_test_image_b64():
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def test_decode_image():
    m = FakeImageModel(name="test", model_dir="/tmp")
    b64 = _make_test_image_b64()
    image = m.decode_image(b64)
    assert isinstance(image, Image.Image)
    assert image.size == (100, 100)
    assert image.mode == "RGB"


def test_decode_image_invalid():
    m = FakeImageModel(name="test", model_dir="/tmp")
    with pytest.raises(ValueError, match="decode"):
        m.decode_image("not_valid_base64!!!")


def test_predict_with_image():
    m = FakeImageModel(name="test", model_dir="/tmp")
    b64 = _make_test_image_b64()
    result = m.safe_predict(b64, 0.5)
    assert result.prediction == 1
    assert result.class_name == "fake"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest sdk/tests/test_image.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deepsafe_sdk.image'`

- [ ] **Step 3: Implement ImageModel**

Create `sdk/deepsafe_sdk/image.py`:

```python
import base64
import io
import logging

from PIL import Image, ImageFile

from deepsafe_sdk.base import DeepSafeModel

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)


class ImageModel(DeepSafeModel):
    def decode_image(self, base64_data: str) -> Image.Image:
        try:
            image_bytes = base64.b64decode(base64_data)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            return image
        except Exception as e:
            raise ValueError(f"Failed to decode image: {e}") from e
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest sdk/tests/test_image.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Update exports and commit**

Update `sdk/deepsafe_sdk/__init__.py` to add:
```python
from deepsafe_sdk.image import ImageModel
```

```bash
git add sdk/deepsafe_sdk/image.py sdk/tests/test_image.py sdk/deepsafe_sdk/__init__.py
git commit -m "feat(sdk): add ImageModel base class with image decoding"
```

---

## Task 6: VideoModel Base Class

**Files:**
- Create: `sdk/deepsafe_sdk/video.py`
- Create: `sdk/tests/test_video.py`

- [ ] **Step 1: Write failing tests for VideoModel utilities**

Create `sdk/tests/test_video.py`:

```python
import base64
import os
import tempfile
import numpy as np
import pytest
from deepsafe_sdk.video import VideoModel
from deepsafe_sdk.types import PredictionResult


class FakeVideoModel(VideoModel):
    def load(self):
        self.model = "loaded"

    def predict(self, input_data: str, threshold: float) -> PredictionResult:
        return self.make_result(probability=0.6, threshold=threshold)


def test_extract_frames_returns_list():
    m = FakeVideoModel(name="test", model_dir="/tmp")
    try:
        import cv2
    except ImportError:
        pytest.skip("cv2 not available")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_path = f.name
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, 25.0, (64, 64))
        for _ in range(30):
            frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

        with open(tmp_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

        frames = m.extract_frames(video_b64, num_frames=5)
        assert isinstance(frames, list)
        assert len(frames) == 5
        assert frames[0].shape[2] == 3  # RGB
    finally:
        os.unlink(tmp_path)


def test_frames_per_video_default():
    m = FakeVideoModel(name="test", model_dir="/tmp")
    assert m.frames_per_video == 15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest sdk/tests/test_video.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deepsafe_sdk.video'`

- [ ] **Step 3: Implement VideoModel**

Create `sdk/deepsafe_sdk/video.py`:

```python
import base64
import logging
import os
import tempfile
from typing import List

import cv2
import numpy as np

from deepsafe_sdk.base import DeepSafeModel

logger = logging.getLogger(__name__)


class VideoModel(DeepSafeModel):
    frames_per_video: int = 15

    def extract_frames(
        self, base64_data: str, num_frames: int = 0
    ) -> List[np.ndarray]:
        if num_frames <= 0:
            num_frames = self.frames_per_video

        video_bytes = base64.b64decode(base64_data)
        tmp_path = os.path.join(
            tempfile.gettempdir(),
            f"deepsafe_{os.urandom(8).hex()}.mp4",
        )
        try:
            with open(tmp_path, "wb") as f:
                f.write(video_bytes)

            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video file: {tmp_path}")
                return []

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                cap.release()
                return []

            indices = np.linspace(
                0, total_frames - 1, num_frames, dtype=int
            )
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            return frames
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest sdk/tests/test_video.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Update exports and commit**

Update `sdk/deepsafe_sdk/__init__.py` to add:
```python
from deepsafe_sdk.video import VideoModel
```

```bash
git add sdk/deepsafe_sdk/video.py sdk/tests/test_video.py sdk/deepsafe_sdk/__init__.py
git commit -m "feat(sdk): add VideoModel base class with frame extraction"
```

---

## Task 7: AudioModel Base Class

**Files:**
- Create: `sdk/deepsafe_sdk/audio.py`
- Create: `sdk/tests/test_audio.py`

- [ ] **Step 1: Write failing tests for AudioModel utilities**

Create `sdk/tests/test_audio.py`:

```python
import base64
import io
import struct
import wave
import pytest
from deepsafe_sdk.audio import AudioModel
from deepsafe_sdk.types import PredictionResult


class FakeAudioModel(AudioModel):
    def load(self):
        self.model = "loaded"

    def predict(self, input_data: str, threshold: float) -> PredictionResult:
        waveform, sr = self.decode_audio(input_data)
        return self.make_result(probability=0.4, threshold=threshold)


def _make_wav_b64(duration_s=0.1, sample_rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        num_samples = int(duration_s * sample_rate)
        samples = struct.pack(f"<{num_samples}h", *([0] * num_samples))
        w.writeframes(samples)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def test_decode_audio():
    m = FakeAudioModel(name="test", model_dir="/tmp")
    b64 = _make_wav_b64()
    waveform, sr = m.decode_audio(b64)
    assert sr == 16000
    assert len(waveform) > 0


def test_decode_audio_invalid():
    m = FakeAudioModel(name="test", model_dir="/tmp")
    with pytest.raises(ValueError, match="decode"):
        m.decode_audio("not_valid_audio!!!")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest sdk/tests/test_audio.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deepsafe_sdk.audio'`

- [ ] **Step 3: Implement AudioModel**

Create `sdk/deepsafe_sdk/audio.py`:

```python
import base64
import io
import logging
import wave
from typing import Tuple

import numpy as np

from deepsafe_sdk.base import DeepSafeModel

logger = logging.getLogger(__name__)


class AudioModel(DeepSafeModel):
    def decode_audio(self, base64_data: str) -> Tuple[np.ndarray, int]:
        try:
            audio_bytes = base64.b64decode(base64_data)
            buf = io.BytesIO(audio_bytes)
            with wave.open(buf, "rb") as w:
                sample_rate = w.getframerate()
                n_frames = w.getnframes()
                raw = w.readframes(n_frames)
                sample_width = w.getsampwidth()

            if sample_width == 2:
                dtype = np.int16
            elif sample_width == 4:
                dtype = np.int32
            else:
                dtype = np.uint8

            waveform = np.frombuffer(raw, dtype=dtype).astype(np.float32)
            max_val = float(np.iinfo(dtype).max)
            waveform = waveform / max_val
            return waveform, sample_rate
        except Exception as e:
            raise ValueError(f"Failed to decode audio: {e}") from e
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest sdk/tests/test_audio.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Update exports and commit**

Update `sdk/deepsafe_sdk/__init__.py` to add:
```python
from deepsafe_sdk.audio import AudioModel
```

```bash
git add sdk/deepsafe_sdk/audio.py sdk/tests/test_audio.py sdk/deepsafe_sdk/__init__.py
git commit -m "feat(sdk): add AudioModel base class with WAV decoding"
```

---

## Task 8: FastAPI Server Factory + `deepsafe serve` CLI

**Files:**
- Create: `sdk/deepsafe_sdk/server.py`
- Create: `sdk/tests/test_server.py`

- [ ] **Step 1: Write failing tests for server factory**

Create `sdk/tests/test_server.py`:

```python
import os
import base64
import io
import pytest
import yaml
from PIL import Image
from fastapi.testclient import TestClient
from deepsafe_sdk.server import create_app
from deepsafe_sdk.manifest import load_manifest


def _make_manifest_dir(tmp_path):
    manifest_data = {
        "name": "stub_detector",
        "version": "1.0.0",
        "media_type": "image",
        "model_class": "stub.StubDetector",
        "port": 5099,
    }
    manifest_path = os.path.join(str(tmp_path), "model.yaml")
    with open(manifest_path, "w") as f:
        yaml.dump(manifest_data, f)

    stub_path = os.path.join(str(tmp_path), "stub.py")
    with open(stub_path, "w") as f:
        f.write(
            "from deepsafe_sdk.image import ImageModel\n"
            "from deepsafe_sdk.types import PredictionResult\n"
            "class StubDetector(ImageModel):\n"
            "    def load(self):\n"
            "        self.model = 'loaded'\n"
            "    def predict(self, input_data, threshold):\n"
            "        return self.make_result("
            "probability=0.7, threshold=threshold)\n"
        )
    return manifest_path


def test_create_app_info_endpoint(tmp_path):
    manifest_path = _make_manifest_dir(tmp_path)
    manifest = load_manifest(manifest_path)
    app = create_app(manifest, str(tmp_path))
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "stub_detector"
    assert data["media_type"] == "image"


def test_create_app_health_endpoint(tmp_path):
    manifest_path = _make_manifest_dir(tmp_path)
    manifest = load_manifest(manifest_path)
    app = create_app(manifest, str(tmp_path))
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["model_name"] == "stub_detector"


def test_create_app_predict_endpoint(tmp_path):
    manifest_path = _make_manifest_dir(tmp_path)
    manifest = load_manifest(manifest_path)
    app = create_app(manifest, str(tmp_path))
    client = TestClient(app)

    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    resp = client.post(
        "/predict", json={"image_data": b64, "threshold": 0.5}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "stub_detector"
    assert data["probability"] == 0.7
    assert data["prediction"] == 1
    assert data["class"] == "fake"
    assert "inference_time" in data


def test_create_app_unload_endpoint(tmp_path):
    manifest_path = _make_manifest_dir(tmp_path)
    manifest = load_manifest(manifest_path)
    app = create_app(manifest, str(tmp_path))
    client = TestClient(app)

    resp = client.post("/unload")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("not_loaded", "unloaded")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest sdk/tests/test_server.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deepsafe_sdk.server'`

- [ ] **Step 3: Implement server factory and CLI**

Create `sdk/deepsafe_sdk/server.py`:

```python
import importlib
import logging
import os
import sys
import threading
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from deepsafe_sdk.base import DeepSafeModel
from deepsafe_sdk.manifest import ModelManifest, load_manifest
from deepsafe_sdk.weights import ensure_weights

logger = logging.getLogger(__name__)


class ImageInput(BaseModel):
    image_data: str
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0)


class VideoInput(BaseModel):
    video_data: str
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0)


class AudioInput(BaseModel):
    audio_data: str
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0)


INPUT_MODELS = {
    "image": ImageInput,
    "video": VideoInput,
    "audio": AudioInput,
}

DATA_FIELD = {
    "image": "image_data",
    "video": "video_data",
    "audio": "audio_data",
}


def _import_model_class(
    model_class_path: str, model_dir: str
) -> type:
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    module_path, class_name = model_class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def create_app(
    manifest: ModelManifest, model_dir: str
) -> FastAPI:
    model_cls = _import_model_class(
        manifest.model_class, model_dir
    )
    instance: DeepSafeModel = model_cls(
        name=manifest.name, model_dir=model_dir
    )

    preload = manifest.environment.get(
        "PRELOAD_MODEL",
        os.environ.get("PRELOAD_MODEL", "false"),
    ).lower() == "true"
    model_timeout = int(
        manifest.environment.get(
            "MODEL_TIMEOUT",
            os.environ.get("MODEL_TIMEOUT", "600"),
        )
    )

    InputModel = INPUT_MODELS[manifest.media_type]
    data_field = DATA_FIELD[manifest.media_type]

    app = FastAPI(
        title=f"DeepSafe Model: {manifest.name}",
        version=manifest.version,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def info():
        return {
            "name": manifest.name,
            "version": manifest.version,
            "media_type": manifest.media_type,
            "model_loaded": instance.is_loaded,
        }

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "model_name": manifest.name,
            "model_loaded": instance.is_loaded,
        }

    @app.post("/predict")
    async def predict(input_data: InputModel):
        try:
            media_data = getattr(input_data, data_field)
            result = instance.safe_predict(
                media_data, input_data.threshold
            )
            return result.model_dump(by_alias=True)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            logger.exception(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/unload")
    async def unload():
        if not instance.is_loaded:
            return {
                "status": "not_loaded",
                "message": f"{manifest.name} is not loaded.",
            }
        instance.unload()
        return {
            "status": "unloaded",
            "message": f"{manifest.name} unloaded.",
        }

    @app.on_event("startup")
    async def startup():
        if preload:
            logger.info(f"Preloading model '{manifest.name}'...")
            try:
                instance._ensure_loaded()
            except Exception as e:
                logger.error(
                    f"Preload failed: {e}", exc_info=True
                )

        if model_timeout > 0:
            def idle_check():
                instance.check_idle_unload(model_timeout)
                if instance.is_loaded:
                    t = threading.Timer(
                        model_timeout / 2.0, idle_check
                    )
                    t.daemon = True
                    t.start()

            t = threading.Timer(
                model_timeout / 2.0, idle_check
            )
            t.daemon = True
            t.start()

    return app


def cli():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(prog="deepsafe")
    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument(
        "--manifest", required=True, help="Path to model.yaml"
    )

    args = parser.parse_args()

    if args.command == "serve":
        manifest_path = os.path.abspath(args.manifest)
        model_dir = os.path.dirname(manifest_path)
        manifest = load_manifest(manifest_path)

        logger.info(
            f"Starting DeepSafe model server: {manifest.name}"
        )
        ensure_weights(manifest.weights, model_dir)

        app = create_app(manifest, model_dir)
        port = int(
            os.environ.get("MODEL_PORT", str(manifest.port))
        )
        uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
    else:
        parser.print_help()
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest sdk/tests/test_server.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Re-install SDK and commit**

```bash
pip install -e sdk/
git add sdk/deepsafe_sdk/server.py sdk/tests/test_server.py
git commit -m "feat(sdk): add FastAPI server factory and deepsafe serve CLI"
```

---

## Task 9: Migrate NPR-DeepfakeDetection

**Files:**
- Create: `models/image/npr_deepfakedetection/model.yaml`
- Create: `models/image/npr_deepfakedetection/detector.py`
- Modify: `models/image/npr_deepfakedetection/Dockerfile`
- Delete: `models/image/npr_deepfakedetection/app.py`

- [ ] **Step 1: Create model.yaml manifest**

Create `models/image/npr_deepfakedetection/model.yaml`:

```yaml
name: npr_deepfakedetection
version: "1.0.0"
media_type: image
model_class: detector.NPRDetector
port: 5001

weights:
  - url: https://github.com/chuangchuangtan/NPR-DeepfakeDetection/raw/main/model_epoch_last_3090.pth
    path: npr_deepfakedetection/weights/NPR.pth
    sha256: ""

environment:
  PRELOAD_MODEL: "false"
  MODEL_TIMEOUT: "600"
  USE_GPU: "false"
```

- [ ] **Step 2: Create detector.py with model-specific logic**

Create `models/image/npr_deepfakedetection/detector.py`:

```python
import os
import sys
import torch
import torchvision.transforms as transforms
from deepsafe_sdk import ImageModel, PredictionResult

# Add model code to path for resnet50 import
MODEL_REPO_SUBDIR = "npr_deepfakedetection"
current_dir = os.path.dirname(os.path.abspath(__file__))
model_code_path = os.path.join(current_dir, MODEL_REPO_SUBDIR)
if model_code_path not in sys.path:
    sys.path.insert(0, model_code_path)

from networks.resnet import resnet50


class NPRDetector(ImageModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        use_gpu = os.environ.get(
            "USE_GPU", "false"
        ).lower() == "true"
        self.device = torch.device(
            "cuda"
            if use_gpu and torch.cuda.is_available()
            else "cpu"
        )
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def load(self):
        weights_path = self.weights_path(
            "npr_deepfakedetection/weights/NPR.pth"
        )
        net = resnet50(num_classes=1)
        state_dict = torch.load(
            weights_path,
            map_location=self.device,
            weights_only=False,
        )
        if all(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {
                k[len("module."):]: v
                for k, v in state_dict.items()
            }
        net.load_state_dict(state_dict)
        net.to(self.device)
        net.eval()
        self.model = net

    def predict(
        self, input_data: str, threshold: float
    ) -> PredictionResult:
        image = self.decode_image(input_data)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self.model(tensor)
            probability = torch.sigmoid(logit).item()
        return self.make_result(
            probability=probability, threshold=threshold
        )
```

- [ ] **Step 3: Update Dockerfile to use SDK**

Replace contents of `models/image/npr_deepfakedetection/Dockerfile` with:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git wget && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY sdk/ /app/sdk/
RUN pip install --no-cache-dir /app/sdk/

RUN pip install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cpu

COPY models/image/npr_deepfakedetection/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY models/image/npr_deepfakedetection/ /app/model/

WORKDIR /app/model

RUN git clone https://github.com/chuangchuangtan/NPR-DeepfakeDetection.git npr_deepfakedetection && \
    mkdir -p npr_deepfakedetection/weights && \
    wget -O npr_deepfakedetection/weights/NPR.pth \
    https://github.com/chuangchuangtan/NPR-DeepfakeDetection/raw/main/model_epoch_last_3090.pth

ENV MODEL_PORT=5001
ENV USE_GPU=false
ENV PRELOAD_MODEL=false
ENV MODEL_TIMEOUT=600

EXPOSE 5001

CMD ["deepsafe", "serve", "--manifest", "model.yaml"]
```

- [ ] **Step 4: Delete old app.py and commit**

```bash
git rm models/image/npr_deepfakedetection/app.py
git add models/image/npr_deepfakedetection/model.yaml \
       models/image/npr_deepfakedetection/detector.py \
       models/image/npr_deepfakedetection/Dockerfile
git commit -m "refactor(npr): migrate NPR-DeepfakeDetection to DeepSafe SDK"
```

---

## Task 10: Migrate UniversalFakeDetect

**Files:**
- Create: `models/image/universalfakedetect/model.yaml`
- Create: `models/image/universalfakedetect/detector.py`
- Modify: `models/image/universalfakedetect/Dockerfile`
- Delete: `models/image/universalfakedetect/app.py`

- [ ] **Step 1: Create model.yaml manifest**

Create `models/image/universalfakedetect/model.yaml`:

```yaml
name: universalfakedetect
version: "1.0.0"
media_type: image
model_class: detector.UniversalFakeDetector
port: 5004

weights:
  - url: https://github.com/WisconsinAIVision/UniversalFakeDetect/raw/main/pretrained_weights/fc_weights.pth
    path: universalfakedetect/pretrained_weights/fc_weights.pth
    sha256: ""

environment:
  PRELOAD_MODEL: "false"
  MODEL_TIMEOUT: "600"
  USE_GPU: "false"
```

- [ ] **Step 2: Create detector.py**

Create `models/image/universalfakedetect/detector.py`:

```python
import os
import sys
import torch
from torchvision import transforms
from deepsafe_sdk import ImageModel, PredictionResult

current_dir = os.path.dirname(os.path.abspath(__file__))
model_code_path = os.path.join(current_dir, "universalfakedetect")
if model_code_path not in sys.path:
    sys.path.append(model_code_path)

from models import get_model


class UniversalFakeDetector(ImageModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        use_gpu = os.environ.get(
            "USE_GPU", "false"
        ).lower() == "true"
        self.device = torch.device(
            "cuda"
            if use_gpu and torch.cuda.is_available()
            else "cpu"
        )
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])

    def load(self):
        weights_path = self.weights_path(
            "universalfakedetect/pretrained_weights/fc_weights.pth"
        )
        net = get_model("CLIP:ViT-L/14")
        state_dict = torch.load(
            weights_path, map_location="cpu", weights_only=False
        )
        net.fc.load_state_dict(state_dict)
        net.to(self.device)
        net.eval()
        self.model = net

    def predict(
        self, input_data: str, threshold: float
    ) -> PredictionResult:
        image = self.decode_image(input_data)
        tensor = (
            self.transform(image).unsqueeze(0).to(self.device)
        )
        with torch.no_grad():
            probability = (
                self.model(tensor).sigmoid().flatten().item()
            )
        return self.make_result(
            probability=probability, threshold=threshold
        )
```

- [ ] **Step 3: Update Dockerfile**

Replace contents of `models/image/universalfakedetect/Dockerfile` with:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY sdk/ /app/sdk/
RUN pip install --no-cache-dir /app/sdk/

RUN pip install --no-cache-dir --upgrade pip "setuptools<70" wheel

RUN pip install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cpu

COPY models/image/universalfakedetect/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY models/image/universalfakedetect/ /app/model/

WORKDIR /app/model

RUN git clone https://github.com/WisconsinAIVision/UniversalFakeDetect.git universalfakedetect && \
    mkdir -p universalfakedetect/pretrained_weights

RUN python -c "import torch; import os; os.makedirs('/root/.cache/clip', exist_ok=True)" && \
    wget -O /root/.cache/clip/ViT-L-14.pt https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt

RUN wget -O universalfakedetect/pretrained_weights/fc_weights.pth \
    https://github.com/WisconsinAIVision/UniversalFakeDetect/raw/main/pretrained_weights/fc_weights.pth

ENV MODEL_PORT=5004
ENV USE_GPU=false
ENV PRELOAD_MODEL=false
ENV MODEL_TIMEOUT=600

EXPOSE 5004

CMD ["deepsafe", "serve", "--manifest", "model.yaml"]
```

- [ ] **Step 4: Delete old app.py and commit**

```bash
git rm models/image/universalfakedetect/app.py
git add models/image/universalfakedetect/model.yaml \
       models/image/universalfakedetect/detector.py \
       models/image/universalfakedetect/Dockerfile
git commit -m "refactor(universalfakedetect): migrate to DeepSafe SDK"
```

---

## Task 11: Migrate Cross-Efficient-ViT

**Files:**
- Create: `models/video/cross_efficient_vit/model.yaml`
- Create: `models/video/cross_efficient_vit/detector.py`
- Modify: `models/video/cross_efficient_vit/Dockerfile`
- Delete: `models/video/cross_efficient_vit/app.py`

- [ ] **Step 1: Create model.yaml manifest**

Create `models/video/cross_efficient_vit/model.yaml`:

```yaml
name: cross_efficient_vit
version: "1.0.0"
media_type: video
model_class: detector.CrossEfficientViTDetector
port: 7001

weights:
  - url: ""
    path: model_code/cross-efficient-vit/pretrained_models/cross_efficient_vit.pth
    sha256: ""
  - url: ""
    path: model_code/efficient-vit/pretrained_models/efficient_vit.pth
    sha256: ""

environment:
  PRELOAD_MODEL: "false"
  MODEL_TIMEOUT: "900"
  USE_GPU: "false"
  DEFAULT_MODEL_VARIANT: "cross_efficient_vit"
  FRAMES_PER_VIDEO: "15"
```

- [ ] **Step 2: Create detector.py**

Create `models/video/cross_efficient_vit/detector.py`:

```python
import os
import sys
import gc
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import yaml
from albumentations import Compose, PadIfNeeded

from deepsafe_sdk import VideoModel, PredictionResult

app_dir = os.path.dirname(os.path.abspath(__file__))
for subpath in [
    "model_code/cross-efficient-vit",
    "model_code/efficient-vit",
    "model_code/preprocessing",
    "model_code/cross-efficient-vit/efficient_net",
]:
    abs_path = os.path.join(app_dir, subpath)
    if abs_path not in sys.path:
        sys.path.insert(0, abs_path)

from cross_efficient_vit import CrossEfficientViT
from efficient_vit import EfficientViT
from facenet_pytorch import MTCNN
from transforms.albu import IsotropicResize

IMAGENET_NORMALIZE = T.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
FACE_THRESHOLDS = [0.7, 0.8, 0.8]
MTCNN_MIN_FACE_SIZE = 40


class CrossEfficientViTDetector(VideoModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frames_per_video = int(
            os.environ.get("FRAMES_PER_VIDEO", "15")
        )
        use_gpu = os.environ.get(
            "USE_GPU", "false"
        ).lower() == "true"
        self.device = torch.device(
            "cuda"
            if use_gpu and torch.cuda.is_available()
            else "cpu"
        )
        self.variant = os.environ.get(
            "DEFAULT_MODEL_VARIANT", "cross_efficient_vit"
        )
        self.face_detector = None
        self.face_transform = None
        self.config = None

    def load(self):
        model_paths = {
            "cross_efficient_vit": self.weights_path(
                "model_code/cross-efficient-vit/"
                "pretrained_models/cross_efficient_vit.pth"
            ),
            "efficient_vit": self.weights_path(
                "model_code/efficient-vit/"
                "pretrained_models/efficient_vit.pth"
            ),
        }
        config_paths = {
            "cross_efficient_vit": self.weights_path(
                "model_code/cross-efficient-vit/"
                "configs/architecture.yaml"
            ),
            "efficient_vit": self.weights_path(
                "model_code/efficient-vit/"
                "configs/architecture.yaml"
            ),
        }

        model_path = model_paths[self.variant]
        config_path = config_paths[self.variant]

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        image_size = self.config["model"]["image-size"]

        if self.variant == "cross_efficient_vit":
            net = CrossEfficientViT(config=self.config)
        else:
            channels = 1280
            if self.config["model"].get(
                "selected_efficient_net", 0
            ) == 7:
                channels = 2560
            net = EfficientViT(
                config=self.config,
                channels=channels,
                selected_efficient_net=self.config["model"].get(
                    "selected_efficient_net", 0
                ),
            )

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False,
        )
        state_dict = checkpoint.get(
            "state_dict", checkpoint.get("model", checkpoint)
        )
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict, strict=True)
        net.to(self.device)
        net.eval()
        self.model = net

        self.face_detector = MTCNN(
            keep_all=True,
            device=self.device,
            thresholds=FACE_THRESHOLDS,
            min_face_size=MTCNN_MIN_FACE_SIZE,
            select_largest=False,
        )
        self.face_transform = Compose([
            IsotropicResize(
                max_side=image_size,
                interpolation_down=cv2.INTER_AREA,
                interpolation_up=cv2.INTER_CUBIC,
            ),
            PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
            ),
        ])

    def predict(
        self, input_data: str, threshold: float
    ) -> PredictionResult:
        frames = self.extract_frames(
            input_data, num_frames=self.frames_per_video
        )
        if not frames:
            return self.make_result(
                probability=0.5, threshold=threshold
            )

        all_scores = []
        for frame in frames:
            boxes, probs, landmarks = (
                self.face_detector.detect(frame, landmarks=True)
            )
            if boxes is None:
                continue
            for box, prob, _ in zip(boxes, probs, landmarks):
                if prob < FACE_THRESHOLDS[2]:
                    continue
                xmin, ymin, xmax, ymax = [int(b) for b in box]
                w_face = xmax - xmin
                h_face = ymax - ymin
                pad_h, pad_w = h_face // 3, w_face // 3
                crop_xmin = max(0, xmin - pad_w)
                crop_ymin = max(0, ymin - pad_h)
                crop_xmax = min(frame.shape[1], xmax + pad_w)
                crop_ymax = min(frame.shape[0], ymax + pad_h)
                face_crop = frame[
                    crop_ymin:crop_ymax, crop_xmin:crop_xmax
                ]
                if face_crop.size == 0:
                    continue

                transformed = self.face_transform(
                    image=face_crop
                )["image"]
                tensor = (
                    torch.from_numpy(
                        transformed.astype(np.float32)
                    ).permute(2, 0, 1)
                    / 255.0
                )
                tensor = (
                    IMAGENET_NORMALIZE(tensor)
                    .unsqueeze(0)
                    .to(self.device)
                )

                with torch.no_grad():
                    logits = self.model(tensor)
                    score = (
                        torch.sigmoid(logits).squeeze().item()
                    )
                all_scores.append(score)

        if not all_scores:
            return self.make_result(
                probability=0.5, threshold=threshold
            )

        probability = float(np.mean(all_scores))
        return self.make_result(
            probability=probability, threshold=threshold
        )

    def unload(self):
        super().unload()
        self.face_detector = None
        self.face_transform = None
        self.config = None
        gc.collect()
```

- [ ] **Step 3: Update Dockerfile**

Replace contents of `models/video/cross_efficient_vit/Dockerfile` with:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git wget unzip build-essential cmake \
    libgl1-mesa-dev ffmpeg \
    libavcodec-dev libavformat-dev libswscale-dev \
    libgstreamer1.0-0 gstreamer1.0-plugins-base \
    libjpeg-dev libpng-dev libtiff-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV OPENCV_IO_ENABLE_NONFREE=0
ENV QT_QPA_PLATFORM=offscreen

COPY sdk/ /app/sdk/
RUN pip install --no-cache-dir /app/sdk/

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir opencv-python-headless

COPY models/video/cross_efficient_vit/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY models/video/cross_efficient_vit/ /app/model/

WORKDIR /app/model

RUN git clone https://github.com/davide-coccomini/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection.git model_code

RUN mkdir -p /app/model/model_code/gdrive_weights && \
    for i in 1 2 3; do \
        gdown --folder https://drive.google.com/drive/folders/19bNOs8_rZ7LmPP3boDS3XvZcR1iryHR1 -O /app/model/model_code/gdrive_weights --remaining-ok && break; \
        echo "gdown attempt $i failed, retrying in 10s..."; \
        sleep 10; \
    done

RUN mkdir -p /app/model/model_code/efficient-vit/pretrained_models && \
    mkdir -p /app/model/model_code/cross-efficient-vit/pretrained_models

RUN if [ -f /app/model/model_code/gdrive_weights/efficient_vit.pth ]; then \
        mv /app/model/model_code/gdrive_weights/efficient_vit.pth /app/model/model_code/efficient-vit/pretrained_models/efficient_vit.pth; \
    fi && \
    if [ -f /app/model/model_code/gdrive_weights/cross_efficient_vit.pth ]; then \
        mv /app/model/model_code/gdrive_weights/cross_efficient_vit.pth /app/model/model_code/cross-efficient-vit/pretrained_models/cross_efficient_vit.pth; \
    fi && \
    rm -rf /app/model/model_code/gdrive_weights

ENV PYTHONPATH=/app/model/model_code/cross-efficient-vit:/app/model/model_code/efficient-vit:/app/model/model_code/preprocessing:/app/model/model_code/cross-efficient-vit/efficient_net:${PYTHONPATH}

ENV MODEL_PORT=7001
ENV PRELOAD_MODEL="false"
ENV MODEL_TIMEOUT="900"
ENV USE_GPU="false"
ENV DEFAULT_MODEL_VARIANT="cross_efficient_vit"
ENV FRAMES_PER_VIDEO="15"

EXPOSE 7001

CMD ["deepsafe", "serve", "--manifest", "model.yaml"]
```

- [ ] **Step 4: Delete old app.py and commit**

```bash
git rm models/video/cross_efficient_vit/app.py
git add models/video/cross_efficient_vit/model.yaml \
       models/video/cross_efficient_vit/detector.py \
       models/video/cross_efficient_vit/Dockerfile
git commit -m "refactor(cross-efficient-vit): migrate to DeepSafe SDK"
```

---

## Task 12: Update Docker Compose Build Contexts

**Files:**
- Modify: `docker-compose.yml`

- [ ] **Step 1: Update docker-compose.yml build contexts**

All model services need `context: .` (repo root) so they can `COPY sdk/` during build. Change each active model service's `build:` field from a simple path to a context+dockerfile pair:

For `npr_deepfakedetection`, change:
```yaml
    build: ./models/image/npr_deepfakedetection
```
to:
```yaml
    build:
      context: .
      dockerfile: models/image/npr_deepfakedetection/Dockerfile
```

For `universalfakedetect`, change:
```yaml
    build: ./models/image/universalfakedetect
```
to:
```yaml
    build:
      context: .
      dockerfile: models/image/universalfakedetect/Dockerfile
```

For `cross_efficient_vit`, change:
```yaml
    build: ./models/video/cross_efficient_vit
```
to:
```yaml
    build:
      context: .
      dockerfile: models/video/cross_efficient_vit/Dockerfile
```

- [ ] **Step 2: Commit**

```bash
git add docker-compose.yml
git commit -m "chore: update docker-compose build contexts for SDK integration"
```

---

## Task 13: Write Contributor Documentation

**Files:**
- Create: `docs/adding-a-model.md`
- Delete: `docs/INTEGRATING_NEW_MODELS.md`

- [ ] **Step 1: Write the contributor guide**

Create `docs/adding-a-model.md` covering:

1. **Quick Start** with the 3-file contribution structure
2. **model.yaml reference** with all fields
3. **detector.py examples** for each media type (ImageModel, VideoModel, AudioModel)
4. **Base class utilities** (decode_image, extract_frames, decode_audio, weights_path, make_result)
5. **Dockerfile template** to copy and customize
6. **docker-compose.yml entry** template
7. **deepsafe_config.json entry** template
8. **Verification** commands to test the model
9. **Checklist** for quick reference

See the full content specified in the design spec for this file.

- [ ] **Step 2: Delete old docs**

```bash
git rm docs/INTEGRATING_NEW_MODELS.md
```

- [ ] **Step 3: Commit**

```bash
git add docs/adding-a-model.md
git commit -m "docs: replace old integration guide with SDK-based contributor docs"
```

---

## Task 14: Finalize SDK Exports + Run Full Test Suite

**Files:**
- Modify: `sdk/deepsafe_sdk/__init__.py`

- [ ] **Step 1: Finalize __init__.py**

The final `sdk/deepsafe_sdk/__init__.py`:

```python
from deepsafe_sdk.types import PredictionResult
from deepsafe_sdk.manifest import ModelManifest, load_manifest
from deepsafe_sdk.weights import ensure_weights
from deepsafe_sdk.base import DeepSafeModel
from deepsafe_sdk.image import ImageModel
from deepsafe_sdk.video import VideoModel
from deepsafe_sdk.audio import AudioModel
from deepsafe_sdk.server import create_app

__all__ = [
    "PredictionResult",
    "ModelManifest",
    "load_manifest",
    "ensure_weights",
    "DeepSafeModel",
    "ImageModel",
    "VideoModel",
    "AudioModel",
    "create_app",
]
```

- [ ] **Step 2: Run full test suite**

Run: `python -m pytest sdk/tests/ -v`
Expected: ALL PASS (all tests from tasks 1-8)

- [ ] **Step 3: Commit**

```bash
git add sdk/deepsafe_sdk/__init__.py
git commit -m "feat(sdk): finalize public API exports"
```
