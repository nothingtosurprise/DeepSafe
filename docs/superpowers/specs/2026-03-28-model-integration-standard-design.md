# DeepSafe Universal Model Integration Standard

## Problem

Adding a new model to DeepSafe currently requires manually recreating ~200 lines of boilerplate across 5+ files: FastAPI app setup, lazy loading, thread-safe locking, health checks, standard response formatting, Dockerfile, docker-compose entry, and config registration. There's no shared framework — each model reimplements the same patterns independently, leading to inconsistency and maintenance burden.

## Solution

A shared SDK (`deepsafe-sdk`) with a base class hierarchy, manifest-driven configuration, and a universal server entrypoint. A contributor adds a new model by writing one `model.yaml` manifest, one Python file with ~15-50 lines of domain logic, and mechanical copy-paste for Docker/config boilerplate.

## Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Approach | Base class framework + docs (not CLI scaffolding) | Eliminates duplication; scaffolding is premature for 3-9 models |
| SDK distribution | Copied into containers at build time | Simple, no registry needed, easy upgrade path to pip package later |
| Existing models | Refactor all 3 to use SDK | Validates the abstraction is truly universal; avoids two patterns in codebase |
| Media-type handling | Inheritance hierarchy (ImageModel, VideoModel, AudioModel) | Right level of abstraction per domain; intuitive for contributors |
| Weight management | Declarative manifest with URL + checksum | Decouples "what" from "when/how"; operators choose bake-in vs lazy download |
| Overall architecture | Manifest-driven SDK with `deepsafe serve` entrypoint | Single source of truth per model; eliminates per-model app.py boilerplate |

## Architecture

### SDK Package Structure

```
sdk/
├── setup.py
└── deepsafe_sdk/
    ├── __init__.py
    ├── base.py          # DeepSafeModel (abstract base)
    ├── image.py          # ImageModel(DeepSafeModel)
    ├── video.py          # VideoModel(DeepSafeModel)
    ├── audio.py          # AudioModel(DeepSafeModel)
    ├── server.py         # FastAPI app factory + `deepsafe serve` CLI
    ├── weights.py        # Weight download/verification from manifest
    ├── manifest.py       # model.yaml parser + validation
    └── types.py          # Shared Pydantic models (PredictionResult, etc.)
```

The SDK lives at `/sdk/` in the repo root. Each model's Dockerfile copies it in and pip-installs it.

### Base Class Hierarchy

```python
# base.py
class DeepSafeModel(ABC):
    """Abstract base for all DeepSafe models.

    Handles: lazy loading, thread-safe locking, timeout-based unloading,
    health reporting, timing, and standard response formatting.
    """

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory."""

    @abstractmethod
    def predict(self, input_data: str, threshold: float) -> PredictionResult:
        """Run inference. input_data is base64-encoded media."""

    def unload(self) -> None:
        """Release model from memory. Default: delete self.model"""

    def weights_path(self, relative_path: str) -> str:
        """Resolve a weight file path relative to the model directory."""

    def make_result(self, probability: float, threshold: float) -> PredictionResult:
        """Build a PredictionResult with timing and classification filled in."""

# image.py
class ImageModel(DeepSafeModel):
    """Base class for image detection models."""

    def decode_image(self, base64_data: str) -> PIL.Image:
        """Base64 string -> PIL Image."""

# video.py
class VideoModel(DeepSafeModel):
    """Base class for video detection models."""

    frames_per_video: int = 15

    def extract_frames(self, base64_data: str) -> List[np.ndarray]:
        """Base64 video -> list of sampled frames as numpy arrays."""

    def detect_faces(self, frame: np.ndarray) -> List[np.ndarray]:
        """Run face detection on a single frame. Returns cropped face arrays."""

# audio.py
class AudioModel(DeepSafeModel):
    """Base class for audio detection models."""

    def decode_audio(self, base64_data: str) -> Tuple[np.ndarray, int]:
        """Base64 audio -> (waveform_array, sample_rate)."""
```

### Standard Response Type

```python
# types.py
class PredictionResult(BaseModel):
    model: str                # model name from manifest
    probability: float        # 0.0 to 1.0 (probability of being fake)
    prediction: int           # 0=real, 1=fake
    class_name: str           # "real" or "fake"
    inference_time: float     # seconds
```

### Model Manifest (model.yaml)

Each model declares its configuration in a single manifest file:

```yaml
name: my_detector
version: "1.0.0"
media_type: image                # image | video | audio
model_class: detector.MyDetector # Python import path relative to model dir
port: 5008

weights:
  - url: https://example.com/model.pth
    path: weights/model.pth      # relative to model dir
    sha256: a1b2c3d4...

environment:
  PRELOAD_MODEL: "false"
  MODEL_TIMEOUT: "600"
  USE_GPU: "false"

dependencies:                    # extra pip packages beyond the SDK
  - timm==0.9.2
  - efficientnet_pytorch==0.7.1
```

The manifest parser validates:
- Required fields present (name, media_type, model_class, port)
- `media_type` is one of: image, video, audio
- `model_class` is importable
- Weight checksums match after download

### `deepsafe serve` Entrypoint

Replaces per-model `app.py` files entirely. Every model container runs:

```
CMD ["deepsafe", "serve", "--manifest", "model.yaml"]
```

The entrypoint:
1. Parses and validates `model.yaml`
2. Downloads/verifies weights if missing (from manifest)
3. Dynamically imports the `model_class`
4. Creates a FastAPI app with standard endpoints:
   - `GET /` — model info (name, version, media_type)
   - `GET /health` — loaded status, memory usage, last used time
   - `POST /predict` — delegates to model's `predict()`, handles lazy loading + locking
   - `POST /unload` — triggers model unload
5. Wraps the model instance with lazy loading, thread-safe locking, and timeout-based unloading
6. Starts Uvicorn on the configured port

### Weight Management

The `weights.py` module:
1. Reads weight entries from the manifest
2. Checks if each weight file already exists at the declared path
3. If missing, downloads from URL with progress reporting
4. Verifies SHA256 checksum after download
5. Raises clear error if checksum mismatch

This runs at container startup (during `deepsafe serve`) before the model is available for requests.

## Contributor Workflow

### What a contributor writes:

**1. `models/image/my_detector/model.yaml`** — manifest (see above)

**2. `models/image/my_detector/detector.py`** — model logic (~15 lines):

```python
import torch
from deepsafe_sdk import ImageModel, PredictionResult

class MyDetector(ImageModel):
    def load(self):
        self.model = torch.load(self.weights_path("weights/model.pth"))
        self.model.eval()

    def predict(self, input_data: str, threshold: float) -> PredictionResult:
        image = self.decode_image(input_data)
        tensor = self.transform(image)
        with torch.no_grad():
            prob = self.model(tensor).sigmoid().item()
        return self.make_result(probability=prob, threshold=threshold)
```

**3. `models/image/my_detector/Dockerfile`** — template copy-paste:

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY sdk/ /app/sdk/
RUN pip install /app/sdk/
COPY models/image/my_detector/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
COPY models/image/my_detector/ /app/model/
WORKDIR /app/model
EXPOSE 5008
CMD ["deepsafe", "serve", "--manifest", "model.yaml"]
```

**4. Add to `docker-compose.yml`:**

```yaml
my_detector:
  build:
    context: .
    dockerfile: models/image/my_detector/Dockerfile
  ports:
    - "5008:5008"
  environment:
    - PRELOAD_MODEL=false
    - MODEL_TIMEOUT=600
    - USE_GPU=false
  networks:
    - deepsafe-network
  restart: unless-stopped
```

**5. Add to `config/deepsafe_config.json`:**

```json
"my_detector": "http://my_detector:5008/predict"   // in model_endpoints
"my_detector": "http://my_detector:5008/health"     // in health_endpoints
```

## Migration Plan

All 3 active models refactored to use the SDK:

### NPR-DeepfakeDetection (image, port 5001)
- Extends `ImageModel`
- `load()`: loads ResNet50 + NPR weights from `weights/NPR.pth`
- `predict()`: applies transforms, forward pass, sigmoid -> probability
- ~180 lines of boilerplate -> ~25 lines of model-specific logic

### UniversalFakeDetect (image, port 5004)
- Extends `ImageModel`
- `load()`: loads CLIP ViT-L/14 + FC weights
- `predict()`: CLIP preprocessing, forward pass through backbone + FC layer
- Weight download logic moves to `model.yaml` manifest
- ~200 lines -> ~30 lines

### Cross-Efficient-ViT (video, port 7001)
- Extends `VideoModel`
- `load()`: loads EfficientViT checkpoint
- `predict()`: uses inherited `extract_frames()` + custom face detection, per-frame inference, score aggregation
- Frame extraction and face detection become `VideoModel` utilities
- ~250 lines -> ~50 lines

### Migration strategy
- Migrate one model at a time, verify via `/health` and `/predict`
- Run integration tests against API gateway after each migration
- Keep old `app.py` in git history for reference

### What stays unchanged
- API gateway (`/api/main.py`) — talks to models via HTTP, unaffected
- Frontend — no changes
- Config structure — same endpoints
- Docker compose ports — same ports, same service names

## Documentation

A single `docs/adding-a-model.md` guide covering:

1. **Quick Start** — minimum viable model contribution end-to-end
2. **Base Class Reference** — what ImageModel/VideoModel/AudioModel provide, what to implement
3. **Manifest Reference** — all model.yaml fields, required vs optional
4. **Step-by-step Walkthrough** — annotated example of adding a hypothetical model
5. **Checklist** — the 5-step copy-paste checklist

This replaces the existing `docs/INTEGRATING_NEW_MODELS.md`.
