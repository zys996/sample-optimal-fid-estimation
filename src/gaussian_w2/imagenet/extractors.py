"""The three ImageNet feature extractors used by the paper.

The project intentionally keeps the image stack optional.  Importing this
module does not load Torch or model weights. The optional image stack is
loaded when a feature extractor is constructed.

The detector protocol is the one implemented by NVLabs EDM2's
``calculate_metrics.py``:

* ``fid``: the 2048-dimensional, TensorFlow-compatible Inception feature;
* ``fd_dinov2``: the 1024-dimensional DINOv2 ViT-L/14 class token;
* ``clip``: the raw 768-dimensional OpenAI CLIP ViT-L/14 image feature.

Images passed to the detectors are NCHW ``uint8`` tensors.  Keeping this
boundary exact avoids silent range or resizing differences between generator
adapters.
"""

from __future__ import annotations

import importlib
import os
import pickle
import sys
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Union


from .adapters import _repository_revision


PathLike = Union[str, os.PathLike[str]]

EDM2_FEATURE_DIMENSIONS: Mapping[str, int] = MappingProxyType(
    {"fid": 2048, "fd_dinov2": 1024}
)
EDM2_FEATURE_KEYS = tuple(EDM2_FEATURE_DIMENSIONS)
EDM2_PROTOCOL = "NVlabs/edm2/calculate_metrics.py"
EDM2_PROTOCOL_REVISION = "4bf8162f601bcc09472ce8a32dd0cbe8889dc8fc"
DINOV2_REPOSITORY = "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"
OPENAI_CLIP_FEATURE_KEY = "clip"
OPENAI_CLIP_FEATURE_DIMENSION = 768
OPENAI_CLIP_MODEL = "ViT-L/14"
OPENAI_CLIP_INPUT_RESOLUTION = 224
OPENAI_CLIP_PROTOCOL = "openai/CLIP"
IMAGENET_FEATURE_DIMENSIONS: Mapping[str, int] = MappingProxyType(
    {**EDM2_FEATURE_DIMENSIONS, OPENAI_CLIP_FEATURE_KEY: OPENAI_CLIP_FEATURE_DIMENSION}
)
IMAGENET_FEATURE_KEYS = tuple(IMAGENET_FEATURE_DIMENSIONS)
IMAGENET_FEATURE_PROTOCOL = "gaussian-w2-imagenet-three-feature-v1"


def edm2_extractor_signature(*, dino_resize_mode: str = "torch") -> dict[str, Any]:
    """Return the stable signature shared by reference and generated pools."""

    if dino_resize_mode != "torch":
        raise ValueError("the paper uses dino_resize_mode='torch'")
    return {
        "protocol": EDM2_PROTOCOL,
        "implementation_revision": EDM2_PROTOCOL_REVISION,
        "fid": {
            "detector": "InceptionV3Detector",
            "dimension": 2048,
            "output": "TensorFlow-compatible Inception feature",
        },
        "fd_dinov2": {
            "detector": "DINOv2Detector",
            "model": "dinov2_vitl14",
            "dimension": 1024,
            "resize_mode": dino_resize_mode,
            "l2_normalized": False,
        },
        "input": "NCHW uint8 RGB",
    }


def openai_clip_extractor_signature() -> dict[str, Any]:
    """Return the unchanged historical identifier for the tensor CLIP protocol.

    The legacy ``preprocessing`` label is retained to match archived feature
    metadata. It does not mean numerical equivalence to CLIP's PIL transform;
    the implemented preprocessing is documented in ``_CLIPFeatures``.
    """

    return {
        "protocol": OPENAI_CLIP_PROTOCOL,
        "detector": "OpenAI CLIP",
        "model": OPENAI_CLIP_MODEL,
        "dimension": OPENAI_CLIP_FEATURE_DIMENSION,
        "input_resolution": OPENAI_CLIP_INPUT_RESOLUTION,
        "preprocessing": "official CLIP resize/crop and channel normalization",
        "projected": True,
        "l2_normalized": False,
        "output": "raw encode_image output before L2 normalization",
        "input": "NCHW uint8 RGB",
    }


def imagenet_extractor_signature(
    *, dino_resize_mode: str = "torch"
) -> dict[str, Any]:
    """Return the unified signature for the three ImageNet embeddings."""

    edm2 = edm2_extractor_signature(dino_resize_mode=dino_resize_mode)
    return {
        "protocol": IMAGENET_FEATURE_PROTOCOL,
        "fid": edm2["fid"],
        "fd_dinov2": edm2["fd_dinov2"],
        OPENAI_CLIP_FEATURE_KEY: openai_clip_extractor_signature(),
        "input": "NCHW uint8 RGB",
    }


def _collision_safe_edm2_detector_classes(
    edm2_repo: PathLike, torch: Any, *, dino_repository: str = DINOV2_REPOSITORY, dino_source: str = "github"
) -> tuple[type, type, Path]:
    """Return an exact lightweight transcription of EDM2's two detectors.

    Importing EDM2's full ``calculate_metrics.py`` also imports its
    ``training`` and ``generate_images`` top-level packages.  StyleGAN-XL uses
    the same names, so that otherwise corrupts a one-process
    generate-then-extract lane.  These two tiny classes follow the official
    detector code at :data:`EDM2_PROTOCOL_REVISION` without importing those
    unrelated CLI modules.
    """

    repo = Path(edm2_repo).expanduser().resolve()
    source = repo / "calculate_metrics.py"
    if not source.is_file():
        raise FileNotFoundError(
            f"EDM2 calculate_metrics.py not found under external repo: {repo}"
        )
    if "dnnlib" not in sys.modules:
        repo_text = str(repo)
        if repo_text not in sys.path:
            sys.path.insert(0, repo_text)
        importlib.invalidate_caches()
    dnnlib = importlib.import_module("dnnlib")
    if not hasattr(dnnlib, "util") or not hasattr(dnnlib.util, "open_url"):
        raise RuntimeError("loaded dnnlib is incompatible with EDM2 detectors")

    class InceptionV3Detector:
        feature_dim = 2048

        def __init__(self):
            url = (
                "https://api.ngc.nvidia.com/v2/models/nvidia/research/"
                "stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl"
            )
            with dnnlib.util.open_url(url, verbose=False) as handle:
                self.model = pickle.load(handle)

        def __call__(self, images):
            return self.model.to(images.device)(images, return_features=True)

    class DINOv2Detector:
        feature_dim = 1024

        def __init__(self, resize_mode="torch"):
            if resize_mode != "torch":
                raise ValueError("the paper uses DINO torch resize")
            self.resize_mode = resize_mode
            if hasattr(dnnlib, "make_cache_dir_path"):
                torch.hub.set_dir(dnnlib.make_cache_dir_path("torch_hub"))
            self.model = torch.hub.load(
                dino_repository,
                "dinov2_vitl14",
                source=dino_source,
                trust_repo=True,
                verbose=False,
                skip_validation=True,
            )
            self.model.eval().requires_grad_(False)

        def __call__(self, images):
            images = torch.nn.functional.interpolate(
                images.to(torch.float32),
                size=(224, 224),
                mode="bicubic",
                antialias=True,
            )
            images = images.to(torch.float32) / 255.0
            mean = torch.as_tensor(
                [0.485, 0.456, 0.406], device=images.device, dtype=images.dtype
            ).reshape(1, -1, 1, 1)
            std = torch.as_tensor(
                [0.229, 0.224, 0.225], device=images.device, dtype=images.dtype
            ).reshape(1, -1, 1, 1)
            return self.model.to(images.device)((images - mean) / std)

    return InceptionV3Detector, DINOv2Detector, source


class _EDM2Features:
    """Load the official EDM2 Inception and DINOv2 detectors.

    The paper uses the Torch bicubic resize for DINOv2. The lightweight
    detector transcription avoids generator-module name collisions.
    """

    feature_keys = EDM2_FEATURE_KEYS
    feature_dimensions = EDM2_FEATURE_DIMENSIONS

    def __init__(
        self,
        edm2_repo: PathLike,
        *,
        device: Any = "cuda",
        dino_resize_mode: str = "torch",
        expected_revision: Optional[str] = EDM2_PROTOCOL_REVISION,
        dino_repository: str = DINOV2_REPOSITORY,
        dino_source: str = "github",
    ) -> None:
        try:
            import torch
        except ImportError as error:  # pragma: no cover - server-only dependency
            raise RuntimeError("EDM2 feature extraction requires PyTorch") from error
        if dino_resize_mode != "torch":
            raise ValueError("the paper uses dino_resize_mode='torch'")
        try:
            inception_class, dino_class, source = _collision_safe_edm2_detector_classes(
                edm2_repo, torch, dino_repository=dino_repository, dino_source=dino_source
            )
        except (ImportError, AttributeError) as error:
            raise RuntimeError(
                "failed to load detectors from the external EDM2 repository"
            ) from error
        self._torch = torch
        self.device = torch.device(device)
        self.edm2_repo = source.parent
        self.dino_resize_mode = dino_resize_mode
        self.dino_repository = dino_repository
        self.dino_source = dino_source
        self.edm2_revision = _repository_revision(self.edm2_repo)
        if (
            expected_revision is not None
            and self.edm2_revision is not None
            and self.edm2_revision != expected_revision
        ):
            raise RuntimeError(
                f"external EDM2 checkout is at {self.edm2_revision}, expected "
                f"{expected_revision}"
            )
        self._detectors = {
            "fid": inception_class(),
            "fd_dinov2": dino_class(resize_mode=dino_resize_mode),
        }
        for key, detector in self._detectors.items():
            if int(getattr(detector, "feature_dim", -1)) != EDM2_FEATURE_DIMENSIONS[key]:
                raise RuntimeError(f"external EDM2 {key} detector has an unexpected dimension")
            model = getattr(detector, "model", None)
            if model is not None:
                model.to(self.device)
                model.eval()
                model.requires_grad_(False)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "protocol": EDM2_PROTOCOL,
            "extractor_signature": edm2_extractor_signature(
                dino_resize_mode=self.dino_resize_mode
            ),
            "feature_keys": list(EDM2_FEATURE_KEYS),
            "feature_dimensions": dict(EDM2_FEATURE_DIMENSIONS),
            "inception": "EDM2 InceptionV3Detector; TensorFlow-compatible 2048-D features",
            "dinov2": "dinov2_vitl14 class token; no L2 normalization",
            "dino_resize_mode": self.dino_resize_mode,
            "dino_repository": self.dino_repository,
            "dino_source": self.dino_source,
            "input": "NCHW uint8 RGB",
            "edm2_repo": str(self.edm2_repo),
            "edm2_revision": self.edm2_revision,
            "edm2_expected_revision": EDM2_PROTOCOL_REVISION,
            "torch_version": str(self._torch.__version__),
            "device": str(self.device),
        }

    def extract(self, images: Any) -> dict[str, Any]:
        """Extract both feature tensors without copying them to host memory."""

        torch = self._torch
        result: dict[str, Any] = {}
        with torch.inference_mode():
            for key in EDM2_FEATURE_KEYS:
                features = self._detectors[key](images)
                if not isinstance(features, torch.Tensor):
                    raise RuntimeError(f"EDM2 {key} detector did not return a tensor")
                expected_shape = (int(images.shape[0]), EDM2_FEATURE_DIMENSIONS[key])
                if tuple(features.shape) != expected_shape:
                    raise RuntimeError(
                        f"EDM2 {key} detector returned {tuple(features.shape)}, "
                        f"expected {expected_shape}"
                    )
                result[key] = features.detach()
        return result


class _CLIPFeatures:
    """Official OpenAI CLIP ViT-L/14 image embedding without L2 normalization.

    The official package's :meth:`encode_image` method returns the projected
    768-dimensional image feature.  CLIP normalizes that tensor only later in
    ``forward`` when it forms image--text logits, so calling ``encode_image``
    directly gives the raw pre-L2 feature required by the experiment.

    Historical preprocessing converts NCHW uint8 RGB to float32, resizes the
    shorter side to 224 with antialiased tensor bicubic interpolation, uses a
    floor-offset center crop, then divides by 255 and applies CLIP's channel
    means and standard deviations. It does not use the official PIL transform
    returned by ``clip.load`` and is not claimed to be numerically equivalent.
    """

    feature_key = OPENAI_CLIP_FEATURE_KEY
    feature_dimension = OPENAI_CLIP_FEATURE_DIMENSION

    def __init__(
        self,
        *,
        device: Any = "cuda",
        download_root: Optional[PathLike] = None,
    ) -> None:
        try:
            import torch
        except ImportError as error:  # pragma: no cover - server-only dependency
            raise RuntimeError("OpenAI CLIP feature extraction requires PyTorch") from error
        try:
            clip = importlib.import_module("clip")
        except ImportError as error:  # pragma: no cover - server-only dependency
            raise RuntimeError(
                "OpenAI CLIP is not installed; install the official openai/CLIP package"
            ) from error
        loader = getattr(clip, "load", None)
        if not callable(loader):
            raise RuntimeError(
                "the imported 'clip' module is not the official openai/CLIP package"
            )

        self._torch = torch
        self.device = torch.device(device)
        if download_root is None:
            configured_root = os.environ.get("OPENAI_CLIP_CACHE_DIR")
            download_root = configured_root if configured_root else None
        self.download_root = (
            None
            if download_root is None
            else Path(download_root).expanduser().resolve()
        )
        load_options: dict[str, Any] = {
            "device": self.device,
            "jit": False,
        }
        if self.download_root is not None:
            self.download_root.mkdir(parents=True, exist_ok=True)
            load_options["download_root"] = str(self.download_root)
        self.model, _ = loader(OPENAI_CLIP_MODEL, **load_options)
        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        resolution = int(getattr(self.model.visual, "input_resolution", -1))
        if resolution != OPENAI_CLIP_INPUT_RESOLUTION:
            raise RuntimeError(
                f"OpenAI CLIP {OPENAI_CLIP_MODEL} expects resolution {resolution}, "
                f"not {OPENAI_CLIP_INPUT_RESOLUTION}"
            )
        self._clip_module_path = str(getattr(clip, "__file__", "")) or None

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "protocol": OPENAI_CLIP_PROTOCOL,
            "extractor_signature": openai_clip_extractor_signature(),
            "feature_key": OPENAI_CLIP_FEATURE_KEY,
            "feature_dimension": OPENAI_CLIP_FEATURE_DIMENSION,
            "model": OPENAI_CLIP_MODEL,
            "output": "raw encode_image output before L2 normalization",
            "input": "NCHW uint8 RGB",
            "download_root": (
                None if self.download_root is None else str(self.download_root)
            ),
            "clip_module_path": self._clip_module_path,
            "torch_version": str(self._torch.__version__),
            "device": str(self.device),
        }

    def _preprocess(self, images: Any) -> Any:
        """Apply the historical tensor resize/crop and CLIP channel statistics."""

        torch = self._torch
        height, width = int(images.shape[-2]), int(images.shape[-1])
        target = OPENAI_CLIP_INPUT_RESOLUTION
        if height < 1 or width < 1:
            raise ValueError("images must have positive spatial dimensions")
        if height <= width:
            resized_height = target
            resized_width = int(target * width / height)
        else:
            resized_width = target
            resized_height = int(target * height / width)
        values = images.to(
            device=self.device, dtype=torch.float32, non_blocking=True
        )
        values = torch.nn.functional.interpolate(
            values,
            size=(resized_height, resized_width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        top = (resized_height - target) // 2
        left = (resized_width - target) // 2
        values = values[..., top : top + target, left : left + target] / 255.0
        mean = torch.as_tensor(
            [0.48145466, 0.4578275, 0.40821073],
            device=self.device,
            dtype=values.dtype,
        ).reshape(1, 3, 1, 1)
        std = torch.as_tensor(
            [0.26862954, 0.26130258, 0.27577711],
            device=self.device,
            dtype=values.dtype,
        ).reshape(1, 3, 1, 1)
        return (values - mean) / std

    def extract(self, images: Any) -> Any:
        """Return one raw 768-D ``encode_image`` feature per input image."""

        torch = self._torch
        count = int(images.shape[0])
        with torch.inference_mode():
            features = self.model.encode_image(self._preprocess(images))
        if not isinstance(features, torch.Tensor):
            raise RuntimeError("OpenAI CLIP encode_image did not return a tensor")
        expected_shape = (count, OPENAI_CLIP_FEATURE_DIMENSION)
        if tuple(features.shape) != expected_shape:
            raise RuntimeError(
                f"OpenAI CLIP returned {tuple(features.shape)}, expected {expected_shape}"
            )
        return features.detach()


class ImageNetFeatureExtractor:
    """Extract matched Inception, DINOv2, and raw OpenAI CLIP features."""

    feature_keys = IMAGENET_FEATURE_KEYS
    feature_dimensions = IMAGENET_FEATURE_DIMENSIONS

    def __init__(
        self,
        edm2_repo: PathLike,
        *,
        device: Any = "cuda",
        dino_resize_mode: str = "torch",
        expected_revision: Optional[str] = EDM2_PROTOCOL_REVISION,
        dino_repository: str = DINOV2_REPOSITORY,
        dino_source: str = "github",
        clip_download_root: Optional[PathLike] = None,
    ) -> None:
        self._edm2 = _EDM2Features(
            edm2_repo,
            device=device,
            dino_resize_mode=dino_resize_mode,
            expected_revision=expected_revision,
            dino_repository=dino_repository,
            dino_source=dino_source,
        )
        self._clip = _CLIPFeatures(
            device=device,
            download_root=clip_download_root,
        )
        self._torch = self._edm2._torch
        self.device = self._edm2.device
        self.dino_resize_mode = dino_resize_mode
        self.dino_repository = dino_repository
        self.dino_source = dino_source

    @property
    def metadata(self) -> dict[str, Any]:
        signature = imagenet_extractor_signature(dino_resize_mode=self.dino_resize_mode)
        if (self.dino_repository, self.dino_source) != (DINOV2_REPOSITORY, "github"):
            signature["fd_dinov2"]["model_source"] = {"repository": self.dino_repository, "source": self.dino_source}
        return {
            "protocol": IMAGENET_FEATURE_PROTOCOL,
            "extractor_signature": signature,
            "feature_keys": list(IMAGENET_FEATURE_KEYS),
            "feature_dimensions": dict(IMAGENET_FEATURE_DIMENSIONS),
            "input": "NCHW uint8 RGB",
            "edm2": self._edm2.metadata,
            "clip": self._clip.metadata,
            "torch_version": str(self._torch.__version__),
            "device": str(self.device),
        }

    def extract(self, images: Any) -> dict[str, Any]:
        """Extract all three feature tensors without copying them to the host."""

        torch = self._torch
        if not isinstance(images, torch.Tensor):
            images = torch.as_tensor(images)
        if images.ndim != 4 or int(images.shape[1]) != 3 or int(images.shape[0]) < 1:
            raise ValueError("images must be a nonempty NCHW tensor with three channels")
        if images.dtype != torch.uint8:
            raise ValueError("ImageNet detector input must have dtype uint8")
        images = images.to(device=self.device, non_blocking=True)
        result = self._edm2.extract(images)
        result[OPENAI_CLIP_FEATURE_KEY] = self._clip.extract(images)
        return result


__all__ = ["ImageNetFeatureExtractor", "imagenet_extractor_signature"]
