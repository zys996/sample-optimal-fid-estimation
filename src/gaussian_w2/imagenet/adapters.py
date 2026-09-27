"""Lazy adapters for the ImageNet generators used by the ImageNet study.

The upstream repositories are deliberately *not* Python packages.  Several
of them expose identically named top-level modules (notably ``models``,
``utils``, ``dnnlib``, and ``torch_utils``), so importing two repositories in
one interpreter can silently construct the wrong model.  This module keeps
all Torch and upstream imports behind :meth:`generate`, rejects an already
loaded conflicting module, and assumes one generator family per worker
process.

Every adapter returns a CUDA ``torch.uint8`` tensor in NCHW layout and in the
closed range [0, 255].  This is the exact boundary expected by the EDM2
Inception and DINOv2 feature extractors.
"""

from __future__ import annotations

import importlib
import os
import pickle
import subprocess
import sys
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Optional, Sequence, Union
from urllib.parse import urlparse


PathLike = Union[str, os.PathLike[str]]

VAR_PATCH_NUMS = {
    256: (1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
    512: (1, 2, 3, 4, 6, 9, 13, 18, 24, 32),
}
VAR_DEPTH = {256: 30, 512: 36}
VAR_SHARED_ADALN = {256: False, 512: True}

_MISSING = object()


def _expand_path(value: PathLike) -> Path:
    """Expand environment variables and ``~`` and return an absolute path."""

    expanded = os.path.expanduser(os.path.expandvars(os.fspath(value)))
    return Path(expanded).resolve(strict=False)


def _expand_location(value: PathLike) -> str:
    """Expand a local path while preserving HTTP(S) model URLs."""

    text = os.path.expanduser(os.path.expandvars(os.fspath(value)))
    if urlparse(text).scheme.lower() in {"http", "https"}:
        return text
    return str(Path(text).resolve(strict=False))


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except (OSError, ValueError):
        return False
    return True


def _module_origins(module: ModuleType) -> tuple[Path, ...]:
    values: list[Path] = []
    filename = getattr(module, "__file__", None)
    if filename:
        values.append(Path(filename))
    namespace_path = getattr(module, "__path__", None)
    if namespace_path is not None:
        values.extend(Path(value) for value in namespace_path)
    return tuple(values)


def _activate_external_repo(repo_path: Path, module_roots: Sequence[str]) -> None:
    """Prepend an upstream checkout after rejecting module-name collisions.

    A module is accepted only when every discoverable origin is inside the
    requested checkout.  Origin-less modules are rejected as well: accepting
    one would make it impossible to establish that the correct upstream code
    will be used.
    """

    repo_path = repo_path.resolve(strict=False)
    if not repo_path.is_dir():
        raise FileNotFoundError(f"external repository is not a directory: {repo_path}")

    conflicts: list[str] = []
    for loaded_name, module in tuple(sys.modules.items()):
        if module is None:
            continue
        root = next(
            (
                candidate
                for candidate in module_roots
                if loaded_name == candidate or loaded_name.startswith(candidate + ".")
            ),
            None,
        )
        if root is None:
            continue
        origins = _module_origins(module)
        if not origins or not all(_is_within(origin, repo_path) for origin in origins):
            rendered = ", ".join(str(value) for value in origins) or "unknown origin"
            conflicts.append(f"{loaded_name} ({rendered})")

    if conflicts:
        joined = "; ".join(conflicts[:8])
        raise RuntimeError(
            "refusing to mix external generator repositories in one Python "
            f"process; modules expected under {repo_path} are already loaded: {joined}. "
            "Run each generator family in a fresh worker process."
        )

    repo_text = str(repo_path)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    importlib.invalidate_caches()


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be an integer >= {minimum}") from error
    if result != value or result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


@dataclass(frozen=True)
class _GenerationRequest:
    seeds: tuple[int, ...]
    class_ids: tuple[int, ...]
    batch_seed: Optional[int]


def _generation_request(
    seeds: Sequence[int],
    class_ids: Sequence[int],
    batch_seed: Optional[int],
) -> _GenerationRequest:
    normalized_seeds = tuple(_integer(value, name="seed") for value in seeds)
    normalized_classes = tuple(
        _integer(value, name="class_id") for value in class_ids
    )
    if not normalized_seeds:
        raise ValueError("seeds and class_ids must contain at least one sample")
    if len(normalized_seeds) != len(normalized_classes):
        raise ValueError("seeds and class_ids must have equal length")
    if any(value >= 1000 for value in normalized_classes):
        raise ValueError("ImageNet class_id must lie in [0, 999]")
    normalized_batch_seed = (
        None
        if batch_seed is None
        else _integer(batch_seed, name="batch_seed")
    )
    return _GenerationRequest(
        normalized_seeds, normalized_classes, normalized_batch_seed
    )


@contextmanager
def _restored_var_reset_parameters(torch: Any):
    """Undo every global ``reset_parameters`` mutation in VAR's builder."""

    names = (
        "Linear",
        "LayerNorm",
        "BatchNorm2d",
        "SyncBatchNorm",
        "Conv1d",
        "Conv2d",
        "ConvTranspose1d",
        "ConvTranspose2d",
    )
    classes = tuple(getattr(torch.nn, name) for name in names)
    originals = {
        cls: cls.__dict__.get("reset_parameters", _MISSING) for cls in classes
    }
    try:
        yield
    finally:
        for cls, original in originals.items():
            if original is _MISSING:
                if "reset_parameters" in cls.__dict__:
                    delattr(cls, "reset_parameters")
            else:
                setattr(cls, "reset_parameters", original)


def _configure_stylegan_cuda_extension_arch(
    torch: Any,
    device: Any,
    custom_ops: ModuleType,
    cpp_extension: ModuleType,
) -> None:
    """Keep PyTorch 2.6 from auto-parsing GH200's ``sm_90a`` target.

    The pinned StyleGAN-XL ``custom_ops.get_plugin`` clears
    ``TORCH_CUDA_ARCH_LIST`` immediately before calling PyTorch's extension
    loader.  PyTorch 2.6 calculates its automatic architecture flags *before*
    appending ``extra_cuda_cflags``; its legacy parser then attempts to convert
    GH200's ``"90a"`` suffix to an integer.  During each StyleGAN plugin load,
    temporarily make that parser see the ordinary Hopper target ``9.0`` and
    restore both PyTorch and the process environment afterward.
    """

    major, minor = (
        int(value) for value in torch.cuda.get_device_capability(device)
    )
    cuda_arch = f"{major}{minor}"
    current = custom_ops.get_plugin
    if getattr(current, "_gw2_explicit_cuda_arch", None) == cuda_arch:
        return

    cuda_arch_spec = f"{major}.{minor}"

    @wraps(current)
    def get_plugin_with_explicit_arch(*args: Any, **kwargs: Any) -> Any:
        original_get_arch_flags = cpp_extension._get_cuda_arch_flags
        original_arch_env = os.environ.get("TORCH_CUDA_ARCH_LIST", _MISSING)

        @wraps(original_get_arch_flags)
        def get_cuda_arch_flags(*arch_args: Any, **arch_kwargs: Any) -> Any:
            parser_arch_env = os.environ.get("TORCH_CUDA_ARCH_LIST", _MISSING)
            os.environ["TORCH_CUDA_ARCH_LIST"] = cuda_arch_spec
            try:
                return original_get_arch_flags(*arch_args, **arch_kwargs)
            finally:
                if parser_arch_env is _MISSING:
                    os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
                else:
                    os.environ["TORCH_CUDA_ARCH_LIST"] = parser_arch_env

        cpp_extension._get_cuda_arch_flags = get_cuda_arch_flags
        try:
            return current(*args, **kwargs)
        finally:
            cpp_extension._get_cuda_arch_flags = original_get_arch_flags
            if original_arch_env is _MISSING:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = original_arch_env

    get_plugin_with_explicit_arch._gw2_explicit_cuda_arch = cuda_arch
    custom_ops.get_plugin = get_plugin_with_explicit_arch


class ImageNetGeneratorAdapter(ABC):
    """Base class for a lazily loaded, CUDA-only ImageNet generator."""

    family = "abstract"
    module_roots: tuple[str, ...] = ()

    def __init__(self, repo_path: PathLike, *, device: str = "cuda") -> None:
        self.repo_path = _expand_path(repo_path)
        self.device_spec = str(device)
        self._torch: Any = None
        self._device: Any = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        torch = importlib.import_module("torch")
        device = torch.device(self.device_spec)
        if getattr(device, "type", None) != "cuda":
            raise ValueError("ImageNet generator adapters require a CUDA device")
        if not bool(torch.cuda.is_available()):
            raise RuntimeError("CUDA is not available to the ImageNet generator adapter")
        _activate_external_repo(self.repo_path, self.module_roots)
        self._load(torch, device)
        self._torch = torch
        self._device = device
        self._loaded = True

    def generate(
        self,
        seeds: Sequence[int],
        class_ids: Sequence[int],
        batch_seed: Optional[int] = None,
    ) -> Any:
        request = _generation_request(seeds, class_ids, batch_seed)
        self._ensure_loaded()
        with self._torch.inference_mode():
            images = self._generate(request)
        return self._validate_output(images, len(request.seeds))

    def load(self) -> "ImageNetGeneratorAdapter":
        """Materialize model weights without generating a throwaway image."""

        self._ensure_loaded()
        return self

    @property
    def metadata(self) -> dict[str, Any]:
        details = {
            "family": self.family,
            "repository_root": str(self.repo_path),
        }
        contract = getattr(self, "_configuration_contract", None)
        if isinstance(contract, Mapping):
            details["configuration"] = dict(contract)
        return details

    def _validate_output(self, images: Any, expected_batch: int) -> Any:
        torch = self._torch
        if not bool(torch.is_tensor(images)):
            raise TypeError("generator output must be a torch.Tensor")
        if int(images.ndim) != 4 or int(images.shape[0]) != expected_batch:
            raise ValueError(
                "generator output must have shape (len(seeds), 3, H, W)"
            )
        if int(images.shape[1]) != 3:
            raise ValueError("generator output must be NCHW with three channels")
        if images.dtype != torch.uint8:
            raise TypeError("generator output must have dtype torch.uint8")
        if getattr(images.device, "type", None) != "cuda":
            raise ValueError("generator output must remain on CUDA")
        return images.contiguous()

    def _uint8_from_minus_one_one(self, images: Any) -> Any:
        """Match StyleGAN-XL's official ``(x + 1) * 255 / 2`` export."""

        torch = self._torch
        return (
            images.to(device=self._device, dtype=torch.float32)
            .add(1.0)
            .mul(127.5)
            .clamp(0, 255)
            .to(dtype=torch.uint8)
        )

    def _uint8_from_zero_one(self, images: Any) -> Any:
        torch = self._torch
        scaled = images.to(device=self._device, dtype=torch.float32).mul(255.0)
        return scaled.clamp(0, 255).to(dtype=torch.uint8)

    @abstractmethod
    def _load(self, torch: Any, device: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def _generate(self, request: _GenerationRequest) -> Any:
        raise NotImplementedError


class StyleGANXLAdapter(ImageNetGeneratorAdapter):
    family = "stylegan_xl"
    module_roots = ("dnnlib", "legacy", "torch_utils")

    def __init__(
        self,
        repo_path: PathLike,
        checkpoint: PathLike,
        *,
        device: str = "cuda",
        truncation_psi: float = 1.0,
        noise_mode: str = "const",
    ) -> None:
        super().__init__(repo_path, device=device)
        self.checkpoint = _expand_location(checkpoint)
        self.truncation_psi = float(truncation_psi)
        if noise_mode not in {"const", "random", "none"}:
            raise ValueError("noise_mode must be 'const', 'random', or 'none'")
        self.noise_mode = noise_mode
        self._generator: Any = None

    def _load(self, torch: Any, device: Any) -> None:
        dnnlib = importlib.import_module("dnnlib")
        custom_ops = importlib.import_module("torch_utils.custom_ops")
        cpp_extension = importlib.import_module("torch.utils.cpp_extension")
        _configure_stylegan_cuda_extension_arch(
            torch, device, custom_ops, cpp_extension
        )
        legacy = importlib.import_module("legacy")
        with dnnlib.util.open_url(self.checkpoint) as handle:
            payload = legacy.load_network_pkl(handle)
        generator = payload["G_ema"]
        self._generator = generator.eval().requires_grad_(False).to(device)
        if int(self._generator.c_dim) != 1000:
            raise ValueError("StyleGAN-XL checkpoint is not ImageNet-1k conditional")

    def _generate(self, request: _GenerationRequest) -> Any:
        torch = self._torch
        np = importlib.import_module("numpy")
        latent = np.stack(
            [
                np.random.RandomState(seed % (1 << 32)).randn(
                    int(self._generator.z_dim)
                )
                for seed in request.seeds
            ]
        ).astype("float32", copy=False)
        z = torch.from_numpy(latent).to(self._device)
        labels = torch.as_tensor(
            request.class_ids, dtype=torch.long, device=self._device
        )
        condition = torch.nn.functional.one_hot(
            labels, num_classes=int(self._generator.c_dim)
        ).to(dtype=torch.float32)
        images = self._generator(
            z,
            condition,
            truncation_psi=self.truncation_psi,
            noise_mode=self.noise_mode,
        )
        return self._uint8_from_minus_one_one(images)


class EDM2Adapter(ImageNetGeneratorAdapter):
    family = "edm2"
    module_roots = ("dnnlib", "torch_utils", "training", "generate_images")

    def __init__(
        self,
        repo_path: PathLike,
        checkpoint: PathLike,
        *,
        device: str = "cuda",
        num_steps: int = 32,
        encoder_batch_size: int = 4,
        sampler_kwargs: Optional[Mapping[str, Any]] = None,
        use_fp16: Optional[bool] = None,
        force_fp32: Optional[bool] = None,
    ) -> None:
        super().__init__(repo_path, device=device)
        self.checkpoint = _expand_location(checkpoint)
        self.num_steps = _integer(num_steps, name="num_steps", minimum=1)
        self.encoder_batch_size = _integer(
            encoder_batch_size, name="encoder_batch_size", minimum=1
        )
        self.sampler_kwargs = dict(sampler_kwargs or {})
        if use_fp16 is not None and not isinstance(use_fp16, bool):
            raise ValueError("use_fp16 must be a boolean or None")
        if force_fp32 is not None and not isinstance(force_fp32, bool):
            raise ValueError("force_fp32 must be a boolean or None")
        self.use_fp16 = use_fp16
        self.force_fp32 = force_fp32
        forbidden = {"guidance", "gnet", "num_steps", "randn_like"}
        overlap = forbidden.intersection(self.sampler_kwargs)
        if overlap:
            raise ValueError(f"sampler_kwargs cannot override {sorted(overlap)}")
        self._network: Any = None
        self._encoder: Any = None
        self._sampler: Any = None
        self._stacked_rng: Any = None

    def _load(self, torch: Any, device: Any) -> None:
        dnnlib = importlib.import_module("dnnlib")
        generation = importlib.import_module("generate_images")
        with dnnlib.util.open_url(self.checkpoint) as handle:
            payload = pickle.load(handle)
        network = payload["ema"].eval().requires_grad_(False).to(device)
        # DDO's released EDM2 checkpoints use the same network and sampler API
        # as EDM2.  Its official inference recipe explicitly selects FP16 and
        # clears force_fp32 after loading; vanilla EDM2 leaves both untouched.
        if self.use_fp16 is not None:
            network.use_fp16 = self.use_fp16
        if self.force_fp32 is not None:
            network.force_fp32 = self.force_fp32
        encoder = payload.get("encoder")
        if encoder is None:
            encoders = importlib.import_module("training.encoders")
            encoder = encoders.StandardRGBEncoder()
        encoder.init(device)
        if hasattr(encoder, "batch_size"):
            encoder.batch_size = self.encoder_batch_size
        if int(network.label_dim) != 1000:
            raise ValueError("EDM2 checkpoint is not ImageNet-1k conditional")
        self._network = network
        self._encoder = encoder
        self._sampler = generation.edm_sampler
        self._stacked_rng = generation.StackedRandomGenerator

    def _generate(self, request: _GenerationRequest) -> Any:
        torch = self._torch
        network = self._network
        rng = self._stacked_rng(self._device, request.seeds)
        noise = rng.randn(
            [
                len(request.seeds),
                int(network.img_channels),
                int(network.img_resolution),
                int(network.img_resolution),
            ],
            device=self._device,
        )
        class_ids = torch.as_tensor(
            request.class_ids, dtype=torch.long, device=self._device
        )
        labels = torch.nn.functional.one_hot(
            class_ids, num_classes=int(network.label_dim)
        ).to(dtype=torch.float32)
        latents = self._sampler(
            network,
            noise,
            labels,
            gnet=network,
            guidance=1,
            num_steps=self.num_steps,
            randn_like=rng.randn_like,
            **self.sampler_kwargs,
        )
        images = self._encoder.decode(latents)
        return images.to(device=self._device, dtype=torch.uint8)


class VARAdapter(ImageNetGeneratorAdapter):
    family = "var"
    module_roots = ("models", "dist")

    def __init__(
        self,
        repo_path: PathLike,
        checkpoint: PathLike,
        vae_checkpoint: PathLike,
        *,
        resolution: int,
        device: str = "cuda",
        cfg: float = 1.5,
        top_p: float = 0.96,
        top_k: int = 900,
        more_smooth: bool = False,
        autocast_dtype: str = "float16",
    ) -> None:
        super().__init__(repo_path, device=device)
        self.checkpoint = _expand_location(checkpoint)
        self.vae_checkpoint = _expand_location(vae_checkpoint)
        self.resolution = _integer(resolution, name="resolution", minimum=1)
        if self.resolution not in VAR_PATCH_NUMS:
            raise ValueError("official VAR adapters support resolution 256 or 512")
        self.cfg = float(cfg)
        self.top_p = float(top_p)
        self.top_k = _integer(top_k, name="top_k")
        self.more_smooth = bool(more_smooth)
        if autocast_dtype not in {"float16", "bfloat16"}:
            raise ValueError("autocast_dtype must be 'float16' or 'bfloat16'")
        self.autocast_dtype = autocast_dtype
        self._vae: Any = None
        self._var: Any = None

    def _load(self, torch: Any, device: Any) -> None:
        models = importlib.import_module("models")
        with _restored_var_reset_parameters(torch):
            vae, var = models.build_vae_var(
                V=4096,
                Cvae=32,
                ch=160,
                share_quant_resi=4,
                device=device,
                patch_nums=VAR_PATCH_NUMS[self.resolution],
                num_classes=1000,
                depth=VAR_DEPTH[self.resolution],
                shared_aln=VAR_SHARED_ADALN[self.resolution],
            )
        vae_state = torch.load(self.vae_checkpoint, map_location="cpu", weights_only=True)
        var_state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        if isinstance(vae_state, Mapping) and "state_dict" in vae_state:
            vae_state = vae_state["state_dict"]
        if isinstance(var_state, Mapping) and "state_dict" in var_state:
            var_state = var_state["state_dict"]
        vae.load_state_dict(vae_state, strict=True)
        var.load_state_dict(var_state, strict=True)
        self._vae = vae.eval().requires_grad_(False)
        self._var = var.eval().requires_grad_(False)

    def _generate(self, request: _GenerationRequest) -> Any:
        if request.batch_seed is None:
            raise ValueError("VAR requires the configured first-image batch seed")
        torch = self._torch
        labels = torch.as_tensor(
            request.class_ids, dtype=torch.long, device=self._device
        )
        dtype = getattr(torch, self.autocast_dtype)
        with torch.autocast(
            device_type="cuda", dtype=dtype, enabled=True, cache_enabled=True
        ):
            images = self._var.autoregressive_infer_cfg(
                B=len(request.seeds),
                label_B=labels,
                g_seed=request.batch_seed % (1 << 63),
                cfg=self.cfg,
                top_k=self.top_k,
                top_p=self.top_p,
                more_smooth=self.more_smooth,
            )
        # The official VAR notebook multiplies by 255 and casts to uint8
        # without rounding before writing PNGs.
        return self._uint8_from_zero_one(images)


def _repository_revision(repo_path: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value if len(value) == 40 else None


def create_imagenet_generator(
    family: str,
    *,
    options: Mapping[str, Any],
    device: str = "cuda",
) -> ImageNetGeneratorAdapter:
    """Construct an adapter from the generation-manifest option schema."""

    if not isinstance(options, Mapping):
        raise TypeError("generator options must be a mapping")
    values = dict(options)
    repo_value = values.pop("repository_root", None)
    if repo_value is None:
        raise ValueError("adapter_options.repository_root is required")
    repo_path = _expand_path(repo_value)
    expected_revision = values.pop("repository_revision", None)
    if expected_revision is not None:
        expected_revision = str(expected_revision)
        observed = _repository_revision(repo_path)
        if observed is None:
            raise RuntimeError(f"cannot determine Git revision for {repo_path}")
        if observed != expected_revision:
            raise RuntimeError(
                f"generator checkout {repo_path} is at {observed}, expected "
                f"{expected_revision}"
            )

    # Provenance/scheduling fields are bound by the manifest but are not
    # constructor arguments for upstream models.
    for key in (
        "repository_url",
        "preset",
        "batch_size",
        "seed_semantics",
        "estimated_gpu_hours",
    ):
        values.pop(key, None)
    classes = {"stylegan_xl": StyleGANXLAdapter, "edm2": EDM2Adapter, "var": VARAdapter}
    if family not in classes:
        raise ValueError("unsupported generator family; expected stylegan_xl, edm2, or var")
    if family in {"stylegan_xl", "edm2"}:
        values.pop("resolution", None)
    if family == "edm2" and float(values.pop("guidance", 1.0)) != 1.0:
        raise ValueError("this experiment fixes EDM2 guidance at 1 (unguided)")
    if family == "var":
        resolution = int(values["resolution"])
        values["resolution"] = resolution
        if values.pop("depth", VAR_DEPTH.get(resolution)) != VAR_DEPTH.get(resolution):
            raise ValueError("VAR depth does not match the official resolution preset")
        patches = tuple(values.pop("patch_nums", VAR_PATCH_NUMS.get(resolution, ())))
        if patches != VAR_PATCH_NUMS.get(resolution):
            raise ValueError("VAR patch_nums do not match the official resolution preset")
        if values.get("vae_checkpoint") is None:
            raise ValueError("VAR adapter_options.vae_checkpoint is required")
    # Model constructors own their defaults; pass the configured values once.
    adapter = classes[family](repo_path=repo_path, device=device, **values)
    adapter._configuration_contract = dict(options)
    return adapter


__all__ = ["create_imagenet_generator"]
