"""Deterministic ImageNet directory/ZIP reader and the EDM2 Dhariwal crop."""
from __future__ import annotations
import os
import zipfile
from pathlib import Path
from typing import Any, Optional, Union
import numpy as np
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def center_crop_dhariwal(image: Any, resolution: int) -> np.ndarray:
    """Apply EDM2 dataset_tool's ImageNet crop exactly.

    ``image`` may be a Pillow image or an HWC NumPy array.  Pillow is imported
    lazily so listing/help/tests that do not decode images need no image stack.
    """

    if resolution < 1:
        raise ValueError("resolution must be positive")
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - server dependency
        raise RuntimeError("ImageNet decoding requires Pillow") from error
    pil_image = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
    pil_image = pil_image.convert("RGB")
    while min(*pil_image.size) >= 2 * resolution:
        pil_image = pil_image.resize(
            tuple(value // 2 for value in pil_image.size),
            resample=Image.Resampling.BOX,
        )
    scale = resolution / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(value * scale) for value in pil_image.size),
        resample=Image.Resampling.BICUBIC,
    )
    array = np.asarray(pil_image, dtype=np.uint8)
    crop_y = (array.shape[0] - resolution) // 2
    crop_x = (array.shape[1] - resolution) // 2
    result = array[crop_y : crop_y + resolution, crop_x : crop_x + resolution]
    if result.shape != (resolution, resolution, 3):
        raise RuntimeError("Dhariwal crop produced an unexpected image shape")
    return np.ascontiguousarray(result)


def _directory_image_names(root: Path) -> list[str]:
    names: list[str] = []
    for directory, _subdirectories, filenames in os.walk(root):
        for filename in filenames:
            path = Path(directory) / filename
            if path.suffix.lower() in IMAGE_SUFFIXES:
                names.append(path.relative_to(root).as_posix())
    names.sort()
    return names


def _zip_image_names(path: Path) -> list[str]:
    with zipfile.ZipFile(path, "r") as archive:
        names = sorted(
            name
            for name in archive.namelist()
            if not name.endswith("/") and Path(name).suffix.lower() in IMAGE_SUFFIXES
        )
    return names


class ImageNetImageDataset:
    """Deterministically ordered raw or EDM2-preprocessed ImageNet images."""

    def __init__(
        self,
        source: Union[os.PathLike[str], str],
        *,
        resolution: int,
        source_format: str,
    ) -> None:
        self.source = Path(source).expanduser().resolve()
        if source_format not in {"raw", "edm2"}:
            raise ValueError("source_format must be 'raw' or 'edm2'")
        if resolution < 1:
            raise ValueError("resolution must be positive")
        self.resolution = int(resolution)
        self.source_format = source_format
        self._archive: Optional[zipfile.ZipFile] = None
        if self.source.is_dir():
            self.source_type = "directory"
            names = _directory_image_names(self.source)
        elif self.source.is_file() and self.source.suffix.lower() == ".zip":
            self.source_type = "zip"
            names = _zip_image_names(self.source)
        else:
            raise FileNotFoundError(
                f"ImageNet source must be a directory or ZIP archive: {self.source}"
            )
        if not names:
            raise ValueError(f"no supported images found under {self.source}")
        self.names = names

    def __len__(self) -> int:
        return len(self.names)

    def _open(self, name: str) -> Any:
        if self.source_type == "directory":
            return (self.source / name).open("rb")
        if self._archive is None:
            self._archive = zipfile.ZipFile(self.source, "r")
        return self._archive.open(name, "r")

    def __getitem__(self, index: int) -> np.ndarray:
        try:
            from PIL import Image
        except ImportError as error:  # pragma: no cover - server dependency
            raise RuntimeError("ImageNet decoding requires Pillow") from error
        with self._open(self.names[int(index)]) as handle:
            with Image.open(handle) as image:
                if self.source_format == "raw":
                    array = center_crop_dhariwal(image, self.resolution)
                else:
                    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
                    if array.shape != (self.resolution, self.resolution, 3):
                        raise ValueError(
                            f"preprocessed image {self.names[int(index)]} has shape "
                            f"{array.shape}, expected "
                            f"({self.resolution},{self.resolution},3)"
                        )
                    array = np.ascontiguousarray(array)
        return np.ascontiguousarray(array.transpose(2, 0, 1))

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_archive"] = None
        return state

    def close(self) -> None:
        if self._archive is not None:
            self._archive.close()
            self._archive = None

    def metadata(self) -> dict[str, Any]:
        return {
            "source": str(self.source),
            "source_type": self.source_type,
            "source_format": self.source_format,
            "sample_count": len(self),
            "resolution": self.resolution,
            "crop": (
                "center-crop-dhariwal"
                if self.source_format == "raw"
                else "already-preprocessed"
            ),
        }


