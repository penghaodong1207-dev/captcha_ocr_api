from __future__ import annotations

import argparse
import base64
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence, Union

import cv2
import numpy as np
import onnxruntime as ort


ImageInput = Union[str, Path, bytes, bytearray, memoryview, np.ndarray]


@dataclass(frozen=True)
class OCRResult:
    text: str
    confidence: float
    accepted: bool


class CaptchaOCR:
    """SVTR-Tiny + CTC 的 ONNX Runtime 生产封装。"""

    def __init__(
        self,
        model_path: str | Path | None = None,
        dict_path: str | Path | None = None,
        *,
        expected_length: int | None = 6,
        min_confidence: float = 0.0,
        providers: Sequence[str] | None = None,
        intra_op_num_threads: int | None = None,
        warmup_runs: int = 3,
    ) -> None:
        base_dir = Path(__file__).resolve().parent
        self.model_path = Path(model_path or base_dir / "captcha_svtr_fp32.onnx").resolve()
        self.dict_path = Path(dict_path or base_dir / "captcha_dict.txt").resolve()

        if not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX 模型不存在: {self.model_path}")
        if not self.dict_path.is_file():
            raise FileNotFoundError(f"字符字典不存在: {self.dict_path}")
        if expected_length is not None and expected_length <= 0:
            raise ValueError("expected_length 必须大于 0 或为 None")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence 必须位于 0~1")

        self.expected_length = expected_length
        self.min_confidence = float(min_confidence)
        self.characters = self._load_characters(self.dict_path)

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_num_threads is not None:
            if intra_op_num_threads <= 0:
                raise ValueError("intra_op_num_threads 必须大于 0")
            options.intra_op_num_threads = intra_op_num_threads

        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=list(providers),
        )

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1:
            raise RuntimeError(f"预期模型只有 1 个输入，实际为 {len(inputs)} 个")
        if not outputs:
            raise RuntimeError("模型没有输出")

        self.input_meta = inputs[0]
        self.output_meta = outputs[0]
        self.input_name = self.input_meta.name
        self.output_name = self.output_meta.name
        self.channels, self.height, self.width = self._resolve_input_shape(self.input_meta.shape)
        self._validate_output_classes(self.output_meta.shape)

        if warmup_runs > 0:
            self.warmup(warmup_runs)

    @staticmethod
    def _load_characters(dict_path: Path) -> list[str]:
        characters = [ch for ch in dict_path.read_text(encoding="utf-8").splitlines() if ch != ""]
        if not characters:
            raise ValueError("字符字典为空")
        invalid = [repr(ch) for ch in characters if len(ch) != 1]
        if invalid:
            raise ValueError("字符字典必须每行恰好一个字符，异常项: " + ", ".join(invalid[:10]))
        if len(set(characters)) != len(characters):
            raise ValueError("字符字典存在重复字符")
        return characters

    @staticmethod
    def _resolve_input_shape(shape: Sequence[object]) -> tuple[int, int, int]:
        if len(shape) != 4:
            raise RuntimeError(f"模型输入应为 NCHW 4维，实际 shape={shape}")
        c, h, w = shape[1], shape[2], shape[3]
        if not all(isinstance(x, (int, np.integer)) for x in (c, h, w)):
            raise RuntimeError(f"模型 C/H/W 必须为静态尺寸，实际 shape={shape}")
        c, h, w = int(c), int(h), int(w)
        if c != 3:
            raise RuntimeError(f"当前预处理只支持 3 通道模型，实际 C={c}")
        return c, h, w

    def _validate_output_classes(self, shape: Sequence[object]) -> None:
        if not shape:
            return
        classes = shape[-1]
        expected_classes = len(self.characters) + 1  # CTC blank
        if isinstance(classes, (int, np.integer)) and int(classes) != expected_classes:
            raise RuntimeError(
                f"模型输出类别数与字符字典不匹配：model={int(classes)}, dict+blank={expected_classes}"
            )

    @staticmethod
    def _decode_image_bytes(data: bytes) -> np.ndarray:
        if not data:
            raise ValueError("图片 bytes 为空")
        arr = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("无法解码图片 bytes，请确认内容确实是 PNG/JPG/WebP 等图片")
        return image

    def _load_image(self, image: ImageInput) -> np.ndarray:
        if isinstance(image, np.ndarray):
            img = image
        elif isinstance(image, (bytes, bytearray, memoryview)):
            img = self._decode_image_bytes(bytes(image))
        elif isinstance(image, (str, Path)):
            path = Path(image)
            if not path.is_file():
                raise FileNotFoundError(f"图片不存在: {path}")
            # 对 Windows 中文路径比 cv2.imread 更稳。
            img = self._decode_image_bytes(path.read_bytes())
        else:
            raise TypeError(f"不支持的图片类型: {type(image).__name__}")

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"不支持的图片 shape: {img.shape}")

        if img.dtype != np.uint8:
            if np.issubdtype(img.dtype, np.floating) and float(np.nanmax(img)) <= 1.0:
                img = img * 255.0
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        # 与 SVTRRecResizeImg(image_shape=[3,64,256], padding=False) 保持一致。
        resized = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        tensor = resized.astype(np.float32, copy=False).transpose(2, 0, 1)
        tensor = tensor / 255.0
        tensor = (tensor - 0.5) / 0.5
        return np.ascontiguousarray(tensor, dtype=np.float32)

    def _ctc_decode(self, prediction: np.ndarray) -> tuple[str, float]:
        if prediction.ndim != 2:
            raise RuntimeError(f"单样本输出应为 [T, C]，实际 shape={prediction.shape}")

        class_ids = np.argmax(prediction, axis=1)
        class_scores = np.max(prediction, axis=1)

        text_chars: list[str] = []
        scores: list[float] = []
        previous_id: int | None = None

        for class_id, score in zip(class_ids, class_scores):
            class_id = int(class_id)
            is_duplicate = previous_id is not None and class_id == previous_id
            previous_id = class_id
            if is_duplicate:
                continue
            if class_id == 0:  # CTC blank
                continue

            char_index = class_id - 1
            if not 0 <= char_index < len(self.characters):
                raise RuntimeError(f"模型输出非法类别 index={class_id}")
            text_chars.append(self.characters[char_index])
            scores.append(float(score))

        text = "".join(text_chars)
        confidence = float(np.mean(scores)) if scores else 0.0
        return text, confidence

    def _is_accepted(self, text: str, confidence: float) -> bool:
        if self.expected_length is not None and len(text) != self.expected_length:
            return False
        return confidence >= self.min_confidence

    def predict(self, image: ImageInput) -> tuple[str, float]:
        """单张识别，返回 (text, confidence)。"""
        img = self._load_image(image)
        batch = self._preprocess(img)[np.newaxis, ...]
        output = self.session.run([self.output_name], {self.input_name: batch})[0]
        if output.ndim != 3 or output.shape[0] != 1:
            raise RuntimeError(f"模型输出应为 [1, T, C]，实际 shape={output.shape}")
        return self._ctc_decode(output[0])

    def predict_detail(self, image: ImageInput) -> OCRResult:
        """返回 text/confidence/accepted。"""
        text, confidence = self.predict(image)
        return OCRResult(
            text=text,
            confidence=confidence,
            accepted=self._is_accepted(text, confidence),
        )

    def predict_batch(self, images: Sequence[ImageInput]) -> list[tuple[str, float]]:
        """利用动态 batch 一次识别多张图片。"""
        if not images:
            return []
        batch = np.stack(
            [self._preprocess(self._load_image(image)) for image in images],
            axis=0,
        ).astype(np.float32, copy=False)
        output = self.session.run([self.output_name], {self.input_name: batch})[0]
        if output.ndim != 3 or output.shape[0] != len(images):
            raise RuntimeError(
                f"模型输出 batch 异常：input={len(images)}, output_shape={output.shape}"
            )
        return [self._ctc_decode(pred) for pred in output]

    def predict_base64(self, data: str) -> tuple[str, float]:
        """支持纯 base64 和 data:image/png;base64,..."""
        if not isinstance(data, str):
            raise TypeError("base64 输入必须是 str")
        value = data.strip()
        if value.startswith("data:"):
            if "," not in value:
                raise ValueError("非法 data URI")
            header, value = value.split(",", 1)
            if ";base64" not in header.lower():
                raise ValueError("data URI 不是 base64 编码")
        value = "".join(value.split())
        try:
            raw = base64.b64decode(value, validate=True)
        except Exception as exc:
            raise ValueError("base64 解码失败") from exc
        return self.predict(raw)

    def warmup(self, runs: int = 3) -> None:
        """预热 ORT，避免第一次真实请求偏慢。"""
        if runs <= 0:
            return
        dummy = np.zeros((1, self.channels, self.height, self.width), dtype=np.float32)
        for _ in range(runs):
            self.session.run([self.output_name], {self.input_name: dummy})

    @property
    def model_info(self) -> dict:
        return {
            "model_path": str(self.model_path),
            "dict_path": str(self.dict_path),
            "providers": self.session.get_providers(),
            "input_name": self.input_name,
            "input_shape": list(self.input_meta.shape),
            "output_name": self.output_name,
            "output_shape": list(self.output_meta.shape),
            "characters": "".join(self.characters),
            "character_count": len(self.characters),
            "expected_length": self.expected_length,
            "min_confidence": self.min_confidence,
        }


def _main() -> None:
    parser = argparse.ArgumentParser(description="SVTR CAPTCHA OCR")
    parser.add_argument("image", help="待识别图片路径")
    parser.add_argument("--model", default=None, help="ONNX 模型路径")
    parser.add_argument("--dict", dest="dict_path", default=None, help="字符字典路径")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--info", action="store_true")
    args = parser.parse_args()

    ocr = CaptchaOCR(
        model_path=args.model,
        dict_path=args.dict_path,
        expected_length=6,
        min_confidence=args.min_confidence,
    )
    payload = asdict(ocr.predict_detail(args.image))
    if args.info:
        payload["model"] = ocr.model_info
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
