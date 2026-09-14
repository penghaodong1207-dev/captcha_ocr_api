# CAPTCHA OCR 接口文档

## 1. 文档说明

本文档对应生产封装文件 `ocr.py`，核心类为 `CaptchaOCR`。

模型架构：SVTR-Tiny + CTC  
推理引擎：ONNX Runtime  
默认运行设备：CPU  
默认验证码长度：6 位  
输入模型尺寸：`N × 3 × 64 × 256`

推荐部署目录：

```text
captcha_ocr/
├─ ocr.py
├─ captcha_svtr_fp32.onnx
└─ captcha_dict.txt
```

生产环境依赖：

```bash
pip install onnxruntime opencv-python numpy
```

不需要安装 PaddlePaddle、PaddleOCR 或 Paddle2ONNX。

---

## 2. 快速开始

```python
from ocr import CaptchaOCR

ocr = CaptchaOCR()

text, confidence = ocr.predict(
    r"D:\test\captcha.png"
)

print(text)
print(confidence)
```

返回示例：

```text
A7q3BT
0.998652
```

建议在程序启动时只初始化一次 `CaptchaOCR`，后续重复调用同一个实例。

---

# 3. CaptchaOCR

## 3.1 初始化

```python
CaptchaOCR(
    model_path=None,
    dict_path=None,
    *,
    expected_length=6,
    min_confidence=0.0,
    providers=None,
    intra_op_num_threads=None,
    warmup_runs=3,
)
```

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `model_path` | `str \| Path \| None` | `None` | ONNX 模型路径。为空时自动读取 `ocr.py` 同目录下的 `captcha_svtr_fp32.onnx` |
| `dict_path` | `str \| Path \| None` | `None` | 字符字典路径。为空时自动读取 `ocr.py` 同目录下的 `captcha_dict.txt` |
| `expected_length` | `int \| None` | `6` | 期望识别结果长度。设为 `None` 时关闭长度校验 |
| `min_confidence` | `float` | `0.0` | `accepted` 判断使用的最低置信度，范围 `0~1` |
| `providers` | `Sequence[str] \| None` | `None` | ONNX Runtime Provider。默认 `["CPUExecutionProvider"]` |
| `intra_op_num_threads` | `int \| None` | `None` | ONNX Runtime 单算子线程数，默认由 ORT 自动决定 |
| `warmup_runs` | `int` | `3` | 初始化时预热次数，用于降低第一次真实推理的额外延迟 |

### 示例

```python
from ocr import CaptchaOCR

ocr = CaptchaOCR(
    model_path=r"D:\captcha_ocr\captcha_svtr_fp32.onnx",
    dict_path=r"D:\captcha_ocr\captcha_dict.txt",
    expected_length=6,
    min_confidence=0.90,
    warmup_runs=3,
)
```

---

# 4. 输入类型

`predict()` 和 `predict_detail()` 支持以下输入：

```python
str
Path
bytes
bytearray
memoryview
numpy.ndarray
```

对应使用场景：

| 输入类型 | 场景 |
|---|---|
| `str / Path` | 本地图片路径 |
| `bytes` | `requests.get(...).content` |
| `bytearray` | 内存图片数据 |
| `memoryview` | 内存缓冲区 |
| `numpy.ndarray` | OpenCV 图片 |
| Base64 字符串 | 使用 `predict_base64()` |

支持常见 PNG、JPG、WebP 等 OpenCV 可解码图片格式。

灰度图会自动转换为 BGR，BGRA 四通道图片也会自动转换为 BGR。

---

# 5. 单张识别接口

## 5.1 `predict()`

### 定义

```python
predict(image) -> tuple[str, float]
```

### 参数

| 参数 | 类型 | 说明 |
|---|---|---|
| `image` | `ImageInput` | 图片路径、图片 bytes 或 OpenCV ndarray |

### 返回值

```python
(text, confidence)
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `text` | `str` | OCR 识别文本 |
| `confidence` | `float` | 识别置信度 |

置信度为 CTC 解码后最终保留字符对应 timestep 最大概率的平均值。

### 示例：本地文件

```python
from ocr import CaptchaOCR

ocr = CaptchaOCR()

text, confidence = ocr.predict(
    r"D:\test\captcha.png"
)

print(text)
print(confidence)
```

---

## 5.2 使用 requests 返回的图片 bytes

```python
import requests

from ocr import CaptchaOCR

ocr = CaptchaOCR()

response = requests.get(
    "https://example.com/captcha",
    timeout=10,
)

response.raise_for_status()

text, confidence = ocr.predict(
    response.content
)

print(text, confidence)
```

不需要先保存临时图片。

---

## 5.3 使用 OpenCV 图片

```python
import cv2

from ocr import CaptchaOCR

ocr = CaptchaOCR()

image = cv2.imread(
    r"D:\test\captcha.png"
)

text, confidence = ocr.predict(image)

print(text, confidence)
```

传入的 OpenCV 图片按 BGR 处理。

---

# 6. 详细结果接口

## 6.1 `predict_detail()`

### 定义

```python
predict_detail(image) -> OCRResult
```

返回一个 `OCRResult`：

```python
@dataclass(frozen=True)
class OCRResult:
    text: str
    confidence: float
    accepted: bool
```

### 字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `text` | `str` | OCR 识别结果 |
| `confidence` | `float` | OCR 置信度 |
| `accepted` | `bool` | 是否满足长度和最低置信度要求 |

`accepted=True` 需要同时满足：

```text
识别长度 == expected_length
并且
confidence >= min_confidence
```

如果 `expected_length=None`，则不检查文本长度。

### 示例

```python
from ocr import CaptchaOCR

ocr = CaptchaOCR(
    expected_length=6,
    min_confidence=0.90,
)

result = ocr.predict_detail(
    r"D:\test\captcha.png"
)

print(result.text)
print(result.confidence)
print(result.accepted)
```

示例：

```text
A7q3BT
0.99763
True
```

推荐生产逻辑：

```python
result = ocr.predict_detail(image)

if not result.accepted:
    # 进入低置信度处理、重新获取图片或其他回退逻辑
    pass
else:
    captcha_text = result.text
```

---

# 7. Base64 接口

## 7.1 `predict_base64()`

### 定义

```python
predict_base64(data: str) -> tuple[str, float]
```

支持两种格式：

### 纯 Base64

```text
iVBORw0KGgoAAAANSUhEUg...
```

### Data URI

```text
data:image/png;base64,iVBORw0KGgoAAAANSUhEUg...
```

### 示例

```python
from ocr import CaptchaOCR

ocr = CaptchaOCR()

text, confidence = ocr.predict_base64(
    base64_string
)

print(text, confidence)
```

网页 `img.src` 示例：

```python
src = element.get_attribute("src")

text, confidence = ocr.predict_base64(src)
```

---

# 8. Batch 批量识别接口

## 8.1 `predict_batch()`

### 定义

```python
predict_batch(images) -> list[tuple[str, float]]
```

利用 ONNX 模型动态 Batch，一次执行多张图片。

### 参数

```python
images: Sequence[ImageInput]
```

不同输入类型可以混合：

```python
results = ocr.predict_batch([
    r"D:\test\1.png",
    r"D:\test\2.png",
    image_bytes,
    cv2_image,
])
```

### 返回值

```python
[
    ("A7q3BT", 0.9981),
    ("B84f7R", 0.9943),
    ...
]
```

### 示例

```python
from ocr import CaptchaOCR

ocr = CaptchaOCR()

results = ocr.predict_batch([
    r"D:\test\1.png",
    r"D:\test\2.png",
    r"D:\test\3.png",
])

for text, confidence in results:
    print(text, confidence)
```

传入空列表时返回：

```python
[]
```

---

# 9. 预热接口

## 9.1 `warmup()`

### 定义

```python
warmup(runs: int = 3) -> None
```

使用全零输入执行若干次 ONNX Runtime 推理，用于减少第一次真实请求的冷启动延迟。

一般无需手动调用，因为初始化 `CaptchaOCR` 时默认：

```python
warmup_runs=3
```

如果需要关闭初始化预热：

```python
ocr = CaptchaOCR(
    warmup_runs=0
)
```

后续再手动：

```python
ocr.warmup(5)
```

---

# 10. 模型信息接口

## 10.1 `model_info`

这是一个只读属性。

### 示例

```python
from ocr import CaptchaOCR

ocr = CaptchaOCR()

info = ocr.model_info

print(info)
```

返回结构：

```python
{
    "model_path": "...",
    "dict_path": "...",
    "providers": ["CPUExecutionProvider"],
    "input_name": "x",
    "input_shape": ["DynamicDimension.0", 3, 64, 256],
    "output_name": "...",
    "output_shape": [...],
    "characters": "23456789ABDEFGHJLNQRTabdefghjnqrt",
    "character_count": 33,
    "expected_length": 6,
    "min_confidence": 0.0,
}
```

该接口适合程序启动日志和环境检查。

---

# 11. OCRResult 数据结构

```python
OCRResult(
    text: str,
    confidence: float,
    accepted: bool,
)
```

示例：

```python
OCRResult(
    text="A7q3BT",
    confidence=0.99763,
    accepted=True,
)
```

对象使用 `frozen=True`，创建后不可修改字段。

---

# 12. 字符字典

默认字符字典文件：

```text
captcha_dict.txt
```

必须和训练时的字符顺序完全一致。

当前字符顺序：

```text
23456789ABDEFGHJLNQRTabdefghjnqrt
```

实际文件格式为每行一个字符：

```text
2
3
4
5
6
7
8
9
A
B
...
```

不要重新排序字符字典。

ONNX 输出类别索引已经和训练时字典顺序绑定。当前 CTC 规则：

```text
index 0 = CTC blank
index 1 = captcha_dict.txt 第 1 个字符
index 2 = captcha_dict.txt 第 2 个字符
...
```

初始化时程序会检查：

```text
模型输出类别数 == 字符数 + 1
```

如果不匹配会直接抛出异常。

---

# 13. 图片预处理规则

当前封装使用与训练一致的预处理：

```text
原始图片
↓
BGR 三通道
↓
resize 到 256 × 64
↓
HWC → CHW
↓
float32
↓
/ 255.0
↓
(x - 0.5) / 0.5
↓
ONNX Runtime
```

对应输入：

```text
N × 3 × 64 × 256
```

不建议在外部再次 resize 或 normalize 后再传入，直接传原始图片即可。

---

# 14. CTC 解码

解码规则：

1. 每个 timestep 选择最大概率类别。
2. 删除连续重复类别。
3. 删除 `index=0` 的 CTC blank。
4. 按 `captcha_dict.txt` 转换成字符。
5. 取最终保留字符对应概率的平均值作为 `confidence`。

调用者无需自行进行 CTC 解码。

---

# 15. 异常说明

接口在参数或文件异常时直接抛出 Python Exception。

常见异常：

| 异常类型 | 场景 |
|---|---|
| `FileNotFoundError` | ONNX 模型不存在、字符字典不存在、图片路径不存在 |
| `ValueError` | `min_confidence` 越界、字典为空、字典重复、Base64 非法、图片无法解码 |
| `TypeError` | 图片输入类型不支持、Base64 参数不是字符串 |
| `RuntimeError` | ONNX 输入输出结构不符合预期、字典与模型类别数不匹配、模型输出 shape 异常 |

推荐生产调用：

```python
try:
    result = ocr.predict_detail(image)

except (FileNotFoundError, ValueError, TypeError, RuntimeError) as exc:
    print("OCR失败:", exc)
```

模型文件、字符字典等部署错误建议在程序启动阶段直接失败，而不是静默忽略。

---

# 16. 命令行接口

`ocr.py` 可以直接作为 CLI 使用。

## 基本调用

```bash
python ocr.py D:\test\captcha.png
```

返回 JSON：

```json
{
  "text": "A7q3BT",
  "confidence": 0.997421,
  "accepted": true
}
```

## 指定模型

```bash
python ocr.py D:\test\captcha.png ^
  --model D:\captcha_ocr\captcha_svtr_fp32.onnx
```

PowerShell：

```powershell
python ocr.py D:\test\captcha.png `
  --model D:\captcha_ocr\captcha_svtr_fp32.onnx
```

## 指定字符字典

```powershell
python ocr.py D:\test\captcha.png `
  --dict D:\captcha_ocr\captcha_dict.txt
```

## 指定最低置信度

```powershell
python ocr.py D:\test\captcha.png `
  --min-confidence 0.90
```

## 输出模型信息

```powershell
python ocr.py D:\test\captcha.png --info
```

---

# 17. 推荐的生产初始化方式

不推荐：

```python
def recognize(image):
    ocr = CaptchaOCR()
    return ocr.predict(image)
```

因为每次都会重新创建 ONNX Runtime Session。

推荐：

```python
from ocr import CaptchaOCR

ocr = CaptchaOCR(
    expected_length=6,
    min_confidence=0.90,
)


def recognize(image):
    return ocr.predict_detail(image)
```

即：

```text
程序启动
↓
加载 ONNX 一次
↓
创建 CaptchaOCR 一次
↓
预热一次
↓
后续重复 predict
```

---

# 18. requests 集成示例

```python
import requests

from ocr import CaptchaOCR


ocr = CaptchaOCR(
    expected_length=6,
    min_confidence=0.90,
)


def get_captcha_text(url: str) -> str | None:
    response = requests.get(
        url,
        timeout=10,
    )

    response.raise_for_status()

    result = ocr.predict_detail(
        response.content
    )

    if not result.accepted:
        return None

    return result.text
```

---

# 19. 网页 Base64 集成示例

```python
from ocr import CaptchaOCR


ocr = CaptchaOCR(
    expected_length=6,
    min_confidence=0.90,
)


def recognize_src(src: str) -> str | None:
    text, confidence = ocr.predict_base64(src)

    if len(text) != 6:
        return None

    if confidence < 0.90:
        return None

    return text
```

---

# 20. 性能说明

当前导出的 ONNX 模型在已测试 CPU 环境、Batch=1、纯模型前向推理下：

```text
Mean : 5.340 ms
P50  : 5.321 ms
P95  : 6.224 ms
P99  : 6.508 ms
Min  : 4.003 ms
Max  : 7.189 ms
```

该数据只表示已测试机器的模型推理耗时，不包含：

```text
网络请求
图片下载
Base64 解码
OpenCV 图片解码
业务逻辑
```

不同 CPU、线程设置和负载下性能会有所变化。

---

# 21. 最简接口总览

```python
from ocr import CaptchaOCR

ocr = CaptchaOCR()
```

### 单张

```python
text, confidence = ocr.predict(image)
```

### 详细结果

```python
result = ocr.predict_detail(image)

result.text
result.confidence
result.accepted
```

### Base64

```python
text, confidence = ocr.predict_base64(base64_data)
```

### Batch

```python
results = ocr.predict_batch(images)
```

### 预热

```python
ocr.warmup(3)
```

### 模型信息

```python
info = ocr.model_info
```
