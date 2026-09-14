# CAPTCHA OCR (SVTR-Tiny + ONNX Runtime)

一个面向固定格式、6 位字符图片的轻量 OCR 推理项目。模型使用 SVTR-Tiny + CTC 训练，并导出为 ONNX，生产环境仅依赖 ONNX Runtime、OpenCV 和 NumPy。

> 仅用于你有权限处理的图片、内部系统和自动化场景。

## 当前模型

- 输入尺寸：`N x 3 x 64 x 256`
- 字符长度：默认 6 位
- 字符集：33 个字符
- 测试集整串准确率：约 **95.19%**
- CPU 单张纯 ONNX 推理：平均约 **5.34 ms**（本机测试）
- 默认 Provider：`CPUExecutionProvider`

字符集顺序固定为：

```text
23456789ABDEFGHJLNQRTabdefghjnqrt
```

`captcha_dict.txt` 的顺序和 ONNX 输出索引绑定，请勿重新排序。

## 目录结构

```text
captcha-ocr/
├─ ocr.py
├─ captcha_svtr_fp32.onnx
├─ captcha_dict.txt
├─ requirements.txt
├─ README.md
├─ .gitignore
├─ .gitattributes
└─ docs/
   └─ API.md
```

## 安装

建议 Python 3.10+：

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

生产推理不需要安装 PaddlePaddle、PaddleOCR 或 Paddle2ONNX。

## 快速开始

```python
from ocr import CaptchaOCR

ocr = CaptchaOCR()

text, confidence = ocr.predict(r"D:\test\captcha.png")

print(text)
print(confidence)
```

示例输出：

```text
A7q3BT
0.998652
```

### 推荐的生产初始化方式

`CaptchaOCR()` 会加载 ONNX Runtime Session，因此应在程序启动时创建一次并复用：

```python
from ocr import CaptchaOCR

ocr = CaptchaOCR(
    expected_length=6,
    min_confidence=0.90,
)


def recognize(image):
    return ocr.predict_detail(image)
```

不要每识别一张图片都重新创建 `CaptchaOCR()`。

## 支持的输入

### 本地图片

```python
text, confidence = ocr.predict(r"D:\test\captcha.png")
```

### HTTP 返回的 bytes

```python
import requests

response = requests.get("https://example.com/captcha", timeout=10)
response.raise_for_status()

text, confidence = ocr.predict(response.content)
```

### OpenCV ndarray

```python
import cv2

image = cv2.imread(r"D:\test\captcha.png")
text, confidence = ocr.predict(image)
```

### Base64 / Data URI

```python
text, confidence = ocr.predict_base64(base64_string)
```

支持：

```text
iVBORw0KGgo...
```

和：

```text
data:image/png;base64,iVBORw0KGgo...
```

## 带校验的结果

```python
result = ocr.predict_detail(image)

print(result.text)
print(result.confidence)
print(result.accepted)
```

`accepted=True` 需要同时满足：

- 识别长度符合 `expected_length`
- `confidence >= min_confidence`

## Batch 推理

模型 batch 维为动态维，可以一次识别多张图片：

```python
results = ocr.predict_batch([
    r"D:\test\1.png",
    r"D:\test\2.png",
    r"D:\test\3.png",
])

for text, confidence in results:
    print(text, confidence)
```

## 命令行使用

```powershell
python ocr.py D:\test\captcha.png
```

示例输出：

```json
{
  "text": "A7q3BT",
  "confidence": 0.997421,
  "accepted": true
}
```

查看模型信息：

```powershell
python ocr.py D:\test\captcha.png --info
```

设置最低置信度：

```powershell
python ocr.py D:\test\captcha.png --min-confidence 0.90
```

## API 文档

完整接口说明见：[`docs/API.md`](docs/API.md)

## 运行文件说明

真正部署时只需要核心运行文件：

```text
ocr.py
captcha_svtr_fp32.onnx
captcha_dict.txt
```

以及 Python 依赖：

```text
onnxruntime
opencv-python
numpy
```

## GitHub 上传注意事项

当前 ONNX 模型约 25 MB，低于 GitHub 单文件 100 MB 限制，可以直接提交，不需要 Git LFS。

建议不要把以下内容提交到仓库：

- 训练图片
- PaddleOCR 源码目录
- Python 虚拟环境
- 训练 checkpoint
- 登录 Cookie / Token / 密钥
- 公司内部接口地址或账号信息

这些内容已经在 `.gitignore` 中做了常见排除。
