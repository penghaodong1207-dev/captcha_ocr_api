# CAPTCHA OCR Dataset

## Dataset Summary

- Images: 1079
- Image size: 150 × 50
- Label length: 6
- Character classes: 33
- Total labeled characters: 6474

## Character Set

`23456789ABDEFGHJLNQRTabdefghjnqrt`

Each image filename is its ground-truth label.

Example:

`A7q3BT.png`

means:

`A7q3BT`

## Split

The repository contains the exact split used for the published model:

- Train: dataset/splits/train.txt
- Validation: dataset/splits/val.txt
- Test: dataset/splits/test.txt

Do not regenerate the split when reproducing the reported benchmark.

## Image Format

- PNG
- Resolution: 150 × 50
- RGB/BGR 3-channel input
- Fixed 6-character sequence

## Benchmark

SVTR-Tiny + CTC:

- Validation sequence accuracy: 94.23%
- Test sequence accuracy: 95.19%
- Test normalized edit distance: 0.99199