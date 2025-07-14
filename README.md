

````markdown
# SparrowOCR - Word-Level OCR with Bounding Boxes using PaddleOCR

SparrowOCR is a lightweight Python tool that uses [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) to extract **words** and their **bounding boxes** from images. It also supports saving the results to a JSON file and visualizing the bounding boxes on the image.

---

## Features

- Extract words with bounding box coordinates
- Save results as JSON
- Visualize word bounding boxes on the original image
- Lightweight and easy to extend

---

## Installation

1. **Clone the repo:**

```bash
git clone https://github.com/yourusername/sparrowocr.git
cd sparrowocr
````

2. **Install dependencies:**

Make sure you have Python 3.8+ installed. Then run:

```bash
pip install paddleocr opencv-python numpy
```

Also install Paddle dependencies:

```bash
pip install paddlepaddle -f https://paddlepaddle.org.cn/whl/mkl/avx/stable.html
```

---

## Usage

### 1. Prepare your image

Place your image in the root directory and update the `image_path` variable in the script.

```python
image_path = "input_image.jpg"
```

### 2. Run the script

```bash
python your_script_name.py
```

It will:

* Extract words and bounding boxes
* Save a JSON file: `word_bboxes.json`
* Save a visualized image: `output_image_with_bboxes.png`

---

## Output Format

### `word_bboxes.json`

```json
[
    {
        "index": 0,
        "word": "Hello",
        "bbox": [100.5, 50.0, 150.7, 75.2],
        "confidence": 0.98
    },
    ...
]
```

### `output_image_with_bboxes.png`

An image with green rectangles drawn around each word, along with word labels.

---

## Project Structure

```
.
├── sparrow_ocr.py
├── input_image.jpg
├── word_bboxes.json
└── output_image_with_bboxes.png
```

---

## Notes

* This script uses a simple estimation to split lines into individual word boxes.
* Works best with clearly printed text and good image quality.
* Extendable for paragraph-level structure or multilingual support (`lang='ch'`, `lang='en'`, etc.).

---

## Author

**Your Name**
Feel free to use, fork, or contribute!

---

## License

This project is open-source and available under the [MIT License](LICENSE).
