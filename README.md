
---

````markdown
# SparrowOCR

SparrowOCR is a Python tool that uses PaddleOCR to extract individual words along with their bounding boxes from images. It also allows saving the output as a JSON file and visualizing the bounding boxes on the image.

## Features

- Extract individual words with bounding boxes
- Save results to a JSON file
- Visualize word-level bounding boxes on the image
- Built using PaddleOCR and OpenCV

## Installation

### Clone the repository

```bash
git clone https://github.com/yourusername/sparrowocr.git
cd sparrowocr
````

### Install required packages

```bash
pip install paddleocr opencv-python numpy
```

### Install PaddlePaddle

Make sure you install a compatible version of PaddlePaddle. You can install the CPU version using:

```bash
pip install paddlepaddle -f https://paddlepaddle.org.cn/whl/mkl/avx/stable.html
```

For GPU support, refer to the official [PaddlePaddle installation guide](https://www.paddlepaddle.org.cn/install/quick).

## Usage

### Step 1: Place your image

Place the image you want to process in the same directory as the script and update the `image_path` variable in the `__main__` block.

Example:

```python
image_path = "input_image.jpg"
```

### Step 2: Run the script

```bash
python sparrow_ocr.py
```

This will:

* Extract words and their bounding boxes from the image
* Save the word data in `word_bboxes.json`
* Save a visualized image with bounding boxes as `output_image_with_bboxes.png`

## Output Files

### word\_bboxes.json

Example structure:

```json
[
    {
        "index": 0,
        "word": "example",
        "bbox": [100.5, 50.0, 150.7, 75.2],
        "confidence": 0.98
    },
    ...
]
```

### output\_image\_with\_bboxes.png

An image with green rectangles drawn around each word and word labels annotated above each box.

## Project Structure

```
.
├── sparrow_ocr.py
├── input_image.jpg
├── word_bboxes.json
└── output_image_with_bboxes.png
```

## Notes

* Bounding boxes for words are approximated based on the full line's bounding box and word length.
* The script currently assumes horizontal English text by default (`lang='en'`).
* For multilingual support, modify the `lang` parameter in the `SparrowOCR` class.

## License

This project is licensed under the MIT License.
