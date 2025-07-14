import cv2
import numpy as np
from paddleocr import PaddleOCR
import json
import os

class SparrowOCR:
    def __init__(self, lang='en'):
        """
        Initialize PaddleOCR with the specified language.
        """
        self.ocr = PaddleOCR(use_textline_orientation=True, lang=lang)

    def extract_words_with_bboxes(self, image_path):
        """
        Extract individual words and their bounding boxes from an image.
        Returns a list of dictionaries with index, word, and bounding box coordinates.
        """
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")

        # Perform OCR to get text lines and their bounding boxes
        result = self.ocr.predict(image_path)

        # Debug: Print the raw result to inspect its format
        print("Raw OCR result:", result)

        # Initialize output list
        word_data = []
        word_index = 0

        # Check if result is valid
        if not result or not isinstance(result, list):
            print("No valid OCR result returned.")
            return word_data

        # Process the first dictionary in the result (assuming single-page image)
        for result_dict in result:
            if not isinstance(result_dict, dict):
                print(f"Skipping invalid result format: {result_dict}")
                continue

            # Extract text, bounding boxes, and confidence scores
            texts = result_dict.get('rec_texts', [])
            bboxes = result_dict.get('rec_polys', [])
            scores = result_dict.get('rec_scores', [])

            # Ensure lengths match
            if not (len(texts) == len(bboxes) == len(scores)):
                print("Mismatch in text, bbox, or score lengths:", len(texts), len(bboxes), len(scores))
                continue

            # Process each text region
            for text, bbox, confidence in zip(texts, bboxes, scores):
                # Debug: Print each text region
                print(f"Processing text: {text}, BBox: {bbox}, Confidence: {confidence}")

                # Split text into individual words
                words = text.split()

                # Calculate approximate bounding boxes for each word
                if words:
                    # Convert bounding box to [x1, y1, x2, y2] format
                    x1 = min([point[0] for point in bbox])
                    y1 = min([point[1] for point in bbox])
                    x2 = max([point[0] for point in bbox])
                    y2 = max([point[1] for point in bbox])
                    line_bbox = [x1, y1, x2, y2]

                    # Estimate width per character (including spaces)
                    line_width = x2 - x1
                    total_chars = len(text)  # Including spaces
                    char_width = line_width / total_chars if total_chars > 0 else line_width

                    # Track current x-position for word bounding boxes
                    current_x = x1
                    char_index = 0

                    for word in words:
                        # Calculate word width based on character count
                        word_length = len(word)
                        word_width = word_length * char_width

                        # Create bounding box for the word
                        word_bbox = [
                            float(current_x),  # x1
                            float(y1),        # y1 (same as line)
                            float(current_x + word_width),  # x2
                            float(y2)         # y2 (same as line)
                        ]

                        # Add to output
                        word_data.append({
                            "index": word_index,
                            "word": word,
                            "bbox": [round(coord, 2) for coord in word_bbox],  # Round for cleaner JSON
                            "confidence": float(confidence)  # Convert to Python float
                        })

                        # Update index and x-position
                        word_index += 1
                        current_x += word_width
                        char_index += word_length

                        # Account for space between words
                        if char_index < len(text) and text[char_index] == ' ':
                            current_x += char_width
                            char_index += 1

        return word_data

    def save_results(self, word_data, output_path):
        """
        Save the word data (index, word, bbox) to a JSON file.
        """
        with open(output_path, 'w') as f:
            json.dump(word_data, f, indent=4)

    def visualize_bboxes(self, image_path, word_data, output_image_path):
        """
        Draw bounding boxes on the image and save the result.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")
        for item in word_data:
            index = item["index"]
            word = item["word"]
            bbox = item["bbox"]
            # Draw rectangle
            cv2.rectangle(
                image,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                (0, 255, 0),  # Green color
                1
            )
            # Add index and word label
            label = f"{index}: {word}"
            cv2.putText(
                image,
                label,
                (int(bbox[0]), int(bbox[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),  # Red text
                1
            )
        cv2.imwrite(output_image_path, image)

# Example usage
if __name__ == "__main__":
    # Initialize SparrowOCR
    ocr = SparrowOCR(lang='en')

    # Input image path
    image_path = "input_image.jpg"  # Matches your input from the error output

    # Extract words and bounding boxes
    word_data = ocr.extract_words_with_bboxes(image_path)

    # Print results
    for item in word_data:
        print(f"Index: {item['index']}, Word: {item['word']}, BBox: {item['bbox']}, Confidence: {item['confidence']:.2f}")

    # Save results to JSON
    output_json_path = "word_bboxes.json"
    ocr.save_results(word_data, output_json_path)

    # Visualize bounding boxes and save the output image
    output_image_path = "output_image_with_bboxes.png"
    ocr.visualize_bboxes(image_path, word_data, output_image_path)