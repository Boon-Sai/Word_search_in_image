import cv2
import numpy as np
from paddleocr import PaddleOCR
import json
import os

class SparrowOCR:
    def __init__(self, lang='en'):
        """
        Initialize PaddleOCR with customized parameters for better corner and edge text detection.
        """
        self.ocr = PaddleOCR(
            use_textline_orientation=True,  # Enabled text line orientation detection
            lang=lang,  # Sets Language for OCR
            text_det_thresh=0.2,  # Lowered threshold for better edge detection
            text_det_box_thresh=0.4,  # Lowered threshold for bounding boxes
            text_det_limit_type='min',  # Updated from det_limit_type
            text_det_limit_side_len=1200  # Updated from det_limit_side_len
        )

    def preprocess_image(self, image_path):
        """
        Preprocess the image to enhance corner and edge text detectability.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")
        
        # Add padding to prevent edge text cropping
        padding = 20  # Pixels
        image = cv2.copyMakeBorder(
            image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        
        # Convert to grayscale and apply adaptive thresholding
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Enhance contrast in corners and bottom edge
        height, width = gray.shape
        corners = [
            gray[0:50, 0:50],  # Top-left
            gray[0:50, -50:],  # Top-right
            gray[-50:, 0:50],  # Bottom-left
            gray[-50:, -50:]   # Bottom-right
        ]
        bottom_edge = gray[-50:, :]  # Bottom 50 pixels across width
        for region in corners + [bottom_edge]:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            region[:] = clahe.apply(region)
        
        # Merge back and save preprocessed image for debugging
        image[:, :, 0] = gray  # Update blue channel with processed gray
        
        # Create outputs folder in the script's directory
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save preprocessed image in outputs folder
        preprocessed_path = os.path.join(output_dir, "preprocessed_" + os.path.basename(image_path))
        
        # Save and verify
        if not cv2.imwrite(preprocessed_path, image):
            raise ValueError(f"Failed to save preprocessed image at {preprocessed_path}")
        
        # Verify the file exists
        if not os.path.exists(preprocessed_path):
            raise ValueError(f"Preprocessed image not found at {preprocessed_path}")
            
        return image, preprocessed_path

    def extract_words_with_bboxes(self, image_path):
        """
        Extract individual words and their bounding boxes from an image.
        Returns a list of dictionaries with index, word, and bounding box coordinates.
        """
        # Preprocess the image
        image, preprocessed_path = self.preprocess_image(image_path)
        
        # Perform OCR on the preprocessed image
        result = self.ocr.predict(preprocessed_path)
        
        # Debug: Visualize raw OCR detections in outputs folder
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
        raw_ocr_path = os.path.join(output_dir, "raw_ocr_detections.png")
        self.visualize_raw_ocr(image_path, result, raw_ocr_path)
        
        # No need to clean up preprocessed_path since it's saved for debugging
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

    def visualize_raw_ocr(self, image_path, ocr_result, output_path):
        """
        Visualize raw OCR detections for debugging.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")
        
        # Add padding to match preprocessing
        padding = 20
        image = cv2.copyMakeBorder(
            image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        
        if not ocr_result or not isinstance(ocr_result, list):
            print("No OCR results to visualize.")
            cv2.imwrite(output_path, image)
            return
        
        for result_dict in ocr_result:
            if not isinstance(result_dict, dict):
                continue
            bboxes = result_dict.get('rec_polys', [])
            texts = result_dict.get('rec_texts', [])
            scores = result_dict.get('rec_scores', [])
            
            for bbox, text, score in zip(bboxes, texts, scores):
                # Convert polygon to rectangle for visualization
                x1 = int(min([point[0] for point in bbox]))
                y1 = int(min([point[1] for point in bbox]))
                x2 = int(max([point[0] for point in bbox]))
                y2 = int(max([point[1] for point in bbox]))
                
                # Draw rectangle
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Blue for raw detections
                # Add text and confidence
                label = f"{text} ({score:.2f})"
                cv2.putText(
                    image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 1
                )
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        cv2.imwrite(output_path, image)

    def save_results(self, word_data, output_path):
        """
        Save the word data (index, word, bbox) to a JSON file.
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(word_data, f, indent=4)

    def visualize_bboxes(self, image_path, word_data, output_image_path):
        """
        Draw bounding boxes on the image and save the result.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")
        
        # Add padding to match preprocessing
        padding = 20
        image = cv2.copyMakeBorder(
            image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        
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
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_image_path) or '.', exist_ok=True)
        cv2.imwrite(output_image_path, image)

# Example usage
if __name__ == "__main__":
    # Initialize SparrowOCR
    ocr = SparrowOCR(lang='en')

    # Input image path
    image_path = "data/ss-1.png"

    # Verify input image exists
    if not os.path.exists(image_path):
        raise ValueError(f"Input image not found at {image_path}")

    # Extract words and bounding boxes
    word_data = ocr.extract_words_with_bboxes(image_path)

    # Print results
    for item in word_data:
        print(f"Index: {item['index']}, Word: {item['word']}, BBox: {item['bbox']}, Confidence: {item['confidence']:.2f}")

    # Save results to JSON
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
    output_json_path = os.path.join(output_dir, "word_bboxes.json")
    ocr.save_results(word_data, output_json_path)

    # Visualize bounding boxes and save the output image
    output_image_path = os.path.join(output_dir, "output_image_with_bboxes.png")
    ocr.visualize_bboxes(image_path, word_data, output_image_path)