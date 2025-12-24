import pytesseract
import cv2
import numpy as np
import os
import logging
from typing import Dict, Any, List
from pathlib import Path

# Configure module-level logger
logger = logging.getLogger(__name__)

# Tesseract configuration
# On Ubuntu/Linux, tesseract is usually in the PATH.
# We allow overriding via environment variable for flexibility.
TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

def extract_text(image_path: str) -> Dict[str, Any]:
    """
    Reads an image and returns the text found on it using Tesseract OCR.
    
    Args:
        image_path (str): The absolute or relative path to the image file.
        
    Returns:
        Dict[str, Any]: A dictionary containing:
            - status (str): 'success' or 'error'
            - raw_text (str): The full text extracted (if success)
            - extracted_data (List[str]): List of non-empty lines (if success)
            - message (str): Error message (if error)
    """
    path = Path(image_path)
    
    if not path.exists():
        logger.error(f"Image file not found at path: {image_path}")
        return {"status": "error", "message": f"Image file not found: {image_path}"}

    try:
        # Load the image using OpenCV
        img = cv2.imread(str(path))

        if img is None:
            logger.error(f"OpenCV failed to load image at path: {image_path}")
            return {"status": "error", "message": "Could not read image file. Ensure it is a valid image format."}

        # Pre-processing pipeline
        # 1. Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Thresholding
        # Using Otsu's binarization for automatic thresholding which is generally more robust
        # than a fixed value for varying lighting conditions.
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Extract Text
        # psm 3: Fully automatic page segmentation, but no OSD. (Default)
        custom_config = r'--oem 3 --psm 3' 
        text = pytesseract.image_to_string(thresh, config=custom_config)

        # Clean up the text (remove empty lines and whitespace)
        clean_text: List[str] = [line.strip() for line in text.split('\n') if line.strip()]

        logger.info(f"OCR extraction successful for {image_path}")
        
        return {
            "status": "success",
            "raw_text": text,
            "extracted_data": clean_text
        }

    except pytesseract.TesseractNotFoundError:
        error_msg = "Tesseract is not installed or not in PATH. Please install tesseract-ocr."
        logger.critical(error_msg)
        return {"status": "error", "message": error_msg}
        
    except Exception as e:
        logger.exception(f"Unexpected error during OCR extraction: {str(e)}")
        return {"status": "error", "message": f"An unexpected error occurred: {str(e)}"}