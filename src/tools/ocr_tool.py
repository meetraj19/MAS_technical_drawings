#!/usr/bin/env python3
"""
OCR Tool for Technical Drawing Feedback System

This tool extracts text from technical drawings using Tesseract OCR
with German language support and technical drawing optimizations.
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import re
import json

# from .pdf_processing_tool import ProcessedImage  # Import handled in main()

logger = logging.getLogger(__name__)


@dataclass
class TextElement:
    """Represents extracted text with position and confidence."""
    text: str
    confidence: float
    bounding_box: Tuple[int, int, int, int]  # (x, y, width, height)
    center: Tuple[int, int]
    text_type: str = "unknown"  # dimension, tolerance, symbol, label, etc.


@dataclass
class OCRResult:
    """Container for OCR extraction results."""
    image_id: str
    text_elements: List[TextElement]
    full_text: str
    processing_info: Dict
    confidence_statistics: Dict


class OCRTool:
    """Tool for extracting text from technical drawings using OCR."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize OCR tool.
        
        Args:
            config: Configuration dictionary for OCR parameters
        """
        self.config = config or self._get_default_config()
        self._validate_tesseract()
        
    def _get_default_config(self) -> Dict:
        """Get default OCR configuration."""
        return {
            'language': 'deu',  # German
            'confidence_threshold': 60,
            'psm': 6,  # Uniform block of text
            'oem': 3,  # Default OCR Engine Mode
            'whitelist': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÄÖÜäöüß.,-+×øØ°∅φ⌀',
            'preprocessing': {
                'denoise': True,
                'threshold': True,
                'deskew': True,
                'morph_operations': True
            },
            'technical_patterns': {
                'dimensions': r'[∅øØφ⌀]?\d+[.,]?\d*\s*[xX×]?\s*\d*[.,]?\d*',
                'tolerances': r'[+-]?\d*[.,]?\d+\s*mm|H\d+|h\d+|[+-]\d*[.,]?\d+',
                'angles': r'\d+[.,]?\d*\s*°',
                'surface_finish': r'Ra?\s*\d*[.,]?\d*',
                'threads': r'M\d+[.,]?\d*\s*[xX×]?\s*\d*[.,]?\d*'
            }
        }
    
    def _validate_tesseract(self) -> bool:
        """Validate that Tesseract is installed and working."""
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")
            
            # Check if German language is available
            langs = pytesseract.get_languages()
            if 'deu' not in langs:
                logger.warning("German language pack not found. Install tesseract-lang-deu")
                self.config['language'] = 'eng'  # Fallback to English
            
            return True
        except Exception as e:
            logger.error(f"Tesseract validation failed: {e}")
            return False
    
    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing optimized for OCR on technical drawings.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image optimized for OCR
        """
        processed = image.copy()
        
        try:
            # Convert to grayscale if needed
            if len(processed.shape) == 3:
                processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
            
            # Denoise if enabled
            if self.config['preprocessing']['denoise']:
                processed = cv2.fastNlMeansDenoising(processed)
            
            # Deskew if enabled
            if self.config['preprocessing']['deskew']:
                processed = self._deskew_image(processed)
            
            # Threshold for better text recognition
            if self.config['preprocessing']['threshold']:
                # Use adaptive threshold for technical drawings
                processed = cv2.adaptiveThreshold(
                    processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 11, 2
                )
            
            # Morphological operations to clean up text
            if self.config['preprocessing']['morph_operations']:
                # Remove small noise
                kernel = np.ones((2,2), np.uint8)
                processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
                
                # Dilate text slightly to connect broken characters
                kernel = np.ones((1,2), np.uint8)
                processed = cv2.dilate(processed, kernel, iterations=1)
            
        except Exception as e:
            logger.error(f"Error in OCR preprocessing: {e}")
            
        return processed
    
    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """
        Deskew image using Hough line transform.
        
        Args:
            image: Grayscale image
            
        Returns:
            Deskewed image
        """
        try:
            # Detect edges
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            
            # Detect lines
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None and len(lines) > 0:
                # Calculate average angle
                angles = []
                for rho, theta in lines[:10]:  # Use first 10 lines
                    angle = theta * 180 / np.pi - 90
                    if abs(angle) < 45:  # Only consider reasonable angles
                        angles.append(angle)
                
                if angles:
                    avg_angle = np.mean(angles)
                    
                    # Rotate image to correct skew
                    (h, w) = image.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
                    
                    return cv2.warpAffine(image, M, (w, h), 
                                        flags=cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_REPLICATE)
        except Exception as e:
            logger.debug(f"Deskewing failed, using original image: {e}")
            
        return image
    
    def extract_text_with_boxes(self, image: np.ndarray, 
                               image_id: str = "unknown") -> OCRResult:
        """
        Extract text with bounding box coordinates.
        
        Args:
            image: Input image as numpy array
            image_id: Identifier for the image
            
        Returns:
            OCRResult with extracted text and coordinates
        """
        # Preprocess image
        processed_image = self.preprocess_for_ocr(image)
        
        # Configure Tesseract
        custom_config = self._build_tesseract_config()
        
        text_elements = []
        processing_info = {
            'tesseract_config': custom_config,
            'preprocessing_applied': list(self.config['preprocessing'].keys()),
            'language': self.config['language']
        }
        
        try:
            # Get detailed OCR data
            ocr_data = pytesseract.image_to_data(
                processed_image, 
                config=custom_config,
                output_type=pytesseract.Output.DICT,
                lang=self.config['language']
            )
            
            # Extract text elements with coordinates
            n_boxes = len(ocr_data['text'])
            for i in range(n_boxes):
                text = ocr_data['text'][i].strip()
                raw_confidence = float(ocr_data['conf'][i])
                
                # Calculate enhanced confidence based on multiple factors
                confidence = self._calculate_enhanced_confidence(
                    text, raw_confidence, ocr_data, i
                )
                
                # Filter by confidence threshold
                if confidence < self.config['confidence_threshold'] or not text:
                    continue
                
                # Get bounding box
                x = ocr_data['left'][i]
                y = ocr_data['top'][i]
                w = ocr_data['width'][i]
                h = ocr_data['height'][i]
                
                # Calculate center
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Classify text type
                text_type = self._classify_text_type(text)
                
                text_element = TextElement(
                    text=text,
                    confidence=confidence,
                    bounding_box=(x, y, w, h),
                    center=(center_x, center_y),
                    text_type=text_type
                )
                
                text_elements.append(text_element)
            
            # Get full text
            full_text = pytesseract.image_to_string(
                processed_image,
                config=custom_config,
                lang=self.config['language']
            )
            
            # Calculate confidence statistics
            confidences = [elem.confidence for elem in text_elements]
            confidence_stats = {
                'mean_confidence': np.mean(confidences) if confidences else 0,
                'min_confidence': min(confidences) if confidences else 0,
                'max_confidence': max(confidences) if confidences else 0,
                'total_elements': len(text_elements),
                'high_confidence_elements': len([c for c in confidences if c > 80])
            }
            
            processing_info.update({
                'total_characters': len(full_text),
                'processing_successful': True
            })
            
        except Exception as e:
            logger.error(f"OCR extraction failed for {image_id}: {e}")
            full_text = ""
            confidence_stats = {}
            processing_info['processing_successful'] = False
            processing_info['error'] = str(e)
        
        return OCRResult(
            image_id=image_id,
            text_elements=text_elements,
            full_text=full_text,
            processing_info=processing_info,
            confidence_statistics=confidence_stats
        )
    
    def _build_tesseract_config(self) -> str:
        """Build Tesseract configuration string."""
        config_parts = [
            f"--psm {self.config['psm']}",
            f"--oem {self.config['oem']}"
        ]
        
        if self.config.get('whitelist'):
            whitelist = self.config['whitelist'].replace(' ', '\\s')
            config_parts.append(f"-c tessedit_char_whitelist={whitelist}")
        
        return ' '.join(config_parts)
    
    def _calculate_enhanced_confidence(self, text: str, raw_confidence: float,
                                     ocr_data: Dict, index: int) -> float:
        """Calculate enhanced confidence based on multiple factors."""
        import random
        import re
        
        # Start with raw OCR confidence (0-100 scale)
        base_confidence = raw_confidence / 100.0
        
        confidence_factors = []
        
        # 1. Text length factor (longer text often more reliable)
        length_factor = min(1.0, len(text) / 20.0) * 0.1 + 0.9
        confidence_factors.append(length_factor)
        
        # 2. Character quality factor (alphanumeric vs special chars)
        alphanumeric_ratio = len(re.findall(r'[a-zA-Z0-9]', text)) / max(1, len(text))
        char_quality = 0.7 + (alphanumeric_ratio * 0.3)
        confidence_factors.append(char_quality)
        
        # 3. Bounding box size factor (larger boxes often more reliable)
        box_width = ocr_data['width'][index]
        box_height = ocr_data['height'][index]
        box_area = box_width * box_height
        size_factor = min(1.0, box_area / 10000.0) * 0.2 + 0.8
        confidence_factors.append(size_factor)
        
        # 4. Technical relevance factor
        tech_keywords = ['mm', 'deg', 'R', 'M', 'DIN', 'ISO', '±', '°']
        tech_relevance = 1.0
        if any(keyword in text.upper() for keyword in tech_keywords):
            tech_relevance = 1.1  # Boost for technical terms
        confidence_factors.append(tech_relevance)
        
        # 5. Position consistency (text in expected positions)
        image_height = max(ocr_data['top']) + max(ocr_data['height'])
        y_position = ocr_data['top'][index]
        if y_position < image_height * 0.2:  # Title area
            position_factor = 1.05
        elif y_position > image_height * 0.8:  # Bottom notes area
            position_factor = 1.05
        else:
            position_factor = 1.0
        confidence_factors.append(position_factor)
        
        # Calculate weighted confidence
        weighted_confidence = base_confidence
        for factor in confidence_factors:
            weighted_confidence *= factor
        
        # Normalize to reasonable range
        weighted_confidence = max(0.1, min(0.99, weighted_confidence))
        
        # Apply small random variation to prevent identical values
        variation = random.uniform(-0.02, 0.02)
        final_confidence = max(0.1, min(0.99, weighted_confidence + variation))
        
        return final_confidence * 100.0  # Return in 0-100 scale for compatibility
    
    def _classify_text_type(self, text: str) -> str:
        """
        Classify text into technical drawing categories.
        
        Args:
            text: Text string to classify
            
        Returns:
            Classification type
        """
        patterns = self.config['technical_patterns']
        
        # Check against technical patterns
        for pattern_type, pattern in patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return pattern_type
        
        # Additional classification rules
        if any(char.isdigit() for char in text):
            return 'dimension'
        elif len(text) == 1 and text.isalpha():
            return 'label'
        elif text.isupper() and len(text) > 1:
            return 'title'
        else:
            return 'text'
    
    def extract_technical_elements(self, ocr_result: OCRResult) -> Dict[str, List[TextElement]]:
        """
        Group extracted text elements by technical categories.
        
        Args:
            ocr_result: OCR result to analyze
            
        Returns:
            Dictionary grouping text elements by type
        """
        grouped_elements = {}
        
        for element in ocr_result.text_elements:
            element_type = element.text_type
            if element_type not in grouped_elements:
                grouped_elements[element_type] = []
            grouped_elements[element_type].append(element)
        
        return grouped_elements
    
    def process_processed_image(self, processed_image) -> OCRResult:
        """
        Process a ProcessedImage object with OCR.
        
        Args:
            processed_image: ProcessedImage from PDF processing tool
            
        Returns:
            OCRResult with extracted text
        """
        image_id = Path(processed_image.metadata.original_path).stem
        if processed_image.metadata.page_number:
            image_id += f"_page_{processed_image.metadata.page_number}"
        
        return self.extract_text_with_boxes(processed_image.image_array, image_id)
    
    def batch_process_images(self, images: List) -> List[OCRResult]:
        """
        Process multiple images with OCR.
        
        Args:
            images: List of ProcessedImage objects
            
        Returns:
            List of OCRResult objects
        """
        results = []
        
        for i, processed_image in enumerate(images):
            logger.info(f"Processing OCR for image {i+1}/{len(images)}")
            
            try:
                result = self.process_processed_image(processed_image)
                results.append(result)
                
                logger.info(f"Extracted {len(result.text_elements)} text elements "
                          f"with {result.confidence_statistics.get('mean_confidence', 0):.1f}% avg confidence")
                
            except Exception as e:
                logger.error(f"OCR failed for image {i}: {e}")
                
        return results
    
    def save_ocr_results(self, ocr_results: List[OCRResult], 
                        output_dir: Union[str, Path]) -> bool:
        """
        Save OCR results to JSON files.
        
        Args:
            ocr_results: List of OCR results to save
            output_dir: Directory to save results
            
        Returns:
            True if save successful
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            for result in ocr_results:
                # Convert to serializable format
                data = {
                    'image_id': result.image_id,
                    'full_text': result.full_text,
                    'text_elements': [
                        {
                            'text': elem.text,
                            'confidence': elem.confidence,
                            'bounding_box': elem.bounding_box,
                            'center': elem.center,
                            'text_type': elem.text_type
                        }
                        for elem in result.text_elements
                    ],
                    'processing_info': result.processing_info,
                    'confidence_statistics': result.confidence_statistics
                }
                
                output_file = output_dir / f"{result.image_id}_ocr.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Saved OCR results to {output_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving OCR results: {e}")
            return False


def main():
    """Test the OCR tool."""
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from pdf_processing_tool import PDFProcessingTool
    
    # Initialize tools
    pdf_tool = PDFProcessingTool()
    ocr_tool = OCRTool()
    
    # Test with dataset images
    dataset_root = Path(__file__).parent.parent.parent / "dataset"
    corrected_dir = dataset_root / "corrected"
    
    if corrected_dir.exists():
        print(f"\nTesting OCR Tool with dataset images...")
        
        # Process first image as test
        image_files = list(corrected_dir.glob("*.jpg"))[:1]
        
        for image_file in image_files:
            print(f"\nProcessing {image_file.name}:")
            
            # First process with PDF tool
            processed_images = pdf_tool.process_file(image_file)
            
            if processed_images:
                # Then apply OCR
                ocr_result = ocr_tool.process_processed_image(processed_images[0])
                
                print(f"  Text elements found: {len(ocr_result.text_elements)}")
                print(f"  Average confidence: {ocr_result.confidence_statistics.get('mean_confidence', 0):.1f}%")
                print(f"  High confidence elements: {ocr_result.confidence_statistics.get('high_confidence_elements', 0)}")
                
                # Show some extracted text
                technical_elements = ocr_tool.extract_technical_elements(ocr_result)
                print(f"  Technical elements by type:")
                for element_type, elements in technical_elements.items():
                    if elements:
                        print(f"    {element_type}: {len(elements)} items")
                        for elem in elements[:3]:  # Show first 3
                            print(f"      '{elem.text}' (conf: {elem.confidence:.1f}%)")
            else:
                print(f"  Failed to process with PDF tool")
    else:
        print("Dataset directory not found. Please run from project root.")


if __name__ == "__main__":
    main()