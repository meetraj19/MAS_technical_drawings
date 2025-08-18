#!/usr/bin/env python3
"""
PDF Processing Tool for Technical Drawing Feedback System

This tool handles loading, processing, and conversion of technical drawings
from various formats (PDF, JPG, PNG) into standardized image data.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image, ImageEnhance
import pdf2image
import fitz  # PyMuPDF
import logging
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class ImageMetadata:
    """Metadata for processed images."""
    original_path: str
    processed_path: str
    width: int
    height: int
    dpi: int
    format: str
    file_size: int
    page_number: Optional[int] = None
    has_text_layer: bool = False
    preprocessing_applied: List[str] = None


@dataclass
class ProcessedImage:
    """Container for processed image data."""
    image_array: np.ndarray
    metadata: ImageMetadata
    text_content: Optional[str] = None
    confidence_scores: Optional[Dict] = None


class PDFProcessingTool:
    """Tool for processing PDFs and images for technical drawing analysis."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize PDF processing tool.
        
        Args:
            config: Configuration dictionary with processing parameters
        """
        self.config = config or self._get_default_config()
        self.supported_formats = ['.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.tif']
        
    def _get_default_config(self) -> Dict:
        """Get default configuration for processing."""
        return {
            'dpi': 300,
            'max_image_size': 4096,
            'output_format': 'RGB',
            'enhance_contrast': True,
            'enhance_sharpness': True,
            'normalize_brightness': True,
            'preserve_aspect_ratio': True,
            'quality': 95
        }
    
    def validate_input(self, file_path: Union[str, Path]) -> bool:
        """
        Validate input file format and existence.
        
        Args:
            file_path: Path to input file
            
        Returns:
            True if file is valid, False otherwise
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File does not exist: {file_path}")
            return False
            
        if file_path.suffix.lower() not in self.supported_formats:
            logger.error(f"Unsupported format: {file_path.suffix}")
            return False
            
        try:
            # Check if file is readable
            if file_path.suffix.lower() == '.pdf':
                doc = fitz.open(str(file_path))
                doc.close()
            else:
                img = Image.open(file_path)
                img.close()
            return True
        except Exception as e:
            logger.error(f"File validation failed for {file_path}: {e}")
            return False
    
    def extract_pdf_pages(self, pdf_path: Union[str, Path], 
                         dpi: Optional[int] = None) -> List[ProcessedImage]:
        """
        Extract pages from PDF as images.
        
        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for conversion (uses config default if None)
            
        Returns:
            List of ProcessedImage objects for each page
        """
        pdf_path = Path(pdf_path)
        dpi = dpi or self.config['dpi']
        processed_images = []
        
        try:
            # Use PyMuPDF for text extraction and pdf2image for image conversion
            doc = fitz.open(str(pdf_path))
            
            # Convert PDF to images
            logger.info(f"Converting PDF {pdf_path.name} at {dpi} DPI...")
            pil_images = pdf2image.convert_from_path(
                str(pdf_path),
                dpi=dpi,
                fmt='RGB'
            )
            
            for page_num, pil_image in enumerate(pil_images, 1):
                # Convert PIL to numpy array
                image_array = np.array(pil_image)
                
                # Extract text from this page
                page = doc[page_num - 1]
                text_content = page.get_text()
                has_text = bool(text_content.strip())
                
                # Create metadata
                metadata = ImageMetadata(
                    original_path=str(pdf_path),
                    processed_path=f"page_{page_num}",
                    width=image_array.shape[1],
                    height=image_array.shape[0],
                    dpi=dpi,
                    format='RGB',
                    file_size=image_array.nbytes,
                    page_number=page_num,
                    has_text_layer=has_text,
                    preprocessing_applied=[]
                )
                
                processed_image = ProcessedImage(
                    image_array=image_array,
                    metadata=metadata,
                    text_content=text_content if has_text else None
                )
                
                processed_images.append(processed_image)
                logger.info(f"Processed page {page_num}: {metadata.width}x{metadata.height}")
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            
        return processed_images
    
    def load_image(self, image_path: Union[str, Path]) -> Optional[ProcessedImage]:
        """
        Load and process a single image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            ProcessedImage object or None if loading fails
        """
        image_path = Path(image_path)
        
        try:
            # Load image with PIL first for metadata
            pil_image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(pil_image)
            
            # Get DPI info if available
            dpi = pil_image.info.get('dpi', (self.config['dpi'], self.config['dpi']))
            if isinstance(dpi, tuple):
                dpi = dpi[0]
            
            # Create metadata
            metadata = ImageMetadata(
                original_path=str(image_path),
                processed_path=str(image_path),
                width=image_array.shape[1],
                height=image_array.shape[0],
                dpi=int(dpi),
                format=pil_image.format or 'Unknown',
                file_size=image_path.stat().st_size,
                preprocessing_applied=[]
            )
            
            processed_image = ProcessedImage(
                image_array=image_array,
                metadata=metadata
            )
            
            logger.info(f"Loaded image {image_path.name}: {metadata.width}x{metadata.height}")
            return processed_image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def preprocess_image(self, processed_image: ProcessedImage) -> ProcessedImage:
        """
        Apply preprocessing to improve image quality for analysis.
        
        Args:
            processed_image: ProcessedImage object to preprocess
            
        Returns:
            ProcessedImage with preprocessing applied
        """
        image_array = processed_image.image_array.copy()
        preprocessing_steps = []
        
        try:
            # Resize if too large
            max_size = self.config['max_image_size']
            height, width = image_array.shape[:2]
            
            if max(height, width) > max_size:
                scale = max_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                image_array = cv2.resize(image_array, (new_width, new_height), 
                                       interpolation=cv2.INTER_LANCZOS4)
                preprocessing_steps.append(f"resized_to_{new_width}x{new_height}")
                logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
            
            # Convert to PIL for enhancement
            pil_image = Image.fromarray(image_array)
            
            # Enhance contrast if enabled
            if self.config.get('enhance_contrast', True):
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(1.2)  # Slight contrast boost
                preprocessing_steps.append("contrast_enhanced")
            
            # Enhance sharpness if enabled
            if self.config.get('enhance_sharpness', True):
                enhancer = ImageEnhance.Sharpness(pil_image)
                pil_image = enhancer.enhance(1.1)  # Slight sharpness boost
                preprocessing_steps.append("sharpness_enhanced")
            
            # Normalize brightness if enabled
            if self.config.get('normalize_brightness', True):
                enhancer = ImageEnhance.Brightness(pil_image)
                pil_image = enhancer.enhance(1.05)  # Slight brightness adjustment
                preprocessing_steps.append("brightness_normalized")
            
            # Convert back to numpy array
            image_array = np.array(pil_image)
            
            # Update metadata
            new_metadata = ImageMetadata(
                original_path=processed_image.metadata.original_path,
                processed_path=processed_image.metadata.processed_path,
                width=image_array.shape[1],
                height=image_array.shape[0],
                dpi=processed_image.metadata.dpi,
                format=processed_image.metadata.format,
                file_size=image_array.nbytes,
                page_number=processed_image.metadata.page_number,
                has_text_layer=processed_image.metadata.has_text_layer,
                preprocessing_applied=preprocessing_steps
            )
            
            return ProcessedImage(
                image_array=image_array,
                metadata=new_metadata,
                text_content=processed_image.text_content,
                confidence_scores=processed_image.confidence_scores
            )
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return processed_image
    
    def process_file(self, file_path: Union[str, Path], 
                    preprocess: bool = True) -> List[ProcessedImage]:
        """
        Process a file (PDF or image) into standardized format.
        
        Args:
            file_path: Path to file to process
            preprocess: Whether to apply preprocessing
            
        Returns:
            List of ProcessedImage objects
        """
        file_path = Path(file_path)
        
        if not self.validate_input(file_path):
            return []
        
        processed_images = []
        
        try:
            if file_path.suffix.lower() == '.pdf':
                # Process PDF
                images = self.extract_pdf_pages(file_path)
            else:
                # Process single image
                image = self.load_image(file_path)
                images = [image] if image else []
            
            # Apply preprocessing if requested
            if preprocess:
                for i, image in enumerate(images):
                    if image:
                        processed_images.append(self.preprocess_image(image))
                    else:
                        processed_images.append(image)
            else:
                processed_images = images
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            
        return processed_images
    
    def save_processed_image(self, processed_image: ProcessedImage, 
                           output_path: Union[str, Path], 
                           save_metadata: bool = True) -> bool:
        """
        Save processed image to disk.
        
        Args:
            processed_image: ProcessedImage to save
            output_path: Path to save image
            save_metadata: Whether to save metadata as JSON
            
        Returns:
            True if save successful, False otherwise
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save image
            pil_image = Image.fromarray(processed_image.image_array)
            pil_image.save(
                output_path,
                quality=self.config['quality'],
                optimize=True
            )
            
            # Save metadata if requested
            if save_metadata:
                metadata_path = output_path.with_suffix('.json')
                metadata_dict = {
                    'original_path': processed_image.metadata.original_path,
                    'processed_path': str(output_path),
                    'width': processed_image.metadata.width,
                    'height': processed_image.metadata.height,
                    'dpi': processed_image.metadata.dpi,
                    'format': processed_image.metadata.format,
                    'file_size': processed_image.metadata.file_size,
                    'page_number': processed_image.metadata.page_number,
                    'has_text_layer': processed_image.metadata.has_text_layer,
                    'preprocessing_applied': processed_image.metadata.preprocessing_applied,
                    'text_content': processed_image.text_content
                }
                
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved processed image to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving processed image to {output_path}: {e}")
            return False
    
    def batch_process(self, input_dir: Union[str, Path], 
                     output_dir: Union[str, Path],
                     file_pattern: str = "*") -> Dict[str, List[ProcessedImage]]:
        """
        Process all files in a directory.
        
        Args:
            input_dir: Directory containing input files
            output_dir: Directory to save processed images
            file_pattern: Pattern to match files (e.g., "*.pdf", "*.jpg")
            
        Returns:
            Dictionary mapping input file names to processed images
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # Find all matching files
        matching_files = []
        for ext in ['.pdf', '.jpg', '.jpeg', '.png', '.tiff']:
            matching_files.extend(input_dir.glob(f"*{ext}"))
            matching_files.extend(input_dir.glob(f"*{ext.upper()}"))
        
        logger.info(f"Found {len(matching_files)} files to process in {input_dir}")
        
        for file_path in matching_files:
            logger.info(f"Processing {file_path.name}...")
            
            processed_images = self.process_file(file_path)
            results[file_path.name] = processed_images
            
            # Save processed images
            for i, processed_image in enumerate(processed_images):
                if processed_image:
                    if len(processed_images) > 1:
                        # Multiple pages/images
                        output_name = f"{file_path.stem}_page_{i+1}.jpg"
                    else:
                        # Single image
                        output_name = f"{file_path.stem}_processed.jpg"
                    
                    output_path = output_dir / output_name
                    self.save_processed_image(processed_image, output_path)
        
        logger.info(f"Batch processing completed. Processed {len(results)} files.")
        return results


def main():
    """Test the PDF processing tool."""
    # Initialize tool
    tool = PDFProcessingTool()
    
    # Test with dataset images
    dataset_root = Path(__file__).parent.parent.parent / "dataset"
    corrected_dir = dataset_root / "corrected"
    
    if corrected_dir.exists():
        print(f"\nTesting PDF Processing Tool with dataset images...")
        
        # Process first few images as test
        image_files = list(corrected_dir.glob("*.jpg"))[:3]
        
        for image_file in image_files:
            print(f"\nProcessing {image_file.name}:")
            
            processed_images = tool.process_file(image_file)
            
            if processed_images:
                img = processed_images[0]
                print(f"  Dimensions: {img.metadata.width}x{img.metadata.height}")
                print(f"  DPI: {img.metadata.dpi}")
                print(f"  Preprocessing: {img.metadata.preprocessing_applied}")
                print(f"  File size: {img.metadata.file_size} bytes")
            else:
                print(f"  Failed to process")
    else:
        print("Dataset directory not found. Please run from project root.")


if __name__ == "__main__":
    main()