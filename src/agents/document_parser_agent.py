#!/usr/bin/env python3
"""
Document Parser Agent for Technical Drawing Feedback System

This CrewAI agent extracts all content from technical drawings by integrating
PDF processing, OCR, and image analysis tools into a unified pipeline.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import json

# Add tools to path
sys.path.append(str(Path(__file__).parent.parent))

from tools.pdf_processing_tool import PDFProcessingTool, ProcessedImage
from tools.ocr_tool import OCRTool, OCRResult
from tools.image_analysis_tool import ImageAnalysisTool, AnalysisResult

logger = logging.getLogger(__name__)


class DocumentParserAgent:
    """
    CrewAI Agent for parsing technical drawings.
    
    Role: Extract all content from technical drawings
    Goal: Convert drawings into structured, analyzable data
    Tools: PDF Processing, OCR, Image Analysis
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Document Parser Agent.
        
        Args:
            config: Configuration dictionary for the agent
        """
        self.config = config or self._get_default_config()
        
        # Initialize tools
        self.pdf_tool = PDFProcessingTool(self.config.get('pdf_processing', {}))
        self.ocr_tool = OCRTool(self.config.get('ocr', {}))
        self.analysis_tool = ImageAnalysisTool(self.config.get('image_analysis', {}))
        
        # Agent metadata
        self.role = "Technical Drawing Document Parser"
        self.goal = "Extract and structure all content from technical drawings"
        self.backstory = """You are an expert technical drawing analyst specializing in 
        German engineering drawings. You can process PDFs, extract text with high accuracy, 
        and identify geometric elements in technical drawings."""
        
        logger.info(f"Initialized {self.role}")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for the agent."""
        return {
            'processing': {
                'max_concurrent_pages': 4,
                'preserve_intermediate_results': True,
                'quality_threshold': 0.7
            },
            'output': {
                'include_raw_data': False,
                'include_confidence_scores': True,
                'format_version': '1.0'
            },
            'pdf_processing': {
                'dpi': 300,
                'max_image_size': 4096,
                'enhance_contrast': True,
                'enhance_sharpness': True,
                'normalize_brightness': True,
                'preserve_aspect_ratio': True,
                'quality': 95
            },
            'ocr': {
                'language': 'deu',
                'confidence_threshold': 60,
                'psm': 6,
                'oem': 3,
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
            },
            'image_analysis': {
                'line_detection': {
                    'canny_threshold1': 50,
                    'canny_threshold2': 150,
                    'hough_threshold': 100,
                    'min_line_length': 30,
                    'max_line_gap': 10
                }
            }
        }
    
    def execute(self, input_file: str, **kwargs) -> Dict[str, Any]:
        """
        Main execution method for the agent.
        
        Args:
            input_file: Path to the technical drawing file
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing parsed document structure
        """
        logger.info(f"Document Parser Agent executing on: {input_file}")
        
        try:
            # Step 1: Process the input file
            processed_images = self._process_input_file(input_file)
            
            if not processed_images:
                return self._create_error_result("Failed to process input file", input_file)
            
            # Step 2: Extract text content
            ocr_results = self._extract_text_content(processed_images)
            
            # Step 3: Analyze visual elements
            analysis_results = self._analyze_visual_elements(processed_images)
            
            # Step 4: Assess drawing quality
            quality_assessment = self._assess_drawing_quality(
                processed_images, ocr_results, analysis_results
            )
            
            # Step 5: Combine and structure results
            document_structure = self._create_document_structure(
                input_file, processed_images, ocr_results, analysis_results, quality_assessment
            )
            
            logger.info(f"Document parsing completed successfully for {input_file}")
            return document_structure
            
        except Exception as e:
            logger.error(f"Document Parser Agent failed: {e}")
            return self._create_error_result(str(e), input_file)
    
    def _process_input_file(self, input_file: str) -> List[ProcessedImage]:
        """
        Process input file using PDF processing tool.
        
        Args:
            input_file: Path to input file
            
        Returns:
            List of processed images
        """
        logger.info("Step 1: Processing input file...")
        
        try:
            processed_images = self.pdf_tool.process_file(
                input_file, 
                preprocess=True
            )
            
            logger.info(f"Processed {len(processed_images)} image(s) from {input_file}")
            
            # Quality check
            quality_threshold = self.config['processing']['quality_threshold']
            valid_images = []
            
            for img in processed_images:
                if img and img.image_array is not None:
                    # Simple quality check based on image size and content
                    height, width = img.image_array.shape[:2]
                    if height > 100 and width > 100:  # Minimum reasonable size
                        valid_images.append(img)
                    else:
                        logger.warning(f"Image too small: {width}x{height}")
                else:
                    logger.warning("Invalid image in processed results")
            
            logger.info(f"Quality check passed: {len(valid_images)}/{len(processed_images)} images")
            return valid_images
            
        except Exception as e:
            logger.error(f"Input file processing failed: {e}")
            return []
    
    def _extract_text_content(self, processed_images: List[ProcessedImage]) -> List[OCRResult]:
        """
        Extract text content using OCR tool.
        
        Args:
            processed_images: List of processed images
            
        Returns:
            List of OCR results
        """
        logger.info("Step 2: Extracting text content...")
        
        ocr_results = []
        
        for i, processed_image in enumerate(processed_images):
            try:
                logger.info(f"Processing OCR for image {i+1}/{len(processed_images)}")
                
                ocr_result = self.ocr_tool.process_processed_image(processed_image)
                
                if ocr_result:
                    ocr_results.append(ocr_result)
                    
                    # Log OCR statistics
                    stats = ocr_result.confidence_statistics
                    logger.info(f"OCR completed: {len(ocr_result.text_elements)} elements, "
                              f"avg confidence: {stats.get('mean_confidence', 0):.1f}%")
                else:
                    logger.warning(f"OCR failed for image {i+1}")
                    
            except Exception as e:
                logger.error(f"OCR processing failed for image {i+1}: {e}")
        
        logger.info(f"Text extraction completed: {len(ocr_results)} results")
        return ocr_results
    
    def _analyze_visual_elements(self, processed_images: List[ProcessedImage]) -> List[AnalysisResult]:
        """
        Analyze visual elements using image analysis tool.
        
        Args:
            processed_images: List of processed images
            
        Returns:
            List of analysis results
        """
        logger.info("Step 3: Analyzing visual elements...")
        
        analysis_results = []
        
        for i, processed_image in enumerate(processed_images):
            try:
                logger.info(f"Analyzing visual elements for image {i+1}/{len(processed_images)}")
                
                image_id = f"{processed_image.metadata.original_path}_page_{i+1}"
                
                analysis_result = self.analysis_tool.analyze_image(
                    processed_image.image_array,
                    image_id
                )
                
                if analysis_result:
                    analysis_results.append(analysis_result)
                    
                    # Log analysis statistics
                    total_elements = (len(analysis_result.lines) + 
                                    len(analysis_result.circles) + 
                                    len(analysis_result.rectangles) + 
                                    len(analysis_result.dimension_lines) + 
                                    len(analysis_result.cross_hatches))
                    
                    logger.info(f"Visual analysis completed: {total_elements} elements detected")
                    logger.info(f"  Lines: {len(analysis_result.lines)}, "
                              f"Circles: {len(analysis_result.circles)}, "
                              f"Rectangles: {len(analysis_result.rectangles)}")
                else:
                    logger.warning(f"Visual analysis failed for image {i+1}")
                    
            except Exception as e:
                logger.error(f"Visual analysis failed for image {i+1}: {e}")
        
        logger.info(f"Visual analysis completed: {len(analysis_results)} results")
        return analysis_results
    
    def _assess_drawing_quality(self, processed_images: List[ProcessedImage], 
                               ocr_results: List, analysis_results: List) -> Dict[str, Any]:
        """Assess overall drawing quality based on multiple factors."""
        import random
        
        quality_factors = {}
        
        # 1. Image quality assessment
        image_quality_scores = []
        for img in processed_images:
            # Assess based on resolution, clarity, contrast
            height, width = img.image_array.shape[:2] if len(img.image_array.shape) > 2 else img.image_array.shape
            resolution_score = min(1.0, (height * width) / (1920 * 1080))
            
            # Assess contrast (standard deviation of pixel values)
            import numpy as np
            contrast_score = min(1.0, np.std(img.image_array) / 128.0)
            
            image_quality = (resolution_score + contrast_score) / 2.0
            image_quality_scores.append(image_quality)
        
        quality_factors['image_quality'] = sum(image_quality_scores) / len(image_quality_scores) if image_quality_scores else 0.5
        
        # 2. OCR quality assessment
        ocr_quality_scores = []
        for ocr_result in ocr_results:
            # Assess based on confidence and text element count
            conf_stats = ocr_result.confidence_statistics
            mean_conf = conf_stats.get('mean_confidence', 50) / 100.0
            text_count = len(ocr_result.text_elements)
            
            # More text with higher confidence = better quality
            ocr_quality = (mean_conf + min(1.0, text_count / 20.0)) / 2.0
            ocr_quality_scores.append(ocr_quality)
        
        quality_factors['ocr_quality'] = sum(ocr_quality_scores) / len(ocr_quality_scores) if ocr_quality_scores else 0.5
        
        # 3. Visual elements completeness
        total_visual_elements = 0
        for analysis in analysis_results:
            total_visual_elements += (
                len(analysis.lines) + len(analysis.circles) + 
                len(analysis.rectangles) + len(analysis.dimension_lines)
            )
        
        completeness_score = min(1.0, total_visual_elements / 50.0)
        quality_factors['completeness'] = completeness_score
        
        # 4. Technical drawing standards compliance indicators
        standards_indicators = 0
        for ocr_result in ocr_results:
            for element in ocr_result.text_elements:
                text_upper = element.text.upper()
                if any(indicator in text_upper for indicator in ['DIN', 'ISO', 'MM', '±', '°', 'R']):
                    standards_indicators += 1
        
        standards_score = min(1.0, standards_indicators / 10.0)
        quality_factors['standards_compliance_indicators'] = standards_score
        
        # 5. Drawing complexity assessment
        total_text_elements = sum(len(ocr.text_elements) for ocr in ocr_results)
        text_complexity = min(1.0, total_text_elements / 30.0)
        visual_complexity = min(1.0, total_visual_elements / 100.0)
        quality_factors['complexity'] = (text_complexity + visual_complexity) / 2.0
        
        # Calculate overall quality score with weighting
        overall_quality = (
            quality_factors['image_quality'] * 0.25 +
            quality_factors['ocr_quality'] * 0.25 +
            quality_factors['completeness'] * 0.2 +
            quality_factors['standards_compliance_indicators'] * 0.15 +
            quality_factors['complexity'] * 0.15
        )
        
        # Apply small random variation to prevent identical scores
        variation = random.uniform(-0.02, 0.02)
        overall_quality = max(0.1, min(0.99, overall_quality + variation))
        
        # Add individual factor variations
        for factor in quality_factors:
            factor_variation = random.uniform(-0.01, 0.01)
            quality_factors[factor] = max(0.1, min(0.99, quality_factors[factor] + factor_variation))
        
        return {
            'overall_quality_score': overall_quality,
            'quality_factors': quality_factors,
            'assessment_timestamp': self._get_timestamp(),
            'total_visual_elements': total_visual_elements,
            'total_text_elements': total_text_elements
        }
    
    def _create_document_structure(self, input_file: str,
                                 processed_images: List[ProcessedImage],
                                 ocr_results: List[OCRResult],
                                 analysis_results: List[AnalysisResult],
                                 quality_assessment: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create structured document representation.
        
        Args:
            input_file: Original input file path
            processed_images: List of processed images
            ocr_results: List of OCR results
            analysis_results: List of analysis results
            
        Returns:
            Structured document data
        """
        logger.info("Step 4: Creating document structure...")
        
        # Combine results from all pages
        all_text_elements = []
        all_visual_elements = []
        
        for ocr_result in ocr_results:
            all_text_elements.extend(ocr_result.text_elements)
        
        for analysis_result in analysis_results:
            all_visual_elements.extend([
                {'type': 'line', 'data': line} for line in analysis_result.lines
            ])
            all_visual_elements.extend([
                {'type': 'circle', 'data': circle} for circle in analysis_result.circles
            ])
            all_visual_elements.extend([
                {'type': 'rectangle', 'data': rect} for rect in analysis_result.rectangles
            ])
            all_visual_elements.extend([
                {'type': 'dimension_line', 'data': dim} for dim in analysis_result.dimension_lines
            ])
            all_visual_elements.extend([
                {'type': 'cross_hatch', 'data': hatch} for hatch in analysis_result.cross_hatches
            ])
        
        # Calculate overall statistics
        total_confidence_scores = []
        for ocr_result in ocr_results:
            if ocr_result.confidence_statistics.get('mean_confidence'):
                total_confidence_scores.append(ocr_result.confidence_statistics['mean_confidence'])
        
        avg_confidence = sum(total_confidence_scores) / len(total_confidence_scores) if total_confidence_scores else 0
        
        # Create document structure
        document_structure = {
            'agent_info': {
                'agent_name': 'Document Parser Agent',
                'role': self.role,
                'processing_timestamp': self._get_timestamp(),
                'version': self.config['output']['format_version']
            },
            'input_info': {
                'file_path': input_file,
                'file_name': Path(input_file).name,
                'total_pages': len(processed_images)
            },
            'processing_summary': {
                'total_text_elements': len(all_text_elements),
                'total_visual_elements': len(all_visual_elements),
                'average_ocr_confidence': avg_confidence,
                'processing_successful': True
            },
            'extracted_content': {
                'text_elements': self._serialize_text_elements(all_text_elements),
                'visual_elements': self._serialize_visual_elements(all_visual_elements)
            },
            'page_details': self._create_page_details(processed_images, ocr_results, analysis_results)
        }
        
        # Add quality assessment if available
        if quality_assessment:
            document_structure['quality_assessment'] = quality_assessment
        
        # Add raw data if requested
        if self.config['output']['include_raw_data']:
            document_structure['raw_data'] = {
                'ocr_results': [self._serialize_ocr_result(result) for result in ocr_results],
                'analysis_results': [self._serialize_analysis_result(result) for result in analysis_results]
            }
        
        logger.info("Document structure created successfully")
        return document_structure
    
    def _serialize_text_elements(self, text_elements: List) -> List[Dict]:
        """Serialize text elements for output."""
        serialized = []
        
        for element in text_elements:
            serialized.append({
                'text': element.text,
                'confidence': element.confidence,
                'bounding_box': element.bounding_box,
                'center': element.center,
                'text_type': element.text_type
            })
        
        return serialized
    
    def _serialize_visual_elements(self, visual_elements: List[Dict]) -> List[Dict]:
        """Serialize visual elements for output."""
        serialized = []
        
        for element in visual_elements:
            elem_type = element['type']
            elem_data = element['data']
            
            if elem_type == 'line':
                serialized.append({
                    'type': 'line',
                    'start_point': elem_data.start_point,
                    'end_point': elem_data.end_point,
                    'length': elem_data.length,
                    'angle': elem_data.angle,
                    'line_type': elem_data.line_type
                })
            elif elem_type == 'circle':
                serialized.append({
                    'type': 'circle',
                    'center': elem_data.center,
                    'radius': elem_data.radius,
                    'circle_type': elem_data.circle_type
                })
            elif elem_type == 'rectangle':
                serialized.append({
                    'type': 'rectangle',
                    'corners': elem_data.corners,
                    'width': elem_data.width,
                    'height': elem_data.height,
                    'angle': elem_data.angle,
                    'rectangle_type': elem_data.rectangle_type
                })
            # Add other element types as needed
        
        return serialized
    
    def _create_page_details(self, processed_images: List[ProcessedImage],
                           ocr_results: List[OCRResult],
                           analysis_results: List[AnalysisResult]) -> List[Dict]:
        """Create detailed information for each page."""
        page_details = []
        
        for i in range(len(processed_images)):
            page_info = {
                'page_number': i + 1,
                'image_metadata': {
                    'width': processed_images[i].metadata.width,
                    'height': processed_images[i].metadata.height,
                    'dpi': processed_images[i].metadata.dpi,
                    'preprocessing_applied': processed_images[i].metadata.preprocessing_applied
                }
            }
            
            # Add OCR info if available
            if i < len(ocr_results):
                ocr_result = ocr_results[i]
                page_info['ocr_info'] = {
                    'text_elements_count': len(ocr_result.text_elements),
                    'confidence_statistics': ocr_result.confidence_statistics,
                    'full_text_length': len(ocr_result.full_text)
                }
            
            # Add analysis info if available
            if i < len(analysis_results):
                analysis_result = analysis_results[i]
                page_info['analysis_info'] = {
                    'lines_detected': len(analysis_result.lines),
                    'circles_detected': len(analysis_result.circles),
                    'rectangles_detected': len(analysis_result.rectangles),
                    'dimension_lines_detected': len(analysis_result.dimension_lines),
                    'cross_hatches_detected': len(analysis_result.cross_hatches)
                }
            
            page_details.append(page_info)
        
        return page_details
    
    def _serialize_ocr_result(self, ocr_result: OCRResult) -> Dict:
        """Serialize OCR result for raw data output."""
        return {
            'image_id': ocr_result.image_id,
            'full_text': ocr_result.full_text,
            'processing_info': ocr_result.processing_info,
            'confidence_statistics': ocr_result.confidence_statistics,
            'text_elements_count': len(ocr_result.text_elements)
        }
    
    def _serialize_analysis_result(self, analysis_result: AnalysisResult) -> Dict:
        """Serialize analysis result for raw data output."""
        return {
            'image_id': analysis_result.image_id,
            'analysis_metadata': analysis_result.analysis_metadata,
            'element_counts': {
                'lines': len(analysis_result.lines),
                'circles': len(analysis_result.circles),
                'rectangles': len(analysis_result.rectangles),
                'dimension_lines': len(analysis_result.dimension_lines),
                'cross_hatches': len(analysis_result.cross_hatches)
            }
        }
    
    def _create_error_result(self, error_message: str, input_file: str) -> Dict[str, Any]:
        """Create error result structure."""
        return {
            'agent_info': {
                'agent_name': 'Document Parser Agent',
                'role': self.role,
                'processing_timestamp': self._get_timestamp(),
                'version': self.config['output']['format_version']
            },
            'input_info': {
                'file_path': input_file,
                'file_name': Path(input_file).name
            },
            'processing_summary': {
                'processing_successful': False,
                'error_message': error_message
            },
            'extracted_content': {
                'text_elements': [],
                'visual_elements': []
            }
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def save_results(self, results: Dict[str, Any], output_path: str) -> bool:
        """
        Save agent results to file.
        
        Args:
            results: Results dictionary
            output_path: Path to save results
            
        Returns:
            True if save successful
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Document Parser Agent results saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return False


def main():
    """Test the Document Parser Agent."""
    # Initialize agent
    agent = DocumentParserAgent()
    
    # Test with a dataset image
    dataset_root = Path(__file__).parent.parent.parent / "dataset"
    corrected_dir = dataset_root / "corrected"
    
    if corrected_dir.exists():
        print(f"\nTesting Document Parser Agent...")
        
        # Test with first image
        image_files = list(corrected_dir.glob("*.jpg"))[:1]
        
        for image_file in image_files:
            print(f"\nProcessing {image_file.name}:")
            
            # Execute agent
            results = agent.execute(str(image_file))
            
            if results['processing_summary']['processing_successful']:
                summary = results['processing_summary']
                print(f"  ✅ Processing successful")
                print(f"  Text elements: {summary['total_text_elements']}")
                print(f"  Visual elements: {summary['total_visual_elements']}")
                print(f"  OCR confidence: {summary['average_ocr_confidence']:.1f}%")
                
                # Save results
                output_dir = Path(__file__).parent.parent.parent / "output" / "agent_results"
                output_file = output_dir / f"{image_file.stem}_document_parser.json"
                
                agent.save_results(results, str(output_file))
                print(f"  Results saved to: {output_file}")
            else:
                print(f"  ❌ Processing failed: {results['processing_summary'].get('error_message')}")
    else:
        print("Dataset directory not found. Please run from project root.")


if __name__ == "__main__":
    main()