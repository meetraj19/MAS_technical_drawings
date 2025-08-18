#!/usr/bin/env python3
"""
Dimension Extraction Tool for Technical Drawing Feedback System

This tool extracts and parses dimensions, tolerances, and technical specifications
from German technical drawings using OCR text and geometric analysis.
"""

import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class Dimension:
    """Represents an extracted dimension."""
    value: float
    unit: str
    text: str
    position: Tuple[int, int]
    dimension_type: str  # diameter, length, radius, angle, etc.
    tolerance: Optional['Tolerance'] = None


@dataclass
class Tolerance:
    """Represents a tolerance specification."""
    upper_limit: Optional[float]
    lower_limit: Optional[float]
    tolerance_class: Optional[str]  # H7, h6, etc.
    tolerance_type: str  # bilateral, unilateral, etc.


@dataclass
class SurfaceFinish:
    """Represents surface finish specification."""
    roughness_value: float
    roughness_type: str  # Ra, Rz, etc.
    position: Tuple[int, int]
    text: str


@dataclass
class ThreadSpecification:
    """Represents thread specification."""
    thread_type: str  # metric, etc.
    nominal_diameter: float
    pitch: Optional[float]
    thread_class: Optional[str]
    position: Tuple[int, int]
    text: str


@dataclass
class ExtractionResult:
    """Container for dimension extraction results."""
    image_id: str
    dimensions: List[Dimension]
    tolerances: List[Tolerance]
    surface_finishes: List[SurfaceFinish]
    thread_specifications: List[ThreadSpecification]
    extraction_metadata: Dict


class DimensionExtractionTool:
    """Tool for extracting technical dimensions and specifications."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize dimension extraction tool.
        
        Args:
            config: Configuration dictionary for extraction parameters
        """
        self.config = config or self._get_default_config()
        self.patterns = self._compile_patterns()
        
    def _get_default_config(self) -> Dict:
        """Get default configuration for dimension extraction."""
        return {
            'german_units': {
                'mm': 'millimeter',
                'cm': 'centimeter', 
                'm': 'meter',
                '°': 'degree',
                'grad': 'degree'
            },
            'tolerance_classes': [
                'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12',
                'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10', 'h11', 'h12',
                'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12',
                'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'g9', 'g10', 'g11', 'g12'
            ],
            'confidence_threshold': 0.7,
            'proximity_threshold': 50  # pixels
        }
    
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for dimension extraction."""
        patterns = {}
        
        # Diameter patterns (∅, ø, Ø, φ, ⌀)
        patterns['diameter'] = re.compile(
            r'[∅øØφ⌀]\s*(\d+(?:[.,]\d+)?)\s*(mm|cm|m)?',
            re.IGNORECASE
        )
        
        # Linear dimensions
        patterns['linear'] = re.compile(
            r'(\d+(?:[.,]\d+)?)\s*(mm|cm|m|×|x|X)\s*(\d+(?:[.,]\d+)?)?',
            re.IGNORECASE
        )
        
        # Angles
        patterns['angle'] = re.compile(
            r'(\d+(?:[.,]\d+)?)\s*[°]|(\d+(?:[.,]\d+)?)\s*grad',
            re.IGNORECASE
        )
        
        # Tolerances
        patterns['tolerance_bilateral'] = re.compile(
            r'[+-]\s*(\d+(?:[.,]\d+)?)\s*(mm|cm|m)?',
            re.IGNORECASE
        )
        
        patterns['tolerance_class'] = re.compile(
            r'([HhGg]\d{1,2})',
            re.IGNORECASE
        )
        
        # Surface finish (Ra, Rz, etc.)
        patterns['surface_finish'] = re.compile(
            r'(Ra|Rz|Rq|Rt)\s*(\d+(?:[.,]\d+)?)',
            re.IGNORECASE
        )
        
        # Thread specifications (M8x1.25, M10, etc.)
        patterns['thread_metric'] = re.compile(
            r'M\s*(\d+(?:[.,]\d+)?)\s*(?:[xX×]\s*(\d+(?:[.,]\d+)?))?',
            re.IGNORECASE
        )
        
        # Radius
        patterns['radius'] = re.compile(
            r'R\s*(\d+(?:[.,]\d+)?)\s*(mm|cm|m)?',
            re.IGNORECASE
        )
        
        # Chamfer
        patterns['chamfer'] = re.compile(
            r'(\d+(?:[.,]\d+)?)\s*[xX×]\s*(\d+(?:[.,]\d+)?)\s*[°]',
            re.IGNORECASE
        )
        
        return patterns
    
    def extract_dimensions_from_text(self, text_elements: List) -> List[Dimension]:
        """
        Extract dimensions from OCR text elements.
        
        Args:
            text_elements: List of TextElement objects from OCR
            
        Returns:
            List of extracted dimensions
        """
        dimensions = []
        
        for text_elem in text_elements:
            text = text_elem.text
            position = text_elem.center
            
            # Try different dimension patterns
            extracted_dims = []
            
            # Diameter
            matches = self.patterns['diameter'].finditer(text)
            for match in matches:
                value = self._parse_number(match.group(1))
                unit = match.group(2) or 'mm'
                
                dimension = Dimension(
                    value=value,
                    unit=unit,
                    text=match.group(0),
                    position=position,
                    dimension_type='diameter'
                )
                extracted_dims.append(dimension)
            
            # Linear dimensions
            matches = self.patterns['linear'].finditer(text)
            for match in matches:
                value = self._parse_number(match.group(1))
                unit = match.group(2) if match.group(2) not in ['×', 'x', 'X'] else 'mm'
                
                # Check if it's a multiplication (dimension x dimension)
                if match.group(3):
                    # Two dimensions (e.g., "40x60")
                    value2 = self._parse_number(match.group(3))
                    
                    # First dimension
                    dimension1 = Dimension(
                        value=value,
                        unit='mm',
                        text=f"{match.group(1)}",
                        position=position,
                        dimension_type='length'
                    )
                    
                    # Second dimension
                    dimension2 = Dimension(
                        value=value2,
                        unit='mm',
                        text=f"{match.group(3)}",
                        position=position,
                        dimension_type='width'
                    )
                    
                    extracted_dims.extend([dimension1, dimension2])
                else:
                    dimension = Dimension(
                        value=value,
                        unit=unit,
                        text=match.group(1),
                        position=position,
                        dimension_type='length'
                    )
                    extracted_dims.append(dimension)
            
            # Angles
            matches = self.patterns['angle'].finditer(text)
            for match in matches:
                value = self._parse_number(match.group(1) or match.group(2))
                
                dimension = Dimension(
                    value=value,
                    unit='degree',
                    text=match.group(0),
                    position=position,
                    dimension_type='angle'
                )
                extracted_dims.append(dimension)
            
            # Radius
            matches = self.patterns['radius'].finditer(text)
            for match in matches:
                value = self._parse_number(match.group(1))
                unit = match.group(2) or 'mm'
                
                dimension = Dimension(
                    value=value,
                    unit=unit,
                    text=match.group(0),
                    position=position,
                    dimension_type='radius'
                )
                extracted_dims.append(dimension)
            
            dimensions.extend(extracted_dims)
        
        logger.info(f"Extracted {len(dimensions)} dimensions from text")
        return dimensions
    
    def extract_tolerances(self, text_elements: List) -> List[Tolerance]:
        """
        Extract tolerance specifications from text.
        
        Args:
            text_elements: List of TextElement objects from OCR
            
        Returns:
            List of extracted tolerances
        """
        tolerances = []
        
        for text_elem in text_elements:
            text = text_elem.text
            
            # Bilateral tolerances (±0.1)
            matches = self.patterns['tolerance_bilateral'].finditer(text)
            for match in matches:
                value = self._parse_number(match.group(1))
                
                tolerance = Tolerance(
                    upper_limit=value,
                    lower_limit=-value,
                    tolerance_class=None,
                    tolerance_type='bilateral'
                )
                tolerances.append(tolerance)
            
            # Tolerance classes (H7, h6, etc.)
            matches = self.patterns['tolerance_class'].finditer(text)
            for match in matches:
                tolerance_class = match.group(1)
                
                tolerance = Tolerance(
                    upper_limit=None,
                    lower_limit=None,
                    tolerance_class=tolerance_class,
                    tolerance_type='class'
                )
                tolerances.append(tolerance)
        
        logger.info(f"Extracted {len(tolerances)} tolerances")
        return tolerances
    
    def extract_surface_finishes(self, text_elements: List) -> List[SurfaceFinish]:
        """
        Extract surface finish specifications.
        
        Args:
            text_elements: List of TextElement objects from OCR
            
        Returns:
            List of extracted surface finishes
        """
        surface_finishes = []
        
        for text_elem in text_elements:
            text = text_elem.text
            position = text_elem.center
            
            matches = self.patterns['surface_finish'].finditer(text)
            for match in matches:
                roughness_type = match.group(1)
                roughness_value = self._parse_number(match.group(2))
                
                surface_finish = SurfaceFinish(
                    roughness_value=roughness_value,
                    roughness_type=roughness_type,
                    position=position,
                    text=match.group(0)
                )
                surface_finishes.append(surface_finish)
        
        logger.info(f"Extracted {len(surface_finishes)} surface finishes")
        return surface_finishes
    
    def extract_thread_specifications(self, text_elements: List) -> List[ThreadSpecification]:
        """
        Extract thread specifications.
        
        Args:
            text_elements: List of TextElement objects from OCR
            
        Returns:
            List of extracted thread specifications
        """
        thread_specs = []
        
        for text_elem in text_elements:
            text = text_elem.text
            position = text_elem.center
            
            matches = self.patterns['thread_metric'].finditer(text)
            for match in matches:
                nominal_diameter = self._parse_number(match.group(1))
                pitch = self._parse_number(match.group(2)) if match.group(2) else None
                
                thread_spec = ThreadSpecification(
                    thread_type='metric',
                    nominal_diameter=nominal_diameter,
                    pitch=pitch,
                    thread_class=None,
                    position=position,
                    text=match.group(0)
                )
                thread_specs.append(thread_spec)
        
        logger.info(f"Extracted {len(thread_specs)} thread specifications")
        return thread_specs
    
    def link_tolerances_to_dimensions(self, dimensions: List[Dimension], 
                                    tolerances: List[Tolerance]) -> None:
        """
        Link tolerance specifications to their corresponding dimensions.
        
        Args:
            dimensions: List of dimensions
            tolerances: List of tolerances
        """
        proximity_threshold = self.config['proximity_threshold']
        
        for dimension in dimensions:
            closest_tolerance = None
            min_distance = float('inf')
            
            for tolerance in tolerances:
                # Simple proximity-based linking
                # In a real implementation, this would be more sophisticated
                if tolerance.tolerance_type == 'class':
                    # Tolerance classes are typically for hole/shaft fits
                    if dimension.dimension_type in ['diameter', 'hole', 'shaft']:
                        dimension.tolerance = tolerance
                        break
        
        logger.info("Linked tolerances to dimensions")
    
    def extract_from_ocr_result(self, ocr_result) -> ExtractionResult:
        """
        Extract all technical specifications from OCR result.
        
        Args:
            ocr_result: OCRResult object
            
        Returns:
            ExtractionResult with extracted specifications
        """
        # Extract different types of specifications
        dimensions = self.extract_dimensions_from_text(ocr_result.text_elements)
        tolerances = self.extract_tolerances(ocr_result.text_elements)
        surface_finishes = self.extract_surface_finishes(ocr_result.text_elements)
        thread_specs = self.extract_thread_specifications(ocr_result.text_elements)
        
        # Link tolerances to dimensions
        self.link_tolerances_to_dimensions(dimensions, tolerances)
        
        # Compile metadata
        extraction_metadata = {
            'total_text_elements': len(ocr_result.text_elements),
            'extraction_patterns_used': list(self.patterns.keys()),
            'extraction_successful': True,
            'statistics': {
                'dimensions_found': len(dimensions),
                'tolerances_found': len(tolerances),
                'surface_finishes_found': len(surface_finishes),
                'threads_found': len(thread_specs)
            }
        }
        
        return ExtractionResult(
            image_id=ocr_result.image_id,
            dimensions=dimensions,
            tolerances=tolerances,
            surface_finishes=surface_finishes,
            thread_specifications=thread_specs,
            extraction_metadata=extraction_metadata
        )
    
    def _parse_number(self, number_str: str) -> float:
        """Parse German number format (comma as decimal separator)."""
        if not number_str:
            return 0.0
            
        # Replace comma with dot for German numbers
        number_str = number_str.replace(',', '.')
        
        try:
            return float(number_str)
        except ValueError:
            logger.warning(f"Could not parse number: {number_str}")
            return 0.0
    
    def save_extraction_results(self, results: List[ExtractionResult], 
                              output_dir: Union[str, Path]) -> bool:
        """
        Save extraction results to JSON files.
        
        Args:
            results: List of extraction results
            output_dir: Directory to save results
            
        Returns:
            True if save successful
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            for result in results:
                # Convert to serializable format
                data = {
                    'image_id': result.image_id,
                    'dimensions': [
                        {
                            'value': dim.value,
                            'unit': dim.unit,
                            'text': dim.text,
                            'position': dim.position,
                            'dimension_type': dim.dimension_type,
                            'tolerance': {
                                'upper_limit': dim.tolerance.upper_limit,
                                'lower_limit': dim.tolerance.lower_limit,
                                'tolerance_class': dim.tolerance.tolerance_class,
                                'tolerance_type': dim.tolerance.tolerance_type
                            } if dim.tolerance else None
                        }
                        for dim in result.dimensions
                    ],
                    'surface_finishes': [
                        {
                            'roughness_value': sf.roughness_value,
                            'roughness_type': sf.roughness_type,
                            'position': sf.position,
                            'text': sf.text
                        }
                        for sf in result.surface_finishes
                    ],
                    'thread_specifications': [
                        {
                            'thread_type': ts.thread_type,
                            'nominal_diameter': ts.nominal_diameter,
                            'pitch': ts.pitch,
                            'thread_class': ts.thread_class,
                            'position': ts.position,
                            'text': ts.text
                        }
                        for ts in result.thread_specifications
                    ],
                    'extraction_metadata': result.extraction_metadata
                }
                
                output_file = output_dir / f"{result.image_id}_dimensions.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Saved extraction results to {output_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving extraction results: {e}")
            return False


def main():
    """Test the dimension extraction tool."""
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from pdf_processing_tool import PDFProcessingTool
    from ocr_tool import OCRTool
    
    # Initialize tools
    pdf_tool = PDFProcessingTool()
    ocr_tool = OCRTool()
    dimension_tool = DimensionExtractionTool()
    
    # Test with dataset images
    dataset_root = Path(__file__).parent.parent.parent / "dataset"
    corrected_dir = dataset_root / "corrected"
    
    if corrected_dir.exists():
        print(f"\nTesting Dimension Extraction Tool...")
        
        # Process first image as test
        image_files = list(corrected_dir.glob("*.jpg"))[:1]
        
        for image_file in image_files:
            print(f"\nProcessing {image_file.name}:")
            
            # Process through pipeline
            processed_images = pdf_tool.process_file(image_file)
            
            if processed_images:
                ocr_result = ocr_tool.process_processed_image(processed_images[0])
                extraction_result = dimension_tool.extract_from_ocr_result(ocr_result)
                
                print(f"  Dimensions extracted: {len(extraction_result.dimensions)}")
                print(f"  Tolerances extracted: {len(extraction_result.tolerances)}")
                print(f"  Surface finishes: {len(extraction_result.surface_finishes)}")
                print(f"  Thread specifications: {len(extraction_result.thread_specifications)}")
                
                # Show some extracted dimensions
                for i, dim in enumerate(extraction_result.dimensions[:5]):
                    print(f"    Dimension {i+1}: {dim.value} {dim.unit} ({dim.dimension_type}) - '{dim.text}'")
                    
            else:
                print(f"  Failed to process image")
    else:
        print("Dataset directory not found. Please run from project root.")


if __name__ == "__main__":
    main()