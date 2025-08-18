#!/usr/bin/env python3
"""
Image Analysis Tool for Technical Drawing Feedback System

This tool analyzes technical drawings to detect geometric elements like
lines, shapes, dimensions, and technical symbols using computer vision.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import json
import math

logger = logging.getLogger(__name__)


@dataclass
class Line:
    """Represents a detected line."""
    start_point: Tuple[int, int]
    end_point: Tuple[int, int]
    length: float
    angle: float  # in degrees
    line_type: str = "unknown"  # dimension, construction, outline, etc.


@dataclass
class Circle:
    """Represents a detected circle."""
    center: Tuple[int, int]
    radius: float
    circle_type: str = "unknown"  # hole, arc, construction, etc.


@dataclass
class Rectangle:
    """Represents a detected rectangle."""
    corners: List[Tuple[int, int]]
    width: float
    height: float
    angle: float
    rectangle_type: str = "unknown"


@dataclass
class DimensionLine:
    """Represents a dimension line with arrows."""
    line: Line
    arrows: List[Tuple[int, int]]
    dimension_text_area: Optional[Tuple[int, int, int, int]] = None
    dimension_value: Optional[str] = None


@dataclass
class CrossHatch:
    """Represents cross-hatched areas."""
    contour: np.ndarray
    area: float
    hatch_lines: List[Line]
    hatch_pattern: str = "unknown"


@dataclass
class AnalysisResult:
    """Container for image analysis results."""
    image_id: str
    lines: List[Line]
    circles: List[Circle]
    rectangles: List[Rectangle]
    dimension_lines: List[DimensionLine]
    cross_hatches: List[CrossHatch]
    analysis_metadata: Dict


class ImageAnalysisTool:
    """Tool for analyzing geometric elements in technical drawings."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize image analysis tool.
        
        Args:
            config: Configuration dictionary for analysis parameters
        """
        self.config = config or self._get_default_config()
        
    def _get_default_config(self) -> Dict:
        """Get default configuration for image analysis."""
        return {
            'line_detection': {
                'canny_threshold1': 50,
                'canny_threshold2': 150,
                'hough_threshold': 100,
                'min_line_length': 30,
                'max_line_gap': 10,
                'angle_tolerance': 5  # degrees
            },
            'circle_detection': {
                'dp': 1,
                'min_dist': 30,
                'param1': 50,
                'param2': 30,
                'min_radius': 5,
                'max_radius': 200
            },
            'rectangle_detection': {
                'epsilon_factor': 0.02,
                'min_area': 100,
                'aspect_ratio_tolerance': 0.1
            },
            'dimension_detection': {
                'arrow_size_threshold': 10,
                'text_area_expansion': 20
            },
            'crosshatch_detection': {
                'min_hatch_lines': 3,
                'hatch_angle_tolerance': 10,
                'line_spacing_tolerance': 5
            }
        }
    
    def preprocess_for_analysis(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess image for geometric analysis.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (grayscale_image, edge_image)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Detect edges using Canny
        config = self.config['line_detection']
        edges = cv2.Canny(gray, config['canny_threshold1'], config['canny_threshold2'])
        
        return gray, edges
    
    def detect_lines(self, edges: np.ndarray) -> List[Line]:
        """
        Detect lines using Hough Line Transform.
        
        Args:
            edges: Edge-detected image
            
        Returns:
            List of detected lines
        """
        lines = []
        config = self.config['line_detection']
        
        try:
            # Use HoughLinesP for line segments
            detected_lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi/180,
                threshold=config['hough_threshold'],
                minLineLength=config['min_line_length'],
                maxLineGap=config['max_line_gap']
            )
            
            if detected_lines is not None:
                for line_data in detected_lines:
                    x1, y1, x2, y2 = line_data[0]
                    
                    # Calculate line properties
                    length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                    angle = math.degrees(math.atan2(y2-y1, x2-x1))
                    
                    # Normalize angle to 0-180 degrees
                    if angle < 0:
                        angle += 180
                    
                    line = Line(
                        start_point=(x1, y1),
                        end_point=(x2, y2),
                        length=length,
                        angle=angle,
                        line_type=self._classify_line_type(length, angle)
                    )
                    
                    lines.append(line)
            
            logger.info(f"Detected {len(lines)} lines")
            
        except Exception as e:
            logger.error(f"Line detection failed: {e}")
            
        return lines
    
    def detect_circles(self, gray: np.ndarray) -> List[Circle]:
        """
        Detect circles using Hough Circle Transform.
        
        Args:
            gray: Grayscale image
            
        Returns:
            List of detected circles
        """
        circles = []
        config = self.config['circle_detection']
        
        try:
            detected_circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=config['dp'],
                minDist=config['min_dist'],
                param1=config['param1'],
                param2=config['param2'],
                minRadius=config['min_radius'],
                maxRadius=config['max_radius']
            )
            
            if detected_circles is not None:
                detected_circles = np.round(detected_circles[0, :]).astype("int")
                
                for (x, y, r) in detected_circles:
                    circle = Circle(
                        center=(x, y),
                        radius=float(r),
                        circle_type=self._classify_circle_type(r)
                    )
                    circles.append(circle)
            
            logger.info(f"Detected {len(circles)} circles")
            
        except Exception as e:
            logger.error(f"Circle detection failed: {e}")
            
        return circles
    
    def detect_rectangles(self, edges: np.ndarray) -> List[Rectangle]:
        """
        Detect rectangles using contour analysis.
        
        Args:
            edges: Edge-detected image
            
        Returns:
            List of detected rectangles
        """
        rectangles = []
        config = self.config['rectangle_detection']
        
        try:
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Approximate contour to polygon
                epsilon = config['epsilon_factor'] * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if it's a quadrilateral with sufficient area
                if len(approx) == 4 and cv2.contourArea(contour) > config['min_area']:
                    # Get corner points
                    corners = [(point[0][0], point[0][1]) for point in approx]
                    
                    # Calculate dimensions
                    width = self._distance(corners[0], corners[1])
                    height = self._distance(corners[1], corners[2])
                    
                    # Calculate angle (assuming first edge as reference)
                    angle = math.degrees(math.atan2(
                        corners[1][1] - corners[0][1],
                        corners[1][0] - corners[0][0]
                    ))
                    
                    rectangle = Rectangle(
                        corners=corners,
                        width=width,
                        height=height,
                        angle=angle,
                        rectangle_type=self._classify_rectangle_type(width, height)
                    )
                    
                    rectangles.append(rectangle)
            
            logger.info(f"Detected {len(rectangles)} rectangles")
            
        except Exception as e:
            logger.error(f"Rectangle detection failed: {e}")
            
        return rectangles
    
    def detect_dimension_lines(self, lines: List[Line], 
                             circles: List[Circle]) -> List[DimensionLine]:
        """
        Detect dimension lines with arrows.
        
        Args:
            lines: Detected lines
            circles: Detected circles (for arrow detection)
            
        Returns:
            List of detected dimension lines
        """
        dimension_lines = []
        config = self.config['dimension_detection']
        
        try:
            # Look for horizontal and vertical lines that could be dimensions
            candidate_lines = [line for line in lines 
                             if self._is_dimension_line_candidate(line)]
            
            for line in candidate_lines:
                # Look for arrows near line endpoints
                arrows = self._find_arrows_near_line(line, lines)
                
                if len(arrows) >= 1:  # At least one arrow found
                    # Estimate text area near the line
                    text_area = self._estimate_dimension_text_area(line)
                    
                    dimension_line = DimensionLine(
                        line=line,
                        arrows=arrows,
                        dimension_text_area=text_area
                    )
                    
                    dimension_lines.append(dimension_line)
            
            logger.info(f"Detected {len(dimension_lines)} dimension lines")
            
        except Exception as e:
            logger.error(f"Dimension line detection failed: {e}")
            
        return dimension_lines
    
    def detect_cross_hatches(self, lines: List[Line], 
                           gray: np.ndarray) -> List[CrossHatch]:
        """
        Detect cross-hatched areas.
        
        Args:
            lines: Detected lines
            gray: Grayscale image
            
        Returns:
            List of detected cross-hatched areas
        """
        cross_hatches = []
        config = self.config['crosshatch_detection']
        
        try:
            # Group lines by similar angles and proximity
            hatch_groups = self._group_hatch_lines(lines)
            
            for group in hatch_groups:
                if len(group) >= config['min_hatch_lines']:
                    # Find the boundary of this hatched area
                    contour = self._find_hatch_boundary(group, gray)
                    
                    if contour is not None:
                        area = cv2.contourArea(contour)
                        pattern = self._classify_hatch_pattern(group)
                        
                        cross_hatch = CrossHatch(
                            contour=contour,
                            area=area,
                            hatch_lines=group,
                            hatch_pattern=pattern
                        )
                        
                        cross_hatches.append(cross_hatch)
            
            logger.info(f"Detected {len(cross_hatches)} cross-hatched areas")
            
        except Exception as e:
            logger.error(f"Cross-hatch detection failed: {e}")
            
        return cross_hatches
    
    def analyze_image(self, image: np.ndarray, image_id: str = "unknown") -> AnalysisResult:
        """
        Perform complete geometric analysis of technical drawing.
        
        Args:
            image: Input image
            image_id: Identifier for the image
            
        Returns:
            AnalysisResult with all detected elements
        """
        logger.info(f"Starting analysis of image {image_id}")
        
        # Preprocess image
        gray, edges = self.preprocess_for_analysis(image)
        
        # Detect geometric elements
        lines = self.detect_lines(edges)
        circles = self.detect_circles(gray)
        rectangles = self.detect_rectangles(edges)
        dimension_lines = self.detect_dimension_lines(lines, circles)
        cross_hatches = self.detect_cross_hatches(lines, gray)
        
        # Compile metadata
        analysis_metadata = {
            'image_dimensions': image.shape[:2],
            'preprocessing_applied': ['gaussian_blur', 'canny_edge_detection'],
            'detection_config': self.config,
            'total_elements_detected': (len(lines) + len(circles) + len(rectangles) + 
                                      len(dimension_lines) + len(cross_hatches))
        }
        
        return AnalysisResult(
            image_id=image_id,
            lines=lines,
            circles=circles,
            rectangles=rectangles,
            dimension_lines=dimension_lines,
            cross_hatches=cross_hatches,
            analysis_metadata=analysis_metadata
        )
    
    # Helper methods
    def _classify_line_type(self, length: float, angle: float) -> str:
        """Classify line type based on properties."""
        # Horizontal or vertical lines might be dimensions
        if abs(angle) < 5 or abs(angle - 90) < 5 or abs(angle - 180) < 5:
            if length > 50:
                return "potential_dimension"
            else:
                return "construction"
        else:
            return "outline"
    
    def _classify_circle_type(self, radius: float) -> str:
        """Classify circle type based on size."""
        if radius < 10:
            return "small_hole"
        elif radius < 50:
            return "hole"
        else:
            return "large_feature"
    
    def _classify_rectangle_type(self, width: float, height: float) -> str:
        """Classify rectangle type based on dimensions."""
        aspect_ratio = max(width, height) / min(width, height)
        
        if aspect_ratio > 5:
            return "line_feature"
        elif abs(aspect_ratio - 1) < 0.2:
            return "square"
        else:
            return "rectangle"
    
    def _distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calculate distance between two points."""
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    def _is_dimension_line_candidate(self, line: Line) -> bool:
        """Check if line could be a dimension line."""
        # Dimension lines are typically horizontal or vertical
        angle_tolerance = self.config['line_detection']['angle_tolerance']
        
        return (abs(line.angle) < angle_tolerance or 
                abs(line.angle - 90) < angle_tolerance or 
                abs(line.angle - 180) < angle_tolerance)
    
    def _find_arrows_near_line(self, line: Line, all_lines: List[Line]) -> List[Tuple[int, int]]:
        """Find arrow markers near line endpoints."""
        arrows = []
        search_radius = 20
        
        # Look for short lines near endpoints that could be arrows
        for other_line in all_lines:
            if other_line == line:
                continue
                
            # Check if other line is near either endpoint of main line
            for endpoint in [line.start_point, line.end_point]:
                if (self._distance(endpoint, other_line.start_point) < search_radius or
                    self._distance(endpoint, other_line.end_point) < search_radius):
                    
                    # Check if it's short enough to be an arrow
                    if other_line.length < 15:
                        arrows.append(endpoint)
                        break
        
        return arrows
    
    def _estimate_dimension_text_area(self, line: Line) -> Tuple[int, int, int, int]:
        """Estimate where dimension text might be located."""
        expansion = self.config['dimension_detection']['text_area_expansion']
        
        # Create bounding box around line with expansion for text
        x1, y1 = line.start_point
        x2, y2 = line.end_point
        
        min_x = min(x1, x2) - expansion
        max_x = max(x1, x2) + expansion
        min_y = min(y1, y2) - expansion
        max_y = max(y1, y2) + expansion
        
        return (min_x, min_y, max_x - min_x, max_y - min_y)
    
    def _group_hatch_lines(self, lines: List[Line]) -> List[List[Line]]:
        """Group lines that could form cross-hatching patterns."""
        groups = []
        used_lines = set()
        
        for i, line in enumerate(lines):
            if i in used_lines:
                continue
                
            # Start a new group
            group = [line]
            used_lines.add(i)
            
            # Find similar lines
            for j, other_line in enumerate(lines):
                if j in used_lines or j == i:
                    continue
                    
                # Check if lines are similar in angle and spacing
                if self._are_hatch_lines_similar(line, other_line):
                    group.append(other_line)
                    used_lines.add(j)
            
            if len(group) >= 3:  # Minimum for hatching
                groups.append(group)
        
        return groups
    
    def _are_hatch_lines_similar(self, line1: Line, line2: Line) -> bool:
        """Check if two lines could be part of the same hatch pattern."""
        angle_tolerance = self.config['crosshatch_detection']['hatch_angle_tolerance']
        
        # Similar angles
        angle_diff = abs(line1.angle - line2.angle)
        return angle_diff < angle_tolerance or angle_diff > (180 - angle_tolerance)
    
    def _find_hatch_boundary(self, hatch_lines: List[Line], 
                           gray: np.ndarray) -> Optional[np.ndarray]:
        """Find the boundary contour of a hatched area."""
        try:
            # Create a mask of the hatch lines
            mask = np.zeros(gray.shape, dtype=np.uint8)
            
            for line in hatch_lines:
                cv2.line(mask, line.start_point, line.end_point, 255, 2)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Return the largest contour
                return max(contours, key=cv2.contourArea)
                
        except Exception as e:
            logger.debug(f"Failed to find hatch boundary: {e}")
            
        return None
    
    def _classify_hatch_pattern(self, hatch_lines: List[Line]) -> str:
        """Classify the type of hatching pattern."""
        if not hatch_lines:
            return "unknown"
            
        # Analyze angles
        angles = [line.angle for line in hatch_lines]
        unique_angles = len(set(round(angle, 0) for angle in angles))
        
        if unique_angles == 1:
            return "parallel"
        elif unique_angles == 2:
            return "cross_hatch"
        else:
            return "complex"


def main():
    """Test the image analysis tool."""
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from pdf_processing_tool import PDFProcessingTool
    
    # Initialize tools
    pdf_tool = PDFProcessingTool()
    analysis_tool = ImageAnalysisTool()
    
    # Test with dataset images
    dataset_root = Path(__file__).parent.parent.parent / "dataset"
    corrected_dir = dataset_root / "corrected"
    
    if corrected_dir.exists():
        print(f"\nTesting Image Analysis Tool with dataset images...")
        
        # Process first image as test
        image_files = list(corrected_dir.glob("*.jpg"))[:1]
        
        for image_file in image_files:
            print(f"\nAnalyzing {image_file.name}:")
            
            # First process with PDF tool
            processed_images = pdf_tool.process_file(image_file)
            
            if processed_images:
                # Then apply image analysis
                result = analysis_tool.analyze_image(
                    processed_images[0].image_array, 
                    image_file.stem
                )
                
                print(f"  Lines detected: {len(result.lines)}")
                print(f"  Circles detected: {len(result.circles)}")
                print(f"  Rectangles detected: {len(result.rectangles)}")
                print(f"  Dimension lines: {len(result.dimension_lines)}")
                print(f"  Cross-hatched areas: {len(result.cross_hatches)}")
                
                # Show line type distribution
                line_types = {}
                for line in result.lines:
                    line_types[line.line_type] = line_types.get(line.line_type, 0) + 1
                
                if line_types:
                    print(f"  Line types:")
                    for line_type, count in line_types.items():
                        print(f"    {line_type}: {count}")
                        
            else:
                print(f"  Failed to process with PDF tool")
    else:
        print("Dataset directory not found. Please run from project root.")


if __name__ == "__main__":
    main()