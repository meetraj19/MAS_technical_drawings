#!/usr/bin/env python3
"""
Visual Annotation Tool for Technical Drawing Feedback System

This tool creates visual annotations on technical drawings using your
bounding box coordinates and German feedback data.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class Annotation:
    """Represents a visual annotation."""
    annotation_id: str
    bounding_box: Tuple[int, int, int, int]  # (x, y, width, height)
    label: str
    color: Tuple[int, int, int]
    severity: str
    feedback_text: Optional[str] = None


@dataclass
class AnnotationStyle:
    """Style configuration for annotations."""
    bbox_thickness: int = 2
    text_size: float = 0.8
    text_color: Tuple[int, int, int] = (0, 0, 0)
    background_alpha: float = 0.7
    font_path: Optional[str] = None


@dataclass
class AnnotatedImage:
    """Container for annotated image and metadata."""
    image_array: np.ndarray
    annotations: List[Annotation]
    legend: Dict[str, Tuple[int, int, int]]
    metadata: Dict


class VisualAnnotationTool:
    """Tool for creating visual annotations on technical drawings."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize visual annotation tool.
        
        Args:
            config: Configuration dictionary for annotation settings
        """
        self.config = config or self._get_default_config()
        self.color_map = self._create_color_map()
        
    def _get_default_config(self) -> Dict:
        """Get default configuration for visual annotations."""
        return {
            'colors': {
                'critical': (255, 0, 0),    # Red
                'major': (255, 165, 0),     # Orange
                'minor': (255, 255, 0),     # Yellow
                'correct': (0, 255, 0),     # Green
                'unknown': (128, 128, 128)  # Gray
            },
            'category_colors': {
                0: (128, 128, 128),  # Unknown - Gray
                1: (255, 0, 255),    # General - Magenta
                2: (255, 0, 0),      # Schraubenkopf - Red
                3: (0, 255, 0),      # Scheibe - Green
                4: (0, 0, 255),      # Platte - Blue
                5: (255, 255, 0),    # Gewindereserve - Yellow
                6: (255, 165, 0),    # Grundloch - Orange
                7: (128, 0, 128),    # Gewindedarstellung - Purple
                8: (0, 255, 255),    # Schraffur - Cyan
                9: (255, 192, 203)   # Schriftfeld - Pink
            },
            'annotation_style': {
                'bbox_thickness': 3,
                'text_size': 16,
                'text_color': (255, 255, 255),
                'background_alpha': 0.8,
                'label_offset': (5, -10)
            },
            'legend': {
                'position': 'top_right',
                'width': 200,
                'height': 300,
                'background_color': (255, 255, 255),
                'border_color': (0, 0, 0)
            }
        }
    
    def create_feedback_overlay(self, image_path: Union[str, Path], 
                              feedback_items: List[Dict],
                              ocr_elements: Optional[List] = None,
                              pattern_matches: Optional[List] = None,
                              output_path: Optional[Union[str, Path]] = None) -> AnnotatedImage:
        """
        Create comprehensive visual feedback overlay on technical drawing.
        
        Args:
            image_path: Path to the original image
            feedback_items: List of feedback items from German feedback generator
            ocr_elements: Optional OCR text elements with bounding boxes
            pattern_matches: Optional pattern matching results
            output_path: Optional path to save annotated image
            
        Returns:
            AnnotatedImage with visual feedback overlays
        """
        logger.info("Creating comprehensive visual feedback overlay...")
        
        # Load image
        if isinstance(image_path, str):
            image_path = Path(image_path)
        
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB for processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return self._create_comprehensive_annotations(
            image_rgb, feedback_items, ocr_elements, pattern_matches, output_path
        )
    
    def _create_comprehensive_annotations(self, image: np.ndarray, feedback_items: List[Dict],
                                        ocr_elements: Optional[List] = None,
                                        pattern_matches: Optional[List] = None,
                                        output_path: Optional[Union[str, Path]] = None) -> AnnotatedImage:
        """Create comprehensive annotations combining all feedback sources."""
        
        # Convert to PIL Image for better text rendering
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image, mode='RGBA')
        
        annotations = []
        
        # 1. Draw feedback item bounding boxes
        for i, feedback_item in enumerate(feedback_items):
            self._draw_feedback_annotation(draw, feedback_item, i, pil_image.size)
            
            # Create annotation record
            annotations.append(Annotation(
                annotation_id=f"feedback_{i}",
                bounding_box=self._get_feedback_bbox(feedback_item, pil_image.size),
                label=feedback_item.get('category_name', 'Unknown'),
                color=self._get_severity_color(feedback_item.get('severity', 'minor')),
                severity=feedback_item.get('severity', 'minor'),
                feedback_text=feedback_item.get('feedback_text', '')
            ))
        
        # 2. Draw OCR element bounding boxes if available
        if ocr_elements:
            for i, ocr_element in enumerate(ocr_elements):
                self._draw_ocr_annotation(draw, ocr_element, i)
        
        # 3. Draw pattern match bounding boxes if available
        if pattern_matches:
            for i, pattern_match in enumerate(pattern_matches):
                self._draw_pattern_annotation(draw, pattern_match, i)
        
        # 4. Add legend
        self._draw_legend(draw, pil_image.size, feedback_items)
        
        # 5. Add summary information
        self._draw_summary_info(draw, pil_image.size, feedback_items)
        
        # Convert back to numpy array
        annotated_array = np.array(pil_image)
        
        # Save if output path specified
        if output_path:
            pil_image.save(output_path, quality=95, dpi=(300, 300))
            logger.info(f"Annotated image saved to: {output_path}")
        
        return AnnotatedImage(
            image_array=annotated_array,
            annotations=annotations,
            legend=self._create_legend_dict(feedback_items),
            metadata={
                'total_feedback_items': len(feedback_items),
                'total_ocr_elements': len(ocr_elements) if ocr_elements else 0,
                'total_pattern_matches': len(pattern_matches) if pattern_matches else 0,
                'image_dimensions': pil_image.size
            }
        )
    
    def _draw_feedback_annotation(self, draw: ImageDraw.Draw, feedback_item: Dict, 
                                index: int, image_size: Tuple[int, int]):
        """Draw annotation for a feedback item."""
        
        # Get or create bounding box for feedback item
        bbox = self._get_feedback_bbox(feedback_item, image_size)
        if not bbox:
            return
            
        x, y, w, h = bbox
        severity = feedback_item.get('severity', 'minor')
        color = self._get_severity_color(severity)
        
        # Draw bounding box
        bbox_coords = [(x, y), (x + w, y + h)]
        bbox_thickness = self.config.get('annotation_style', {}).get('bbox_thickness', 3)
        draw.rectangle(bbox_coords, outline=color, width=bbox_thickness)
        
        # Draw label background
        label_text = f"{index + 1}. {feedback_item.get('category_name', 'Unknown')}"
        
        # Estimate text size (simple approximation)
        text_width = len(label_text) * 8
        text_height = 20
        
        label_bg_coords = [
            (x, y - text_height - 5),
            (x + text_width + 10, y - 5)
        ]
        draw.rectangle(label_bg_coords, fill=color + (200,))  # Semi-transparent
        
        # Draw label text
        draw.text((x + 5, y - text_height), label_text, 
                 fill=(255, 255, 255), anchor="lt")
        
        # Draw severity indicator
        severity_colors = {'critical': (255, 0, 0), 'major': (255, 165, 0), 'minor': (255, 255, 0)}
        severity_color = severity_colors.get(severity, (128, 128, 128))
        
        # Small severity indicator circle
        circle_coords = [(x + w - 20, y + 5), (x + w - 5, y + 20)]
        draw.ellipse(circle_coords, fill=severity_color, outline=(0, 0, 0))
        
        # Add feedback number in circle
        draw.text((x + w - 12, y + 12), str(index + 1), 
                 fill=(0, 0, 0), anchor="mm")
    
    def _draw_ocr_annotation(self, draw: ImageDraw.Draw, ocr_element, index: int):
        """Draw annotation for OCR detected text."""
        if not hasattr(ocr_element, 'bounding_box'):
            return
            
        x, y, w, h = ocr_element.bounding_box
        
        # Light blue for OCR elements
        ocr_color = (0, 150, 255)
        
        # Draw thin bounding box for OCR
        bbox_coords = [(x, y), (x + w, y + h)]
        draw.rectangle(bbox_coords, outline=ocr_color, width=1)
        
        # Add small text confidence indicator
        if hasattr(ocr_element, 'confidence'):
            conf_text = f"{ocr_element.confidence:.0f}%"
            draw.text((x, y - 12), conf_text, fill=ocr_color, anchor="lt")
    
    def _draw_pattern_annotation(self, draw: ImageDraw.Draw, pattern_match, index: int):
        """Draw annotation for pattern matches."""
        if not hasattr(pattern_match, 'bounding_box'):
            return
            
        # Green for pattern matches
        pattern_color = (0, 200, 0)
        
        # Draw pattern match indicator
        # Implementation depends on pattern match structure
        pass
    
    def _draw_legend(self, draw: ImageDraw.Draw, image_size: Tuple[int, int], 
                   feedback_items: List[Dict]):
        """Draw legend showing severity levels and categories."""
        
        width, height = image_size
        legend_width = 250
        legend_height = 200
        
        # Position legend in top-right corner with margin
        legend_x = width - legend_width - 20
        legend_y = 20
        
        # Draw legend background
        legend_coords = [(legend_x, legend_y), (legend_x + legend_width, legend_y + legend_height)]
        draw.rectangle(legend_coords, fill=(255, 255, 255, 230), outline=(0, 0, 0), width=2)
        
        # Legend title
        draw.text((legend_x + 10, legend_y + 10), "Feedback Legend", 
                 fill=(0, 0, 0), anchor="lt")
        
        # Severity levels
        severities = [
            ('Critical', (255, 0, 0), 'Kritische Mängel'),
            ('Major', (255, 165, 0), 'Größere Mängel'),
            ('Minor', (255, 255, 0), 'Kleinere Mängel')
        ]
        
        y_offset = 40
        for severity, color, german_text in severities:
            # Color box
            color_box = [(legend_x + 15, legend_y + y_offset), 
                        (legend_x + 30, legend_y + y_offset + 15)]
            draw.rectangle(color_box, fill=color, outline=(0, 0, 0))
            
            # Text
            draw.text((legend_x + 40, legend_y + y_offset + 7), 
                     f"{severity} - {german_text}", fill=(0, 0, 0), anchor="lm")
            
            y_offset += 25
        
        # Summary statistics
        critical_count = len([f for f in feedback_items if f.get('severity') == 'critical'])
        major_count = len([f for f in feedback_items if f.get('severity') == 'major'])
        minor_count = len([f for f in feedback_items if f.get('severity') == 'minor'])
        
        summary_y = legend_y + 130
        draw.text((legend_x + 15, summary_y), "Zusammenfassung:", fill=(0, 0, 0), anchor="lt")
        draw.text((legend_x + 15, summary_y + 20), f"Kritisch: {critical_count}", fill=(255, 0, 0), anchor="lt")
        draw.text((legend_x + 15, summary_y + 35), f"Größer: {major_count}", fill=(255, 165, 0), anchor="lt")
        draw.text((legend_x + 15, summary_y + 50), f"Kleiner: {minor_count}", fill=(255, 255, 0), anchor="lt")
    
    def _draw_summary_info(self, draw: ImageDraw.Draw, image_size: Tuple[int, int], 
                         feedback_items: List[Dict]):
        """Draw summary information at the bottom of the image."""
        
        width, height = image_size
        
        # Summary background
        summary_height = 40
        summary_y = height - summary_height - 10
        
        summary_coords = [(10, summary_y), (width - 10, height - 10)]
        draw.rectangle(summary_coords, fill=(0, 0, 0, 180), outline=(255, 255, 255))
        
        # Summary text
        total_items = len(feedback_items)
        summary_text = f"Technische Zeichnung Analyse - {total_items} Feedback-Elemente erkannt"
        
        draw.text((width // 2, summary_y + 20), summary_text, 
                 fill=(255, 255, 255), anchor="mm")
    
    def _get_feedback_bbox(self, feedback_item: Dict, image_size: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        """Get or generate bounding box for feedback item."""
        
        # Check if position is already provided
        if 'position' in feedback_item and feedback_item['position']:
            pos = feedback_item['position']
            if isinstance(pos, (list, tuple)) and len(pos) >= 4:
                return tuple(pos[:4])
        
        # Generate bounding box based on category or other heuristics
        width, height = image_size
        category = feedback_item.get('category_name', 'Unknown')
        
        # Simple heuristic placement based on category
        category_positions = {
            'Title_Block': (width - 300, height - 150, 280, 130),
            'Schriftfeld': (width - 300, height - 150, 280, 130),
            'Dimensioning': (width // 3, height // 3, 100, 50),
            'Tolerancing': (width // 2, height // 2, 120, 60),
            'Surface_Finish': (width // 4, height // 4, 80, 40),
            'General': (50, 50, 150, 75),
            'Allgemein': (50, 50, 150, 75)
        }
        
        return category_positions.get(category, (50, 50, 150, 75))
    
    def _get_severity_color(self, severity: str) -> Tuple[int, int, int]:
        """Get color for severity level."""
        severity_colors = {
            'critical': (255, 0, 0),    # Red
            'major': (255, 165, 0),     # Orange  
            'minor': (255, 255, 0),     # Yellow
            'info': (0, 191, 255)       # Deep Sky Blue
        }
        return severity_colors.get(severity, (128, 128, 128))  # Gray default
    
    def _create_legend_dict(self, feedback_items: List[Dict]) -> Dict[str, Tuple[int, int, int]]:
        """Create legend dictionary for the annotation."""
        legend = {}
        
        for item in feedback_items:
            severity = item.get('severity', 'minor')
            category = item.get('category_name', 'Unknown')
            color = self._get_severity_color(severity)
            legend[f"{category} ({severity})"] = color
            
        return legend
    
    def _create_color_map(self) -> Dict:
        """Create color mapping for different annotation types."""
        return {
            **self.config['colors'],
            **self.config['category_colors']
        }
    
    def annotate_with_bounding_boxes(self, image: np.ndarray, 
                                   bounding_boxes: List,
                                   feedback_items: Optional[List] = None,
                                   image_id: str = "unknown") -> AnnotatedImage:
        """
        Annotate image with bounding boxes and feedback.
        
        Args:
            image: Input image
            bounding_boxes: List of BoundingBox objects  
            feedback_items: Optional list of feedback items
            image_id: Image identifier
            
        Returns:
            AnnotatedImage with annotations applied
        """
        # Create copy of image for annotation
        annotated_image = image.copy()
        annotations = []
        
        # Create feedback lookup by position
        feedback_lookup = {}
        if feedback_items:
            for feedback in feedback_items:
                if hasattr(feedback, 'position') and feedback.position:
                    feedback_lookup[feedback.position] = feedback
        
        # Process each bounding box
        for i, bbox in enumerate(bounding_boxes):
            # Convert to pixel coordinates
            img_height, img_width = image.shape[:2]
            x1, y1, x2, y2 = bbox.to_pixel_coords(img_width, img_height)
            
            # Get category color
            category_color = self.color_map.get(bbox.class_id, self.color_map[0])
            
            # Find corresponding feedback
            feedback_text = None
            severity = "unknown"
            
            # Simple position-based matching (would be more sophisticated in practice)
            bbox_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            for pos, feedback in feedback_lookup.items():
                if abs(pos[0] - bbox_center[0]) < 50 and abs(pos[1] - bbox_center[1]) < 50:
                    feedback_text = feedback.feedback_text
                    severity = feedback.severity
                    break
            
            # Create annotation
            annotation = Annotation(
                annotation_id=f"bbox_{i}",
                bounding_box=(x1, y1, x2-x1, y2-y1),
                label=f"{bbox.class_id}: {bbox.get_category_name()}",
                color=category_color,
                severity=severity,
                feedback_text=feedback_text
            )
            
            annotations.append(annotation)
            
            # Draw bounding box
            annotated_image = self._draw_bounding_box(
                annotated_image, annotation
            )
        
        # Add legend
        annotated_image = self._add_legend(annotated_image, annotations)
        
        # Create metadata
        metadata = {
            'image_id': image_id,
            'total_annotations': len(annotations),
            'annotation_types': list(set(ann.severity for ann in annotations)),
            'categories_present': list(set(bbox.class_id for bbox in bounding_boxes))
        }
        
        return AnnotatedImage(
            image_array=annotated_image,
            annotations=annotations,
            legend=self._create_legend_data(annotations),
            metadata=metadata
        )
    
    def _draw_bounding_box(self, image: np.ndarray, annotation: Annotation) -> np.ndarray:
        """Draw a single bounding box with label."""
        x, y, w, h = annotation.bounding_box
        color = annotation.color
        
        style = self.config['annotation_style']
        
        # Draw rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), color, style['bbox_thickness'])
        
        # Prepare label text
        label_text = annotation.label
        if len(label_text) > 20:  # Truncate long labels
            label_text = label_text[:17] + "..."
        
        # Calculate text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        text_thickness = 2
        
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, font, font_scale, text_thickness
        )
        
        # Draw text background
        text_x = x + style['label_offset'][0]
        text_y = y + style['label_offset'][1]
        
        # Ensure text is within image bounds
        text_x = max(0, min(text_x, image.shape[1] - text_width - 10))
        text_y = max(text_height + 5, text_y)
        
        # Draw filled rectangle for text background
        cv2.rectangle(
            image,
            (text_x - 5, text_y - text_height - 5),
            (text_x + text_width + 5, text_y + baseline + 5),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            image,
            label_text,
            (text_x, text_y),
            font,
            font_scale,
            style['text_color'],
            text_thickness
        )
        
        return image
    
    def _add_legend(self, image: np.ndarray, annotations: List[Annotation]) -> np.ndarray:
        """Add legend to the annotated image."""
        legend_config = self.config['legend']
        
        # Get unique categories and severities
        categories = {}
        severities = {}
        
        for annotation in annotations:
            # Parse category from label
            if ':' in annotation.label:
                cat_part = annotation.label.split(':')[1].strip()
                categories[cat_part] = annotation.color
            
            if annotation.severity != "unknown":
                sev_color = self.config['colors'].get(annotation.severity, (128, 128, 128))
                severities[annotation.severity] = sev_color
        
        if not categories and not severities:
            return image
        
        # Calculate legend dimensions
        legend_width = legend_config['width']
        line_height = 25
        total_items = len(categories) + len(severities) + 2  # +2 for headers
        legend_height = max(legend_config['height'], total_items * line_height + 40)
        
        # Position legend
        img_height, img_width = image.shape[:2]
        legend_x = img_width - legend_width - 10
        legend_y = 10
        
        # Create legend overlay
        overlay = image.copy()
        
        # Draw legend background
        cv2.rectangle(
            overlay,
            (legend_x, legend_y),
            (legend_x + legend_width, legend_y + legend_height),
            legend_config['background_color'],
            -1
        )
        
        # Draw legend border
        cv2.rectangle(
            overlay,
            (legend_x, legend_y),
            (legend_x + legend_width, legend_y + legend_height),
            legend_config['border_color'],
            2
        )
        
        # Add legend content
        current_y = legend_y + 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        
        # Categories header
        if categories:
            cv2.putText(
                overlay,
                "Kategorien:",
                (legend_x + 10, current_y),
                font, font_scale, (0, 0, 0), font_thickness
            )
            current_y += line_height
            
            # Category entries
            for category, color in categories.items():
                # Draw color box
                cv2.rectangle(
                    overlay,
                    (legend_x + 15, current_y - 12),
                    (legend_x + 30, current_y - 2),
                    color,
                    -1
                )
                
                # Draw category name (truncated)
                cat_name = category[:15] + "..." if len(category) > 15 else category
                cv2.putText(
                    overlay,
                    cat_name,
                    (legend_x + 35, current_y - 2),
                    font, font_scale, (0, 0, 0), font_thickness
                )
                current_y += line_height
        
        # Severity header
        if severities:
            current_y += 10
            cv2.putText(
                overlay,
                "Schweregrad:",
                (legend_x + 10, current_y),
                font, font_scale, (0, 0, 0), font_thickness
            )
            current_y += line_height
            
            # Severity entries
            for severity, color in severities.items():
                # Draw color box
                cv2.rectangle(
                    overlay,
                    (legend_x + 15, current_y - 12),
                    (legend_x + 30, current_y - 2),
                    color,
                    -1
                )
                
                # Draw severity name
                sev_names = {
                    'critical': 'Kritisch',
                    'major': 'Größer',
                    'minor': 'Kleiner'
                }
                sev_name = sev_names.get(severity, severity.capitalize())
                
                cv2.putText(
                    overlay,
                    sev_name,
                    (legend_x + 35, current_y - 2),
                    font, font_scale, (0, 0, 0), font_thickness
                )
                current_y += line_height
        
        # Blend overlay with original image
        alpha = legend_config.get('alpha', 0.9)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        return image
    
    def _create_legend_data(self, annotations: List[Annotation]) -> Dict:
        """Create legend data for export."""
        legend_data = {
            'categories': {},
            'severities': {},
            'color_mapping': self.color_map
        }
        
        for annotation in annotations:
            # Extract category info
            if ':' in annotation.label:
                cat_part = annotation.label.split(':')[1].strip()
                legend_data['categories'][cat_part] = annotation.color
            
            # Extract severity info
            if annotation.severity != "unknown":
                legend_data['severities'][annotation.severity] = self.config['colors'].get(
                    annotation.severity, (128, 128, 128)
                )
        
        return legend_data
    
    def create_annotated_pdf(self, annotated_images: List[AnnotatedImage],
                           output_path: Union[str, Path]) -> bool:
        """
        Create annotated PDF from multiple annotated images.
        
        Args:
            annotated_images: List of AnnotatedImage objects
            output_path: Path to save PDF
            
        Returns:
            True if creation successful
        """
        try:
            from PIL import Image
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert images to PIL format
            pil_images = []
            for annotated_img in annotated_images:
                # Convert BGR to RGB if needed
                if len(annotated_img.image_array.shape) == 3:
                    rgb_image = cv2.cvtColor(annotated_img.image_array, cv2.COLOR_BGR2RGB)
                else:
                    rgb_image = annotated_img.image_array
                
                pil_image = Image.fromarray(rgb_image)
                pil_images.append(pil_image)
            
            # Save as PDF
            if pil_images:
                pil_images[0].save(
                    output_path,
                    save_all=True,
                    append_images=pil_images[1:],
                    format='PDF'
                )
                
                logger.info(f"Created annotated PDF: {output_path}")
                return True
            
        except Exception as e:
            logger.error(f"Error creating annotated PDF: {e}")
            
        return False
    
    def save_annotated_image(self, annotated_image: AnnotatedImage,
                           output_path: Union[str, Path],
                           save_metadata: bool = True) -> bool:
        """
        Save annotated image to disk.
        
        Args:
            annotated_image: AnnotatedImage to save
            output_path: Path to save image
            save_metadata: Whether to save annotation metadata
            
        Returns:
            True if save successful
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save image
            cv2.imwrite(str(output_path), annotated_image.image_array)
            
            # Save metadata if requested
            if save_metadata:
                metadata_path = output_path.with_suffix('.json')
                
                metadata = {
                    'image_metadata': annotated_image.metadata,
                    'annotations': [
                        {
                            'annotation_id': ann.annotation_id,
                            'bounding_box': ann.bounding_box,
                            'label': ann.label,
                            'color': ann.color,
                            'severity': ann.severity,
                            'feedback_text': ann.feedback_text
                        }
                        for ann in annotated_image.annotations
                    ],
                    'legend': annotated_image.legend
                }
                
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Saved annotation metadata to {metadata_path}")
            
            logger.info(f"Saved annotated image to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving annotated image: {e}")
            return False


def main():
    """Test the visual annotation tool."""
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent / "utils"))
    from data_pipeline import DataPipeline
    
    # Initialize tool
    tool = VisualAnnotationTool()
    
    # Test with dataset
    dataset_root = Path(__file__).parent.parent.parent / "dataset"
    
    if dataset_root.exists():
        print(f"\nTesting Visual Annotation Tool...")
        
        # Load a sample drawing with bounding boxes
        pipeline = DataPipeline(str(dataset_root))
        drawings_data = pipeline.load_all_drawings()
        
        if drawings_data:
            # Take first drawing as test
            drawing_id = list(drawings_data.keys())[0]
            drawing_data = drawings_data[drawing_id]
            
            print(f"Annotating drawing: {drawing_id}")
            print(f"  Bounding boxes: {len(drawing_data.bounding_boxes)}")
            print(f"  Feedback items: {len(drawing_data.feedback_items)}")
            
            # Load images
            corrected_img, uncorrected_img = drawing_data.load_images()
            
            if corrected_img is not None:
                # Create annotations
                annotated_image = tool.annotate_with_bounding_boxes(
                    corrected_img,
                    drawing_data.bounding_boxes,
                    drawing_data.feedback_items,
                    drawing_id
                )
                
                print(f"  Created {len(annotated_image.annotations)} annotations")
                print(f"  Legend categories: {len(annotated_image.legend['categories'])}")
                
                # Save annotated image
                output_dir = Path(__file__).parent.parent.parent / "output" / "annotated"
                output_path = output_dir / f"{drawing_id}_annotated.jpg"
                
                success = tool.save_annotated_image(annotated_image, output_path)
                if success:
                    print(f"  Saved to: {output_path}")
                    
            else:
                print("  Could not load image")
        else:
            print("No drawings data loaded")
    else:
        print("Dataset directory not found. Please run from project root.")


if __name__ == "__main__":
    main()