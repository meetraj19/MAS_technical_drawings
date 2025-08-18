#!/usr/bin/env python3
"""
Data Pipeline for Technical Drawing Feedback System

This module provides utilities for loading, parsing, and processing
the German technical drawing feedback data and bounding boxes.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import cv2
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class TechnicalDrawingCategory:
    """Enum-like class for technical drawing categories."""
    
    UNKNOWN = 0
    SCHRAUBENKOPF = 2  # Screw Head
    SCHEIBE = 3        # Washer  
    PLATTE = 4         # Plate
    GEWINDERESERVE = 5 # Thread Reserve
    GRUNDLOCH = 6      # Pilot Hole
    GEWINDEDARSTELLUNG = 7  # Thread Representation
    SCHRAFFUR = 8      # Hatching
    SCHRIFTFELD = 9    # Title Block
    
    @classmethod
    def get_category_name(cls, category_id: int) -> str:
        """Get German category name by ID."""
        category_map = {
            0: "Unknown",
            2: "Schraubenkopf (Screw Head)",
            3: "Scheibe (Washer)",
            4: "Platte (Plate)",
            5: "Gewindereserve (Thread Reserve)",
            6: "Grundloch (Pilot Hole)",
            7: "Gewindedarstellung (Thread Representation)",
            8: "Schraffur (Hatching)",
            9: "Schriftfeld (Title Block)"
        }
        return category_map.get(category_id, "Unknown")


class BoundingBox:
    """Represents a bounding box in YOLO format."""
    
    def __init__(self, class_id: int, center_x: float, center_y: float, 
                 width: float, height: float, confidence: float = 1.0):
        """
        Initialize bounding box.
        
        Args:
            class_id: Category class ID
            center_x: Normalized center X coordinate (0-1)
            center_y: Normalized center Y coordinate (0-1)
            width: Normalized width (0-1)
            height: Normalized height (0-1)
            confidence: Confidence score (0-1)
        """
        self.class_id = class_id
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height
        self.confidence = confidence
        
    def to_pixel_coords(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """
        Convert normalized coordinates to pixel coordinates.
        
        Args:
            img_width: Image width in pixels
            img_height: Image height in pixels
            
        Returns:
            Tuple of (x1, y1, x2, y2) in pixel coordinates
        """
        center_x_px = int(self.center_x * img_width)
        center_y_px = int(self.center_y * img_height)
        width_px = int(self.width * img_width)
        height_px = int(self.height * img_height)
        
        x1 = center_x_px - width_px // 2
        y1 = center_y_px - height_px // 2
        x2 = center_x_px + width_px // 2
        y2 = center_y_px + height_px // 2
        
        return (x1, y1, x2, y2)
    
    def get_category_name(self) -> str:
        """Get category name for this bounding box."""
        return TechnicalDrawingCategory.get_category_name(self.class_id)
    
    def __repr__(self) -> str:
        return (f"BoundingBox(class_id={self.class_id}, "
                f"center=({self.center_x:.3f}, {self.center_y:.3f}), "
                f"size=({self.width:.3f}, {self.height:.3f}), "
                f"category='{self.get_category_name()}')")


class FeedbackItem:
    """Represents a feedback item for a technical drawing."""
    
    def __init__(self, category_id: int, element_description: str, 
                 feedback_text: str, status: str = "Unknown"):
        """
        Initialize feedback item.
        
        Args:
            category_id: Category class ID
            element_description: Description of the element
            feedback_text: Feedback text
            status: Status (e.g., "Sicher" for correct)
        """
        self.category_id = category_id
        self.element_description = element_description
        self.feedback_text = feedback_text
        self.status = status
        
    def get_category_name(self) -> str:
        """Get category name for this feedback."""
        return TechnicalDrawingCategory.get_category_name(self.category_id)
    
    def is_correct(self) -> bool:
        """Check if this feedback indicates correct implementation."""
        return "Sicher" in self.status or "korrekt" in self.feedback_text.lower()
    
    def __repr__(self) -> str:
        return (f"FeedbackItem(category={self.category_id}, "
                f"element='{self.element_description}', "
                f"status='{self.status}')")


class TechnicalDrawingData:
    """Represents data for a single technical drawing."""
    
    def __init__(self, drawing_id: str, corrected_image_path: str,
                 uncorrected_image_path: str):
        """
        Initialize technical drawing data.
        
        Args:
            drawing_id: Unique identifier for the drawing
            corrected_image_path: Path to corrected image
            uncorrected_image_path: Path to uncorrected image
        """
        self.drawing_id = drawing_id
        self.corrected_image_path = Path(corrected_image_path)
        self.uncorrected_image_path = Path(uncorrected_image_path)
        self.feedback_items: List[FeedbackItem] = []
        self.bounding_boxes: List[BoundingBox] = []
        
    def add_feedback(self, feedback_item: FeedbackItem):
        """Add a feedback item."""
        self.feedback_items.append(feedback_item)
        
    def add_bounding_box(self, bbox: BoundingBox):
        """Add a bounding box."""
        self.bounding_boxes.append(bbox)
        
    def get_feedback_by_category(self, category_id: int) -> List[FeedbackItem]:
        """Get all feedback items for a specific category."""
        return [item for item in self.feedback_items if item.category_id == category_id]
    
    def get_bboxes_by_category(self, category_id: int) -> List[BoundingBox]:
        """Get all bounding boxes for a specific category."""
        return [bbox for bbox in self.bounding_boxes if bbox.class_id == category_id]
    
    def get_category_summary(self) -> Dict[int, Dict]:
        """Get summary of categories present in this drawing."""
        summary = {}
        
        # Count feedback items by category
        feedback_counts = {}
        for item in self.feedback_items:
            feedback_counts[item.category_id] = feedback_counts.get(item.category_id, 0) + 1
            
        # Count bounding boxes by category
        bbox_counts = {}
        for bbox in self.bounding_boxes:
            bbox_counts[bbox.class_id] = bbox_counts.get(bbox.class_id, 0) + 1
            
        # Combine information
        all_categories = set(feedback_counts.keys()) | set(bbox_counts.keys())
        for category_id in all_categories:
            summary[category_id] = {
                "name": TechnicalDrawingCategory.get_category_name(category_id),
                "feedback_count": feedback_counts.get(category_id, 0),
                "bbox_count": bbox_counts.get(category_id, 0),
                "has_annotations": category_id in bbox_counts
            }
            
        return summary
    
    def load_images(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load corrected and uncorrected images.
        
        Returns:
            Tuple of (corrected_image, uncorrected_image) as numpy arrays
        """
        corrected_img = None
        uncorrected_img = None
        
        try:
            if self.corrected_image_path.exists():
                corrected_img = cv2.imread(str(self.corrected_image_path))
                corrected_img = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Error loading corrected image {self.corrected_image_path}: {e}")
            
        try:
            if self.uncorrected_image_path.exists():
                uncorrected_img = cv2.imread(str(self.uncorrected_image_path))
                uncorrected_img = cv2.cvtColor(uncorrected_img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Error loading uncorrected image {self.uncorrected_image_path}: {e}")
            
        return corrected_img, uncorrected_img
    
    def __repr__(self) -> str:
        return (f"TechnicalDrawingData(id='{self.drawing_id}', "
                f"feedback_items={len(self.feedback_items)}, "
                f"bounding_boxes={len(self.bounding_boxes)})")


class DataPipeline:
    """Main data pipeline for loading and processing technical drawing data."""
    
    def __init__(self, dataset_root: str):
        """
        Initialize data pipeline.
        
        Args:
            dataset_root: Root directory of the dataset
        """
        self.dataset_root = Path(dataset_root)
        self.metadata_file = self.dataset_root / "metadata.json"
        self.drawings_data: Dict[str, TechnicalDrawingData] = {}
        
    def load_metadata(self) -> bool:
        """Load dataset metadata."""
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded metadata for {len(self.metadata['image_pairs'])} image pairs")
            return True
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return False
    
    def parse_feedback_text(self, feedback_text: str) -> List[FeedbackItem]:
        """
        Parse German feedback text into structured feedback items.
        
        Args:
            feedback_text: Raw German feedback text
            
        Returns:
            List of FeedbackItem objects
        """
        feedback_items = []
        lines = feedback_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Try to extract category number and content
            parts = line.split(' ', 1)
            if len(parts) >= 2 and parts[0].isdigit():
                category_id = int(parts[0])
                content = parts[1]
                
                # Parse the content for element description and feedback
                # German format: "Element-Feedback-Status"
                content_parts = content.split('-')
                
                if len(content_parts) >= 2:
                    element_description = content_parts[0].strip()
                    feedback_text = '-'.join(content_parts[1:-1]).strip() if len(content_parts) > 2 else content_parts[1].strip()
                    status = content_parts[-1].strip() if len(content_parts) > 2 else "Unknown"
                    
                    feedback_item = FeedbackItem(
                        category_id=category_id,
                        element_description=element_description,
                        feedback_text=feedback_text,
                        status=status
                    )
                    feedback_items.append(feedback_item)
                    
        return feedback_items
    
    def parse_bounding_boxes(self, bbox_text: str) -> List[BoundingBox]:
        """
        Parse YOLO format bounding boxes.
        
        Args:
            bbox_text: Raw bounding box text
            
        Returns:
            List of BoundingBox objects
        """
        bounding_boxes = []
        lines = bbox_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) == 5:
                try:
                    class_id = int(parts[0])
                    center_x = float(parts[1])
                    center_y = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    bbox = BoundingBox(
                        class_id=class_id,
                        center_x=center_x,
                        center_y=center_y,
                        width=width,
                        height=height
                    )
                    bounding_boxes.append(bbox)
                    
                except ValueError as e:
                    logger.warning(f"Invalid bounding box format: {line} - {e}")
                    
        return bounding_boxes
    
    def load_drawing_data(self, drawing_id: str) -> Optional[TechnicalDrawingData]:
        """
        Load complete data for a specific drawing.
        
        Args:
            drawing_id: ID of the drawing to load
            
        Returns:
            TechnicalDrawingData object or None if not found
        """
        # Find the drawing in metadata
        drawing_metadata = None
        for pair in self.metadata.get('image_pairs', []):
            if pair['id'] == drawing_id:
                drawing_metadata = pair
                break
                
        if not drawing_metadata:
            logger.error(f"Drawing {drawing_id} not found in metadata")
            return None
            
        # Create drawing data object
        drawing_data = TechnicalDrawingData(
            drawing_id=drawing_id,
            corrected_image_path=drawing_metadata['corrected_image'],
            uncorrected_image_path=drawing_metadata['uncorrected_image']
        )
        
        # Load feedback data
        feedback_file = drawing_metadata.get('feedback_file')
        if feedback_file and Path(feedback_file).exists():
            try:
                with open(feedback_file, 'r', encoding='utf-8') as f:
                    feedback_text = f.read()
                feedback_items = self.parse_feedback_text(feedback_text)
                for item in feedback_items:
                    drawing_data.add_feedback(item)
                logger.info(f"Loaded {len(feedback_items)} feedback items for drawing {drawing_id}")
            except Exception as e:
                logger.error(f"Error loading feedback for drawing {drawing_id}: {e}")
                
        # Load bounding box data
        bbox_file = drawing_metadata.get('bbox_file')
        if bbox_file and Path(bbox_file).exists():
            try:
                with open(bbox_file, 'r') as f:
                    bbox_text = f.read()
                bounding_boxes = self.parse_bounding_boxes(bbox_text)
                for bbox in bounding_boxes:
                    drawing_data.add_bounding_box(bbox)
                logger.info(f"Loaded {len(bounding_boxes)} bounding boxes for drawing {drawing_id}")
            except Exception as e:
                logger.error(f"Error loading bounding boxes for drawing {drawing_id}: {e}")
                
        return drawing_data
    
    def load_all_drawings(self) -> Dict[str, TechnicalDrawingData]:
        """
        Load data for all drawings in the dataset.
        
        Returns:
            Dictionary mapping drawing IDs to TechnicalDrawingData objects
        """
        if not self.load_metadata():
            return {}
            
        self.drawings_data = {}
        
        for pair in self.metadata.get('image_pairs', []):
            drawing_id = pair['id']
            drawing_data = self.load_drawing_data(drawing_id)
            if drawing_data:
                self.drawings_data[drawing_id] = drawing_data
                
        logger.info(f"Loaded {len(self.drawings_data)} drawings successfully")
        return self.drawings_data
    
    def get_dataset_statistics(self) -> Dict:
        """Get comprehensive statistics about the dataset."""
        if not self.drawings_data:
            self.load_all_drawings()
            
        stats = {
            "total_drawings": len(self.drawings_data),
            "total_feedback_items": 0,
            "total_bounding_boxes": 0,
            "category_distribution": {},
            "feedback_status_distribution": {},
            "drawings_by_category": {}
        }
        
        for drawing_data in self.drawings_data.values():
            stats["total_feedback_items"] += len(drawing_data.feedback_items)
            stats["total_bounding_boxes"] += len(drawing_data.bounding_boxes)
            
            # Category distribution
            for feedback in drawing_data.feedback_items:
                cat_name = feedback.get_category_name()
                stats["category_distribution"][cat_name] = stats["category_distribution"].get(cat_name, 0) + 1
                
                # Track which drawings have each category
                if cat_name not in stats["drawings_by_category"]:
                    stats["drawings_by_category"][cat_name] = []
                if drawing_data.drawing_id not in stats["drawings_by_category"][cat_name]:
                    stats["drawings_by_category"][cat_name].append(drawing_data.drawing_id)
                    
            # Status distribution
            for feedback in drawing_data.feedback_items:
                status = "Correct" if feedback.is_correct() else "Needs Correction"
                stats["feedback_status_distribution"][status] = stats["feedback_status_distribution"].get(status, 0) + 1
                
        return stats


def main():
    """Test the data pipeline."""
    # Initialize pipeline
    dataset_root = Path(__file__).parent.parent.parent / "dataset"
    pipeline = DataPipeline(str(dataset_root))
    
    # Load all drawings
    drawings = pipeline.load_all_drawings()
    
    # Print statistics
    stats = pipeline.get_dataset_statistics()
    print("\nDataset Statistics:")
    print(f"Total drawings: {stats['total_drawings']}")
    print(f"Total feedback items: {stats['total_feedback_items']}")
    print(f"Total bounding boxes: {stats['total_bounding_boxes']}")
    
    print("\nCategory distribution:")
    for category, count in stats['category_distribution'].items():
        print(f"  {category}: {count}")
        
    print("\nFeedback status distribution:")
    for status, count in stats['feedback_status_distribution'].items():
        print(f"  {status}: {count}")
    
    # Test loading a specific drawing
    if drawings:
        first_drawing_id = list(drawings.keys())[0]
        print(f"\nExample drawing ({first_drawing_id}):")
        drawing = drawings[first_drawing_id]
        print(f"  Feedback items: {len(drawing.feedback_items)}")
        print(f"  Bounding boxes: {len(drawing.bounding_boxes)}")
        
        category_summary = drawing.get_category_summary()
        print("  Categories present:")
        for cat_id, info in category_summary.items():
            print(f"    {info['name']}: {info['feedback_count']} feedback, {info['bbox_count']} bboxes")


if __name__ == "__main__":
    main()