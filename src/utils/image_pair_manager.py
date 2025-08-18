#!/usr/bin/env python3
"""
Image Pair Manager for Technical Drawing Feedback System

This module manages the relationships between corrected and uncorrected
technical drawing images, enabling comparison and analysis.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
from PIL import Image

from .data_pipeline import DataPipeline, TechnicalDrawingData, BoundingBox

logger = logging.getLogger(__name__)


@dataclass
class ImageComparisonResult:
    """Results from comparing corrected and uncorrected images."""
    drawing_id: str
    similarity_score: float
    difference_regions: List[Dict]
    corrected_features: Dict
    uncorrected_features: Dict
    recommendations: List[str]


class ImagePairManager:
    """Manages corrected/uncorrected image pairs for analysis."""
    
    def __init__(self, dataset_root: str):
        """
        Initialize image pair manager.
        
        Args:
            dataset_root: Root directory of the dataset
        """
        self.dataset_root = Path(dataset_root)
        self.data_pipeline = DataPipeline(str(dataset_root))
        self.drawings_data: Dict[str, TechnicalDrawingData] = {}
        
    def load_all_pairs(self) -> bool:
        """Load all image pairs from the dataset."""
        try:
            self.drawings_data = self.data_pipeline.load_all_drawings()
            logger.info(f"Loaded {len(self.drawings_data)} image pairs")
            return True
        except Exception as e:
            logger.error(f"Error loading image pairs: {e}")
            return False
    
    def get_pair_by_id(self, drawing_id: str) -> Optional[TechnicalDrawingData]:
        """
        Get image pair data by drawing ID.
        
        Args:
            drawing_id: ID of the drawing pair
            
        Returns:
            TechnicalDrawingData object or None if not found
        """
        return self.drawings_data.get(drawing_id)
    
    def get_all_pair_ids(self) -> List[str]:
        """Get list of all available drawing pair IDs."""
        return list(self.drawings_data.keys())
    
    def validate_pair(self, drawing_id: str) -> Dict[str, bool]:
        """
        Validate that both images in a pair exist and are loadable.
        
        Args:
            drawing_id: ID of the drawing pair to validate
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            "pair_exists": False,
            "corrected_image_exists": False,
            "uncorrected_image_exists": False,
            "corrected_image_loadable": False,
            "uncorrected_image_loadable": False,
            "has_feedback": False,
            "has_bboxes": False
        }
        
        drawing_data = self.get_pair_by_id(drawing_id)
        if not drawing_data:
            return validation
            
        validation["pair_exists"] = True
        
        # Check if image files exist
        validation["corrected_image_exists"] = drawing_data.corrected_image_path.exists()
        validation["uncorrected_image_exists"] = drawing_data.uncorrected_image_path.exists()
        
        # Try to load images
        try:
            corrected_img, uncorrected_img = drawing_data.load_images()
            validation["corrected_image_loadable"] = corrected_img is not None
            validation["uncorrected_image_loadable"] = uncorrected_img is not None
        except Exception as e:
            logger.error(f"Error loading images for {drawing_id}: {e}")
            
        # Check for feedback and bounding boxes
        validation["has_feedback"] = len(drawing_data.feedback_items) > 0
        validation["has_bboxes"] = len(drawing_data.bounding_boxes) > 0
        
        return validation
    
    def get_image_dimensions(self, drawing_id: str) -> Optional[Tuple[int, int, int, int]]:
        """
        Get dimensions of both images in a pair.
        
        Args:
            drawing_id: ID of the drawing pair
            
        Returns:
            Tuple of (corrected_height, corrected_width, uncorrected_height, uncorrected_width)
            or None if images cannot be loaded
        """
        drawing_data = self.get_pair_by_id(drawing_id)
        if not drawing_data:
            return None
            
        try:
            corrected_img, uncorrected_img = drawing_data.load_images()
            
            if corrected_img is not None and uncorrected_img is not None:
                return (corrected_img.shape[0], corrected_img.shape[1],
                       uncorrected_img.shape[0], uncorrected_img.shape[1])
        except Exception as e:
            logger.error(f"Error getting dimensions for {drawing_id}: {e}")
            
        return None
    
    def calculate_image_similarity(self, drawing_id: str) -> Optional[float]:
        """
        Calculate similarity between corrected and uncorrected images.
        
        Args:
            drawing_id: ID of the drawing pair
            
        Returns:
            Similarity score (0.0 to 1.0) or None if calculation fails
        """
        drawing_data = self.get_pair_by_id(drawing_id)
        if not drawing_data:
            return None
            
        try:
            corrected_img, uncorrected_img = drawing_data.load_images()
            
            if corrected_img is None or uncorrected_img is None:
                return None
                
            # Resize images to same size for comparison
            target_size = (800, 600)  # Standard size for comparison
            corrected_resized = cv2.resize(corrected_img, target_size)
            uncorrected_resized = cv2.resize(uncorrected_img, target_size)
            
            # Convert to grayscale
            corrected_gray = cv2.cvtColor(corrected_resized, cv2.COLOR_RGB2GRAY)
            uncorrected_gray = cv2.cvtColor(uncorrected_resized, cv2.COLOR_RGB2GRAY)
            
            # Calculate structural similarity
            from skimage.metrics import structural_similarity as ssim
            similarity_score = ssim(corrected_gray, uncorrected_gray)
            
            return float(similarity_score)
            
        except Exception as e:
            logger.error(f"Error calculating similarity for {drawing_id}: {e}")
            return None
    
    def find_difference_regions(self, drawing_id: str, threshold: float = 0.1) -> List[Dict]:
        """
        Find regions where corrected and uncorrected images differ significantly.
        
        Args:
            drawing_id: ID of the drawing pair
            threshold: Threshold for considering regions as different
            
        Returns:
            List of difference region dictionaries
        """
        drawing_data = self.get_pair_by_id(drawing_id)
        if not drawing_data:
            return []
            
        try:
            corrected_img, uncorrected_img = drawing_data.load_images()
            
            if corrected_img is None or uncorrected_img is None:
                return []
                
            # Resize to same dimensions
            h, w = min(corrected_img.shape[0], uncorrected_img.shape[0]), \
                   min(corrected_img.shape[1], uncorrected_img.shape[1])
            
            corrected_resized = cv2.resize(corrected_img, (w, h))
            uncorrected_resized = cv2.resize(uncorrected_img, (w, h))
            
            # Convert to grayscale
            corrected_gray = cv2.cvtColor(corrected_resized, cv2.COLOR_RGB2GRAY)
            uncorrected_gray = cv2.cvtColor(uncorrected_resized, cv2.COLOR_RGB2GRAY)
            
            # Calculate absolute difference
            diff = cv2.absdiff(corrected_gray, uncorrected_gray)
            
            # Apply threshold
            _, thresh = cv2.threshold(diff, threshold * 255, 255, cv2.THRESH_BINARY)
            
            # Find contours of difference regions
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            difference_regions = []
            for i, contour in enumerate(contours):
                # Filter out small regions
                area = cv2.contourArea(contour)
                if area > 100:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    difference_regions.append({
                        "region_id": i,
                        "bounding_box": {
                            "x": int(x),
                            "y": int(y),
                            "width": int(w),
                            "height": int(h)
                        },
                        "area": float(area),
                        "center": {
                            "x": int(x + w // 2),
                            "y": int(y + h // 2)
                        }
                    })
                    
            return difference_regions
            
        except Exception as e:
            logger.error(f"Error finding difference regions for {drawing_id}: {e}")
            return []
    
    def compare_bounding_boxes(self, drawing_id: str) -> Dict:
        """
        Compare bounding boxes with actual difference regions.
        
        Args:
            drawing_id: ID of the drawing pair
            
        Returns:
            Comparison results dictionary
        """
        drawing_data = self.get_pair_by_id(drawing_id)
        if not drawing_data:
            return {}
            
        # Get difference regions
        diff_regions = self.find_difference_regions(drawing_id)
        
        # Get image dimensions for bbox conversion
        dimensions = self.get_image_dimensions(drawing_id)
        if not dimensions:
            return {}
            
        corrected_h, corrected_w = dimensions[:2]
        
        comparison_results = {
            "drawing_id": drawing_id,
            "total_bboxes": len(drawing_data.bounding_boxes),
            "total_diff_regions": len(diff_regions),
            "matched_regions": [],
            "unmatched_bboxes": [],
            "unmatched_diff_regions": []
        }
        
        # Convert bounding boxes to pixel coordinates
        pixel_bboxes = []
        for bbox in drawing_data.bounding_boxes:
            x1, y1, x2, y2 = bbox.to_pixel_coords(corrected_w, corrected_h)
            pixel_bboxes.append({
                "bbox": bbox,
                "pixel_coords": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "center": {"x": (x1 + x2) // 2, "y": (y1 + y2) // 2}
            })
        
        # Match bounding boxes with difference regions
        matched_bboxes = set()
        matched_regions = set()
        
        for i, bbox_data in enumerate(pixel_bboxes):
            bbox_center = bbox_data["center"]
            
            for j, diff_region in enumerate(diff_regions):
                diff_center = diff_region["center"]
                
                # Calculate distance between centers
                distance = np.sqrt((bbox_center["x"] - diff_center["x"])**2 + 
                                 (bbox_center["y"] - diff_center["y"])**2)
                
                # If centers are close, consider them matched
                if distance < 100:  # Threshold in pixels
                    comparison_results["matched_regions"].append({
                        "bbox_index": i,
                        "diff_region_index": j,
                        "distance": float(distance),
                        "bbox_category": bbox_data["bbox"].get_category_name(),
                        "bbox_coords": bbox_data["pixel_coords"],
                        "diff_region": diff_region
                    })
                    matched_bboxes.add(i)
                    matched_regions.add(j)
                    break
        
        # Record unmatched items
        for i, bbox_data in enumerate(pixel_bboxes):
            if i not in matched_bboxes:
                comparison_results["unmatched_bboxes"].append({
                    "bbox_index": i,
                    "bbox_category": bbox_data["bbox"].get_category_name(),
                    "bbox_coords": bbox_data["pixel_coords"]
                })
        
        for j, diff_region in enumerate(diff_regions):
            if j not in matched_regions:
                comparison_results["unmatched_diff_regions"].append({
                    "diff_region_index": j,
                    "diff_region": diff_region
                })
        
        return comparison_results
    
    def generate_pair_analysis(self, drawing_id: str) -> Optional[ImageComparisonResult]:
        """
        Generate comprehensive analysis of an image pair.
        
        Args:
            drawing_id: ID of the drawing pair to analyze
            
        Returns:
            ImageComparisonResult object or None if analysis fails
        """
        drawing_data = self.get_pair_by_id(drawing_id)
        if not drawing_data:
            return None
            
        try:
            # Calculate similarity
            similarity_score = self.calculate_image_similarity(drawing_id) or 0.0
            
            # Find difference regions
            difference_regions = self.find_difference_regions(drawing_id)
            
            # Extract features from corrected image
            corrected_features = {
                "feedback_count": len(drawing_data.feedback_items),
                "bbox_count": len(drawing_data.bounding_boxes),
                "categories": [item.category_id for item in drawing_data.feedback_items],
                "corrections_needed": len([item for item in drawing_data.feedback_items if not item.is_correct()])
            }
            
            # Extract features from uncorrected image (placeholder - would need actual analysis)
            uncorrected_features = {
                "difference_regions": len(difference_regions),
                "estimated_errors": len(difference_regions)
            }
            
            # Generate recommendations
            recommendations = self._generate_recommendations(drawing_data, difference_regions)
            
            return ImageComparisonResult(
                drawing_id=drawing_id,
                similarity_score=similarity_score,
                difference_regions=difference_regions,
                corrected_features=corrected_features,
                uncorrected_features=uncorrected_features,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error generating analysis for {drawing_id}: {e}")
            return None
    
    def _generate_recommendations(self, drawing_data: TechnicalDrawingData, 
                                difference_regions: List[Dict]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Analyze feedback items
        corrections_needed = [item for item in drawing_data.feedback_items if not item.is_correct()]
        
        if corrections_needed:
            recommendations.append(f"Address {len(corrections_needed)} identified corrections")
            
            # Group by category
            category_counts = {}
            for item in corrections_needed:
                cat_name = item.get_category_name()
                category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
                
            for category, count in category_counts.items():
                recommendations.append(f"Review {category}: {count} items need correction")
        
        # Analyze difference regions
        if difference_regions:
            recommendations.append(f"Investigate {len(difference_regions)} visual difference regions")
            
            large_regions = [r for r in difference_regions if r["area"] > 1000]
            if large_regions:
                recommendations.append(f"Priority: {len(large_regions)} large difference regions detected")
        
        # Check for missing annotations
        if len(drawing_data.bounding_boxes) == 0:
            recommendations.append("Warning: No bounding box annotations found")
            
        if len(drawing_data.feedback_items) == 0:
            recommendations.append("Warning: No feedback items found")
        
        return recommendations
    
    def export_pair_analysis(self, drawing_id: str, output_path: str) -> bool:
        """
        Export comprehensive analysis of a pair to JSON file.
        
        Args:
            drawing_id: ID of the drawing pair
            output_path: Path to save the analysis
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            analysis = self.generate_pair_analysis(drawing_id)
            if not analysis:
                return False
                
            # Convert to serializable format
            export_data = {
                "drawing_id": analysis.drawing_id,
                "similarity_score": analysis.similarity_score,
                "difference_regions": analysis.difference_regions,
                "corrected_features": analysis.corrected_features,
                "uncorrected_features": analysis.uncorrected_features,
                "recommendations": analysis.recommendations,
                "bbox_comparison": self.compare_bounding_boxes(drawing_id),
                "validation": self.validate_pair(drawing_id)
            }
            
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Analysis exported to: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting analysis for {drawing_id}: {e}")
            return False
    
    def get_dataset_summary(self) -> Dict:
        """Get summary statistics for all image pairs."""
        if not self.drawings_data:
            self.load_all_pairs()
            
        summary = {
            "total_pairs": len(self.drawings_data),
            "valid_pairs": 0,
            "pairs_with_feedback": 0,
            "pairs_with_bboxes": 0,
            "average_similarity": 0.0,
            "total_difference_regions": 0,
            "pair_details": {}
        }
        
        similarities = []
        total_diff_regions = 0
        
        for drawing_id in self.drawings_data.keys():
            validation = self.validate_pair(drawing_id)
            analysis = self.generate_pair_analysis(drawing_id)
            
            if validation["corrected_image_loadable"] and validation["uncorrected_image_loadable"]:
                summary["valid_pairs"] += 1
                
            if validation["has_feedback"]:
                summary["pairs_with_feedback"] += 1
                
            if validation["has_bboxes"]:
                summary["pairs_with_bboxes"] += 1
                
            if analysis:
                similarities.append(analysis.similarity_score)
                total_diff_regions += len(analysis.difference_regions)
                
                summary["pair_details"][drawing_id] = {
                    "similarity": analysis.similarity_score,
                    "difference_regions": len(analysis.difference_regions),
                    "feedback_items": len(self.drawings_data[drawing_id].feedback_items),
                    "bounding_boxes": len(self.drawings_data[drawing_id].bounding_boxes)
                }
        
        if similarities:
            summary["average_similarity"] = np.mean(similarities)
            
        summary["total_difference_regions"] = total_diff_regions
        
        return summary


def main():
    """Test the image pair manager."""
    # Initialize manager
    dataset_root = Path(__file__).parent.parent.parent / "dataset"
    manager = ImagePairManager(str(dataset_root))
    
    # Load all pairs
    if not manager.load_all_pairs():
        print("Failed to load image pairs")
        return
    
    # Get dataset summary
    summary = manager.get_dataset_summary()
    print("\nDataset Summary:")
    print(f"Total pairs: {summary['total_pairs']}")
    print(f"Valid pairs: {summary['valid_pairs']}")
    print(f"Pairs with feedback: {summary['pairs_with_feedback']}")
    print(f"Pairs with bboxes: {summary['pairs_with_bboxes']}")
    print(f"Average similarity: {summary['average_similarity']:.3f}")
    print(f"Total difference regions: {summary['total_difference_regions']}")
    
    # Test analysis of first pair
    pair_ids = manager.get_all_pair_ids()
    if pair_ids:
        first_id = pair_ids[0]
        print(f"\nAnalyzing pair {first_id}:")
        
        validation = manager.validate_pair(first_id)
        print(f"  Validation: {validation}")
        
        analysis = manager.generate_pair_analysis(first_id)
        if analysis:
            print(f"  Similarity: {analysis.similarity_score:.3f}")
            print(f"  Difference regions: {len(analysis.difference_regions)}")
            print(f"  Recommendations: {len(analysis.recommendations)}")
            for rec in analysis.recommendations:
                print(f"    - {rec}")


if __name__ == "__main__":
    main()