#!/usr/bin/env python3
"""
Data Integration Utility for Technical Drawing Feedback System

This module handles the integration of existing data folder structure
into the proper dataset organization required by the system.
"""

import os
import shutil
import json
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIntegrator:
    """Handles integration of existing data into proper structure."""
    
    def __init__(self, project_root: str):
        """
        Initialize the DataIntegrator.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "data"
        self.dataset_dir = self.project_root / "dataset"
        
        # Original data directories
        self.corrected_data_dir = self.data_dir / "corrected_data"
        self.uncorrected_data_dir = self.data_dir / "uncorrected_data"
        self.feedback_dir = self.data_dir / "feedback_for _correcteddata"
        self.bbox_dir = self.data_dir / "boundingBox_for_correcteddata"
        
        # Target directories
        self.target_corrected_dir = self.dataset_dir / "corrected"
        self.target_uncorrected_dir = self.dataset_dir / "uncorrected"
        
    def validate_source_data(self) -> bool:
        """
        Validate that all required source data directories exist.
        
        Returns:
            bool: True if all directories exist, False otherwise
        """
        required_dirs = [
            self.corrected_data_dir,
            self.uncorrected_data_dir,
            self.feedback_dir,
            self.bbox_dir
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                logger.error(f"Required directory not found: {dir_path}")
                return False
            logger.info(f"Found directory: {dir_path}")
            
        return True
    
    def get_image_pairs(self) -> List[Tuple[str, str]]:
        """
        Get list of corrected/uncorrected image pairs.
        
        Returns:
            List of tuples (corrected_image, uncorrected_image)
        """
        pairs = []
        
        corrected_files = list(self.corrected_data_dir.glob("*.jpg"))
        
        for corrected_file in sorted(corrected_files):
            image_id = corrected_file.stem  # e.g., "1", "2", etc.
            uncorrected_file = self.uncorrected_data_dir / f"{image_id}.jpg"
            
            if uncorrected_file.exists():
                pairs.append((str(corrected_file), str(uncorrected_file)))
                logger.info(f"Found pair: {image_id}")
            else:
                logger.warning(f"No uncorrected pair found for: {image_id}")
                
        return pairs
    
    def parse_feedback_file(self, feedback_file: Path) -> Dict:
        """
        Parse a German feedback file into structured format.
        
        Args:
            feedback_file: Path to feedback file
            
        Returns:
            Dictionary with parsed feedback data
        """
        feedback_data = {
            "file_id": feedback_file.stem.replace("_name", ""),
            "categories": {},
            "raw_content": ""
        }
        
        try:
            with open(feedback_file, 'r', encoding='utf-8') as f:
                content = f.read()
                feedback_data["raw_content"] = content
                
                # Parse each line for category information
                lines = content.split('\n')
                for line in lines:
                    if line.strip():
                        # Extract category number at start of line
                        parts = line.split(' ', 1)
                        if parts[0].isdigit():
                            category_id = int(parts[0])
                            category_text = parts[1] if len(parts) > 1 else ""
                            
                            if category_id not in feedback_data["categories"]:
                                feedback_data["categories"][category_id] = []
                            
                            feedback_data["categories"][category_id].append({
                                "text": category_text,
                                "raw_line": line
                            })
                            
        except Exception as e:
            logger.error(f"Error parsing feedback file {feedback_file}: {e}")
            
        return feedback_data
    
    def parse_bbox_file(self, bbox_file: Path) -> List[Dict]:
        """
        Parse bounding box file (YOLO format).
        
        Args:
            bbox_file: Path to bounding box file
            
        Returns:
            List of bounding box dictionaries
        """
        bboxes = []
        
        try:
            with open(bbox_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) == 5:
                            class_id, center_x, center_y, width, height = parts
                            bboxes.append({
                                "class_id": int(class_id),
                                "center_x": float(center_x),
                                "center_y": float(center_y),
                                "width": float(width),
                                "height": float(height),
                                "line_number": line_num
                            })
                        else:
                            logger.warning(f"Invalid bbox format in {bbox_file}:{line_num}")
                            
        except Exception as e:
            logger.error(f"Error parsing bbox file {bbox_file}: {e}")
            
        return bboxes
    
    def create_dataset_metadata(self) -> Dict:
        """
        Create comprehensive metadata for the dataset.
        
        Returns:
            Dictionary containing dataset metadata
        """
        metadata = {
            "dataset_info": {
                "name": "Technical Drawing Feedback Dataset",
                "description": "German technical drawing analysis with corrected/uncorrected pairs",
                "version": "1.0",
                "total_images": 0,
                "categories": {
                    1: "Unknown",
                    2: "Schraubenkopf (Screw Head)",
                    3: "Scheibe (Washer)",
                    4: "Platte (Plate)",
                    5: "Gewindereserve (Thread Reserve)",
                    6: "Grundloch (Pilot Hole)",
                    7: "Gewindedarstellung (Thread Representation)",
                    8: "Schraffur (Hatching)",
                    9: "Schriftfeld (Title Block)"
                }
            },
            "image_pairs": [],
            "feedback_summary": {},
            "bbox_summary": {}
        }
        
        # Process each image pair
        image_pairs = self.get_image_pairs()
        metadata["dataset_info"]["total_images"] = len(image_pairs)
        
        for corrected_path, uncorrected_path in image_pairs:
            corrected_file = Path(corrected_path)
            image_id = corrected_file.stem
            
            # Get feedback data
            feedback_file = self.feedback_dir / f"{image_id}_name.txt"
            feedback_data = {}
            if feedback_file.exists():
                feedback_data = self.parse_feedback_file(feedback_file)
            
            # Get bounding box data
            bbox_file = self.bbox_dir / f"{image_id}.txt"
            bbox_data = []
            if bbox_file.exists():
                bbox_data = self.parse_bbox_file(bbox_file)
            
            pair_metadata = {
                "id": image_id,
                "corrected_image": corrected_path,
                "uncorrected_image": uncorrected_path,
                "feedback_file": str(feedback_file) if feedback_file.exists() else None,
                "bbox_file": str(bbox_file) if bbox_file.exists() else None,
                "feedback_categories": list(feedback_data.get("categories", {}).keys()),
                "bbox_count": len(bbox_data),
                "bbox_categories": list(set(bbox["class_id"] for bbox in bbox_data))
            }
            
            metadata["image_pairs"].append(pair_metadata)
            
        return metadata
    
    def integrate_data(self, copy_images: bool = True) -> bool:
        """
        Integrate existing data into proper dataset structure.
        
        Args:
            copy_images: Whether to copy images to dataset directory
            
        Returns:
            bool: True if integration successful
        """
        try:
            # Validate source data
            if not self.validate_source_data():
                return False
            
            # Create target directories
            self.target_corrected_dir.mkdir(parents=True, exist_ok=True)
            self.target_uncorrected_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate metadata
            logger.info("Generating dataset metadata...")
            metadata = self.create_dataset_metadata()
            
            # Save metadata
            metadata_file = self.dataset_dir / "metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Metadata saved to: {metadata_file}")
            
            # Copy images if requested
            if copy_images:
                logger.info("Copying images to dataset directory...")
                image_pairs = self.get_image_pairs()
                
                for corrected_path, uncorrected_path in image_pairs:
                    corrected_file = Path(corrected_path)
                    uncorrected_file = Path(uncorrected_path)
                    
                    # Copy corrected image
                    target_corrected = self.target_corrected_dir / corrected_file.name
                    shutil.copy2(corrected_file, target_corrected)
                    
                    # Copy uncorrected image
                    target_uncorrected = self.target_uncorrected_dir / uncorrected_file.name
                    shutil.copy2(uncorrected_file, target_uncorrected)
                    
                    logger.info(f"Copied pair: {corrected_file.name}")
            
            logger.info("Data integration completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Data integration failed: {e}")
            return False


def main():
    """Main function for data integration."""
    # Get project root directory
    project_root = Path(__file__).parent.parent.parent
    
    integrator = DataIntegrator(str(project_root))
    
    logger.info("Starting data integration...")
    success = integrator.integrate_data(copy_images=True)
    
    if success:
        logger.info("Data integration completed successfully!")
    else:
        logger.error("Data integration failed!")


if __name__ == "__main__":
    main()