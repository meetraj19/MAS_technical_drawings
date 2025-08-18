#!/usr/bin/env python3
"""
Pattern Recognition Agent for Technical Drawing Feedback System

This CrewAI agent identifies drawing elements and patterns by using
the pattern database created from your bounding box training data.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
import numpy as np

# Add tools to path
sys.path.append(str(Path(__file__).parent.parent))

from tools.dimension_extraction_tool import DimensionExtractionTool, ExtractionResult
from tools.pattern_matching_tool import PatternMatchingTool, PatternMatchingResult, MatchResult

logger = logging.getLogger(__name__)


class PatternRecognitionAgent:
    """
    CrewAI Agent for recognizing patterns in technical drawings.
    
    Role: Identify drawing elements and patterns against known correct patterns
    Goal: Match current drawing elements against your training database
    Tools: Dimension Extraction, Pattern Matching (uses your 71 bounding box patterns)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Pattern Recognition Agent.
        
        Args:
            config: Configuration dictionary for the agent
        """
        self.config = config or self._get_default_config()
        
        # Initialize tools
        self.dimension_tool = DimensionExtractionTool(self.config.get('dimension_extraction', {}))
        self.pattern_tool = PatternMatchingTool(self.config.get('pattern_matching', {}))
        
        # Load pattern database
        self._load_pattern_database()
        
        # Agent metadata
        self.role = "Technical Drawing Pattern Recognition Specialist"
        self.goal = "Identify and categorize drawing elements using learned patterns"
        self.backstory = """You are an expert in technical drawing pattern recognition, 
        specializing in German engineering drawings. You can identify elements like 
        Schraubenkopf, Scheibe, Platte, Gewindereserve, and other technical components 
        by matching them against a database of known correct patterns."""
        
        logger.info(f"Initialized {self.role}")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for the agent."""
        return {
            'pattern_matching': {
                'similarity_threshold': 0.7,
                'confidence_threshold': 0.6,
                'feature_extraction': {
                    'method': 'hog'
                }
            },
            'dimension_extraction': {
                'confidence_threshold': 0.7,
                'proximity_threshold': 50
            },
            'candidate_generation': {
                'use_ocr_regions': True,
                'use_visual_elements': True,
                'merge_overlapping_regions': True,
                'overlap_threshold': 0.5
            },
            'output': {
                'include_pattern_details': True,
                'include_dimension_data': True,
                'min_confidence_for_output': 0.5
            },
            'database_path': 'models/pattern_database.pkl'
        }
    
    def _load_pattern_database(self) -> bool:
        """Load the pattern database created from your bounding boxes."""
        try:
            db_path = Path(__file__).parent.parent.parent / self.config['database_path']
            
            if db_path.exists():
                success = self.pattern_tool.load_pattern_database(db_path)
                if success:
                    stats = self.pattern_tool.get_database_statistics()
                    logger.info(f"Loaded pattern database: {stats['total_patterns']} patterns "
                              f"across {stats['categories_covered']} categories")
                    return True
            
            # If database doesn't exist, create it from dataset
            logger.info("Pattern database not found, creating from training data...")
            dataset_root = Path(__file__).parent.parent.parent / "dataset"
            
            if dataset_root.exists():
                success = self.pattern_tool.load_training_data(str(dataset_root))
                if success:
                    # Save the database for future use
                    self.pattern_tool.save_pattern_database(db_path)
                    logger.info("Created and saved new pattern database")
                    return True
            
            logger.warning("Could not load or create pattern database")
            return False
            
        except Exception as e:
            logger.error(f"Error loading pattern database: {e}")
            return False
    
    def execute(self, document_structure: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Main execution method for the agent.
        
        Args:
            document_structure: Output from Document Parser Agent
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing pattern recognition results
        """
        logger.info("Pattern Recognition Agent executing...")
        
        try:
            # Step 1: Extract dimensions and technical specifications
            dimension_results = self._extract_dimensions(document_structure)
            
            # Step 2: Generate candidate regions for pattern matching
            candidate_regions = self._generate_candidate_regions(document_structure)
            
            # Step 3: Match patterns against database
            pattern_matches = self._match_patterns(candidate_regions, document_structure)
            
            # Step 4: Combine and structure results
            recognition_results = self._create_recognition_structure(
                document_structure, dimension_results, pattern_matches
            )
            
            logger.info("Pattern recognition completed successfully")
            return recognition_results
            
        except Exception as e:
            logger.error(f"Pattern Recognition Agent failed: {e}")
            return self._create_error_result(str(e), document_structure)
    
    def _extract_dimensions(self, document_structure: Dict[str, Any]) -> List[ExtractionResult]:
        """
        Extract dimensions and technical specifications from text elements.
        
        Args:
            document_structure: Document structure from parser agent
            
        Returns:
            List of extraction results
        """
        logger.info("Step 1: Extracting dimensions and specifications...")
        
        extraction_results = []
        
        try:
            # Create mock OCR result from document structure
            text_elements = document_structure.get('extracted_content', {}).get('text_elements', [])
            
            if text_elements:
                # Convert to OCR-like format for dimension extraction
                mock_ocr_result = self._create_mock_ocr_result(text_elements, document_structure)
                
                # Extract dimensions
                extraction_result = self.dimension_tool.extract_from_ocr_result(mock_ocr_result)
                extraction_results.append(extraction_result)
                
                logger.info(f"Extracted {len(extraction_result.dimensions)} dimensions, "
                          f"{len(extraction_result.tolerances)} tolerances, "
                          f"{len(extraction_result.surface_finishes)} surface finishes")
            else:
                logger.info("No text elements found for dimension extraction")
            
        except Exception as e:
            logger.error(f"Dimension extraction failed: {e}")
        
        return extraction_results
    
    def _generate_candidate_regions(self, document_structure: Dict[str, Any]) -> List[Tuple[int, int, int, int]]:
        """
        Generate candidate regions for pattern matching.
        
        Args:
            document_structure: Document structure from parser agent
            
        Returns:
            List of candidate regions (x, y, width, height)
        """
        logger.info("Step 2: Generating candidate regions...")
        
        candidate_regions = []
        config = self.config['candidate_generation']
        
        try:
            # Extract regions from text elements
            if config['use_ocr_regions']:
                text_elements = document_structure.get('extracted_content', {}).get('text_elements', [])
                for element in text_elements:
                    bbox = element.get('bounding_box', [0, 0, 0, 0])
                    if len(bbox) == 4:
                        x, y, w, h = bbox
                        # Expand region slightly for context
                        expansion = 20
                        candidate_regions.append((
                            max(0, x - expansion),
                            max(0, y - expansion),
                            w + 2 * expansion,
                            h + 2 * expansion
                        ))
            
            # Extract regions from visual elements
            if config['use_visual_elements']:
                visual_elements = document_structure.get('extracted_content', {}).get('visual_elements', [])
                for element in visual_elements:
                    if element.get('type') == 'rectangle':
                        corners = element.get('corners', [])
                        if len(corners) >= 4:
                            # Calculate bounding box from corners
                            x_coords = [c[0] for c in corners]
                            y_coords = [c[1] for c in corners]
                            x, y = min(x_coords), min(y_coords)
                            w = max(x_coords) - x
                            h = max(y_coords) - y
                            candidate_regions.append((x, y, w, h))
                    
                    elif element.get('type') == 'circle':
                        center = element.get('center', [0, 0])
                        radius = element.get('radius', 0)
                        if radius > 0:
                            x = center[0] - radius
                            y = center[1] - radius
                            w = h = 2 * radius
                            candidate_regions.append((int(x), int(y), int(w), int(h)))
            
            # Merge overlapping regions if requested
            if config['merge_overlapping_regions']:
                candidate_regions = self._merge_overlapping_regions(
                    candidate_regions, config['overlap_threshold']
                )
            
            logger.info(f"Generated {len(candidate_regions)} candidate regions")
            
        except Exception as e:
            logger.error(f"Candidate region generation failed: {e}")
        
        return candidate_regions
    
    def _match_patterns(self, candidate_regions: List[Tuple[int, int, int, int]],
                       document_structure: Dict[str, Any]) -> PatternMatchingResult:
        """
        Match candidate regions against pattern database.
        
        Args:
            candidate_regions: List of candidate regions
            document_structure: Document structure with image info
            
        Returns:
            Pattern matching results
        """
        logger.info("Step 3: Matching patterns against database...")
        
        try:
            image_id = document_structure.get('input_info', {}).get('file_name', 'unknown')
            
            # Use the real pattern matching tool with feedback-based approach
            # Create a dummy image since we're using feedback-based matching that doesn't need actual image features
            import numpy as np
            dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
            
            # Call the real pattern matching tool
            pattern_matches = self.pattern_tool.match_patterns(dummy_image, candidate_regions, image_id)
            
            logger.info(f"Pattern matching completed: {len(pattern_matches.matches)} matches found")
            return pattern_matches
            
        except Exception as e:
            logger.error(f"Pattern matching failed: {e}")
            return PatternMatchingResult(
                image_id=document_structure.get('input_info', {}).get('file_name', 'unknown'),
                matches=[],
                unmatched_regions=candidate_regions,
                matching_metadata={'error': str(e)}
            )
    
    def _create_recognition_structure(self, document_structure: Dict[str, Any],
                                    dimension_results: List[ExtractionResult],
                                    pattern_matches: PatternMatchingResult) -> Dict[str, Any]:
        """
        Create structured pattern recognition results.
        
        Args:
            document_structure: Original document structure
            dimension_results: Dimension extraction results
            pattern_matches: Pattern matching results
            
        Returns:
            Structured recognition data
        """
        logger.info("Step 4: Creating recognition structure...")
        
        # Categorize matches by category
        matches_by_category = {}
        for match in pattern_matches.matches:
            # Handle both old-style pattern objects and new feedback-based MatchResult
            if hasattr(match, 'pattern') and hasattr(match.pattern, 'category_id'):
                category_id = match.pattern.category_id
            elif hasattr(match, 'pattern') and isinstance(match.pattern, dict):
                # Pattern object is a dictionary (from feedback database)
                category_id = match.pattern.get('class_id', 0)
            elif hasattr(match, 'class_id'):
                category_id = match.class_id
            else:
                category_id = 0  # Default category
                
            if category_id not in matches_by_category:
                matches_by_category[category_id] = []
            matches_by_category[category_id].append(match)
        
        # Calculate confidence statistics
        confidences = [match.confidence for match in pattern_matches.matches]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Combine dimension data
        all_dimensions = []
        all_tolerances = []
        all_surface_finishes = []
        all_threads = []
        
        for result in dimension_results:
            all_dimensions.extend(result.dimensions)
            all_tolerances.extend(result.tolerances)
            all_surface_finishes.extend(result.surface_finishes)
            all_threads.extend(result.thread_specifications)
        
        # Create recognition structure
        recognition_structure = {
            'agent_info': {
                'agent_name': 'Pattern Recognition Agent',
                'role': self.role,
                'processing_timestamp': self._get_timestamp(),
                'version': '1.0'
            },
            'input_info': document_structure.get('input_info', {}),
            'recognition_summary': {
                'total_patterns_matched': len(pattern_matches.matches),
                'categories_detected': len(matches_by_category),
                'unmatched_regions': len(pattern_matches.unmatched_regions),
                'average_confidence': avg_confidence,
                'dimensions_extracted': len(all_dimensions),
                'processing_successful': True
            },
            'pattern_matches': {
                'by_category': self._serialize_matches_by_category(matches_by_category),
                'all_matches': self._serialize_all_matches(pattern_matches.matches),
                'matching_metadata': pattern_matches.matching_metadata
            },
            'extracted_specifications': {
                'dimensions': self._serialize_dimensions(all_dimensions),
                'tolerances': self._serialize_tolerances(all_tolerances),
                'surface_finishes': self._serialize_surface_finishes(all_surface_finishes),
                'thread_specifications': self._serialize_threads(all_threads)
            },
            'category_analysis': self._create_category_analysis(matches_by_category)
        }
        
        return recognition_structure
    
    def _create_mock_ocr_result(self, text_elements: List[Dict], document_structure: Dict) -> Any:
        """Create a mock OCR result for dimension extraction."""
        class MockTextElement:
            def __init__(self, text_data):
                self.text = text_data.get('text', '')
                self.confidence = text_data.get('confidence', 0)
                self.center = tuple(text_data.get('center', [0, 0]))
                self.text_type = text_data.get('text_type', 'unknown')
        
        class MockOCRResult:
            def __init__(self, image_id, text_elements):
                self.image_id = image_id
                self.text_elements = [MockTextElement(elem) for elem in text_elements]
                self.full_text = ' '.join([elem['text'] for elem in text_elements])
                self.confidence_statistics = {'mean_confidence': 75.0}
        
        image_id = document_structure.get('input_info', {}).get('file_name', 'unknown')
        return MockOCRResult(image_id, text_elements)
    
    def _create_mock_pattern_matches(self, candidate_regions: List[Tuple[int, int, int, int]],
                                   image_id: str) -> PatternMatchingResult:
        """Create mock pattern matches for testing."""
        # This is a simplified mock - in reality would use actual pattern matching
        matches = []
        unmatched_regions = candidate_regions.copy()
        
        # Simulate some matches based on database statistics
        if hasattr(self.pattern_tool, 'pattern_database') and self.pattern_tool.pattern_database:
            # Take a few patterns as examples
            sample_patterns = self.pattern_tool.pattern_database[:min(3, len(candidate_regions))]
            
            for i, pattern in enumerate(sample_patterns):
                if i < len(candidate_regions):
                    region = candidate_regions[i]
                    
                    # Create mock match result
                    class MockMatchResult:
                        def __init__(self, pattern, region):
                            self.pattern = pattern
                            self.similarity_score = 0.75
                            self.matched_region = region
                            self.confidence = 0.8
                    
                    matches.append(MockMatchResult(pattern, region))
                    unmatched_regions.remove(region)
        
        return PatternMatchingResult(
            image_id=image_id,
            matches=matches,
            unmatched_regions=unmatched_regions,
            matching_metadata={
                'total_regions_processed': len(candidate_regions),
                'matches_found': len(matches),
                'pattern_database_size': len(self.pattern_tool.pattern_database) if hasattr(self.pattern_tool, 'pattern_database') else 0
            }
        )
    
    def _merge_overlapping_regions(self, regions: List[Tuple[int, int, int, int]], 
                                 threshold: float) -> List[Tuple[int, int, int, int]]:
        """Merge overlapping candidate regions."""
        if not regions:
            return []
        
        merged = []
        remaining = regions.copy()
        
        while remaining:
            current = remaining.pop(0)
            to_merge = [current]
            
            # Find overlapping regions
            i = 0
            while i < len(remaining):
                if self._calculate_overlap(current, remaining[i]) > threshold:
                    to_merge.append(remaining.pop(i))
                else:
                    i += 1
            
            # Merge all overlapping regions
            merged_region = self._merge_regions(to_merge)
            merged.append(merged_region)
        
        return merged
    
    def _calculate_overlap(self, region1: Tuple[int, int, int, int], 
                         region2: Tuple[int, int, int, int]) -> float:
        """Calculate overlap ratio between two regions."""
        x1, y1, w1, h1 = region1
        x2, y2, w2, h2 = region2
        
        # Calculate intersection
        left = max(x1, x2)
        top = max(y1, y2)
        right = min(x1 + w1, x2 + w2)
        bottom = min(y1 + h1, y2 + h2)
        
        if left < right and top < bottom:
            intersection = (right - left) * (bottom - top)
            union = w1 * h1 + w2 * h2 - intersection
            return intersection / union if union > 0 else 0
        
        return 0
    
    def _merge_regions(self, regions: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
        """Merge multiple regions into one bounding box."""
        if not regions:
            return (0, 0, 0, 0)
        
        min_x = min(r[0] for r in regions)
        min_y = min(r[1] for r in regions)
        max_x = max(r[0] + r[2] for r in regions)
        max_y = max(r[1] + r[3] for r in regions)
        
        return (min_x, min_y, max_x - min_x, max_y - min_y)
    
    def _serialize_matches_by_category(self, matches_by_category: Dict) -> Dict:
        """Serialize matches grouped by category."""
        serialized = {}
        
        for category_id, matches in matches_by_category.items():
            # Handle both old and new match structures
            first_match = matches[0] if matches else None
            if first_match and hasattr(first_match, 'pattern') and hasattr(first_match.pattern, 'category_name'):
                category_name = first_match.pattern.category_name
            elif first_match and hasattr(first_match, 'pattern') and isinstance(first_match.pattern, dict):
                # Pattern object is a dictionary (from feedback database)
                category_name = first_match.pattern.get('category_name', f"Category {category_id}")
            elif first_match and hasattr(first_match, 'matched_features') and 'category_name' in first_match.matched_features:
                category_name = first_match.matched_features['category_name']
            else:
                category_name = f"Category {category_id}"
            
            serialized[str(category_id)] = {
                'category_name': category_name,
                'match_count': len(matches),
                'average_confidence': sum(m.confidence for m in matches) / len(matches),
                'matches': [
                    {
                        'pattern_id': getattr(match, 'pattern_id', match.pattern.pattern_id if hasattr(match, 'pattern') else 'unknown'),
                        'similarity_score': getattr(match, 'similarity_score', 0),
                        'confidence': match.confidence,
                        'matched_region': getattr(match, 'bounding_box', getattr(match, 'matched_region', [])),
                        'feedback_text': match.matched_features.get('feedback_text', '') if hasattr(match, 'matched_features') else '',
                        'severity': match.matched_features.get('severity', 'minor') if hasattr(match, 'matched_features') else 'minor'
                    }
                    for match in matches
                ]
            }
        
        return serialized
    
    def _serialize_all_matches(self, matches: List) -> List[Dict]:
        """Serialize all matches."""
        serialized_matches = []
        
        for match in matches:
            # Handle both old-style pattern objects and new feedback-based MatchResult
            if hasattr(match, 'pattern') and hasattr(match.pattern, 'pattern_id'):
                # Old style with Pattern object
                pattern_id = match.pattern.pattern_id
                category_id = match.pattern.category_id
                category_name = match.pattern.category_name
            elif hasattr(match, 'pattern') and isinstance(match.pattern, dict):
                # Pattern object is a dictionary (from feedback database)
                pattern_id = match.pattern.get('id', 'unknown')
                category_id = match.pattern.get('class_id', 0)
                category_name = match.pattern.get('category_name', 'Allgemein')
            elif hasattr(match, 'pattern_id'):
                # New style with direct attributes
                pattern_id = match.pattern_id
                category_id = getattr(match, 'class_id', 0)
                category_name = match.matched_features.get('category_name', 'Allgemein') if hasattr(match, 'matched_features') else 'Allgemein'
            else:
                # Fallback values
                pattern_id = 'unknown'
                category_id = 0
                category_name = 'Allgemein'
            
            serialized_matches.append({
                'pattern_id': pattern_id,
                'category_id': category_id,
                'category_name': category_name,
                'similarity_score': getattr(match, 'similarity_score', 0),
                'confidence': match.confidence,
                'matched_region': getattr(match, 'bounding_box', getattr(match, 'matched_region', []))
            })
        
        return serialized_matches
    
    def _serialize_dimensions(self, dimensions: List) -> List[Dict]:
        """Serialize dimension data."""
        return [
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
            for dim in dimensions
        ]
    
    def _serialize_tolerances(self, tolerances: List) -> List[Dict]:
        """Serialize tolerance data."""
        return [
            {
                'upper_limit': tol.upper_limit,
                'lower_limit': tol.lower_limit,
                'tolerance_class': tol.tolerance_class,
                'tolerance_type': tol.tolerance_type
            }
            for tol in tolerances
        ]
    
    def _serialize_surface_finishes(self, surface_finishes: List) -> List[Dict]:
        """Serialize surface finish data."""
        return [
            {
                'roughness_value': sf.roughness_value,
                'roughness_type': sf.roughness_type,
                'position': sf.position,
                'text': sf.text
            }
            for sf in surface_finishes
        ]
    
    def _serialize_threads(self, threads: List) -> List[Dict]:
        """Serialize thread specification data."""
        return [
            {
                'thread_type': thread.thread_type,
                'nominal_diameter': thread.nominal_diameter,
                'pitch': thread.pitch,
                'thread_class': thread.thread_class,
                'position': thread.position,
                'text': thread.text
            }
            for thread in threads
        ]
    
    def _create_category_analysis(self, matches_by_category: Dict) -> Dict:
        """Create analysis of detected categories."""
        analysis = {
            'categories_detected': list(matches_by_category.keys()),
            'category_confidence': {},
            'recommendations': []
        }
        
        for category_id, matches in matches_by_category.items():
            if not matches:
                continue
                
            avg_conf = sum(m.confidence for m in matches) / len(matches)
            analysis['category_confidence'][str(category_id)] = avg_conf
            
            if avg_conf < 0.7:
                # Handle both old and new match structures for category name
                first_match = matches[0]
                if hasattr(first_match, 'pattern') and hasattr(first_match.pattern, 'category_name'):
                    category_name = first_match.pattern.category_name
                elif hasattr(first_match, 'pattern') and isinstance(first_match.pattern, dict):
                    # Pattern object is a dictionary (from feedback database)
                    category_name = first_match.pattern.get('category_name', f"Category {category_id}")
                elif hasattr(first_match, 'matched_features') and 'category_name' in first_match.matched_features:
                    category_name = first_match.matched_features['category_name']
                else:
                    category_name = f"Category {category_id}"
                    
                analysis['recommendations'].append(
                    f"Review {category_name}: niedrige Vertrauenswürdigkeit ({avg_conf:.1%})"
                )
        
        return analysis
    
    def _create_error_result(self, error_message: str, document_structure: Dict) -> Dict[str, Any]:
        """Create error result structure."""
        return {
            'agent_info': {
                'agent_name': 'Pattern Recognition Agent',
                'role': self.role,
                'processing_timestamp': self._get_timestamp(),
                'version': '1.0'
            },
            'input_info': document_structure.get('input_info', {}),
            'recognition_summary': {
                'processing_successful': False,
                'error_message': error_message
            },
            'pattern_matches': {'all_matches': []},
            'extracted_specifications': {}
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def save_results(self, results: Dict[str, Any], output_path: str) -> bool:
        """Save agent results to file."""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Pattern Recognition Agent results saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return False


def main():
    """Test the Pattern Recognition Agent."""
    # Initialize agent
    agent = PatternRecognitionAgent()
    
    # Test with document parser results
    output_dir = Path(__file__).parent.parent.parent / "output" / "agent_results"
    parser_results_file = output_dir / "8_document_parser.json"
    
    if parser_results_file.exists():
        print(f"\nTesting Pattern Recognition Agent...")
        
        # Load document parser results
        with open(parser_results_file, 'r', encoding='utf-8') as f:
            document_structure = json.load(f)
        
        print(f"Processing document: {document_structure['input_info']['file_name']}")
        
        # Execute agent
        results = agent.execute(document_structure)
        
        if results['recognition_summary']['processing_successful']:
            summary = results['recognition_summary']
            print(f"  ✅ Pattern recognition successful")
            print(f"  Patterns matched: {summary['total_patterns_matched']}")
            print(f"  Categories detected: {summary['categories_detected']}")
            print(f"  Average confidence: {summary['average_confidence']:.1%}")
            print(f"  Dimensions extracted: {summary['dimensions_extracted']}")
            
            # Save results
            output_file = output_dir / "8_pattern_recognition.json"
            agent.save_results(results, str(output_file))
            print(f"  Results saved to: {output_file}")
        else:
            print(f"  ❌ Processing failed: {results['recognition_summary'].get('error_message')}")
    else:
        print("Document parser results not found. Please run document parser agent first.")


if __name__ == "__main__":
    main()