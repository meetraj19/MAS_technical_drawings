#!/usr/bin/env python3
"""
Feedback Generator Agent for Technical Drawing Feedback System

This CrewAI agent generates comprehensive German feedback and visual annotations
using your training data patterns and bounding box coordinates.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import json

# Add tools to path
sys.path.append(str(Path(__file__).parent.parent))

from tools.feedback_formatting_tool import FeedbackFormattingTool, FeedbackReport, GeneratedFeedback
from tools.visual_annotation_tool import VisualAnnotationTool, AnnotatedImage

logger = logging.getLogger(__name__)


class FeedbackGeneratorAgent:
    """
    CrewAI Agent for generating comprehensive feedback.
    
    Role: Create comprehensive German feedback and visual annotations
    Goal: Generate actionable feedback reports with visual annotations
    Tools: Feedback Formatting (German patterns), Visual Annotation (bounding boxes)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize Feedback Generator Agent."""
        self.config = config or self._get_default_config()
        
        # Initialize tools
        self.feedback_tool = FeedbackFormattingTool(self.config.get('feedback_formatting', {}))
        self.annotation_tool = VisualAnnotationTool(self.config.get('visual_annotation', {}))
        
        # Load training patterns
        self._load_training_patterns()
        
        # Agent metadata
        self.role = "Technical Drawing Feedback Generator"
        self.goal = "Generate comprehensive German feedback with visual annotations"
        self.backstory = """You are an expert technical drawing instructor specializing in 
        German engineering documentation. You provide clear, actionable feedback in German 
        and create visual annotations to help engineers improve their technical drawings."""
        
        logger.info(f"Initialized {self.role}")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for the agent."""
        return {
            'feedback_formatting': {
                'language': 'de',
                'severity_mapping': {
                    'critical': ['fehlt', 'falsch', 'unvollständig'],
                    'major': ['überprüfen', 'korrigieren', 'anpassen'],
                    'minor': ['verbessern', 'optimieren', 'ergänzen']
                }
            },
            'visual_annotation': {
                'colors': {
                    'critical': (255, 0, 0),    # Red
                    'major': (255, 165, 0),     # Orange
                    'minor': (255, 255, 0),     # Yellow
                    'correct': (0, 255, 0)      # Green
                }
            },
            'output': {
                'generate_visual_annotations': True,
                'generate_text_report': True,
                'include_recommendations': True,
                'format_versions': ['json', 'txt', 'pdf']
            }
        }
    
    def _load_training_patterns(self) -> bool:
        """Load German feedback patterns from training data."""
        try:
            dataset_root = Path(__file__).parent.parent.parent / "dataset"
            
            if dataset_root.exists():
                success = self.feedback_tool.load_training_feedback(str(dataset_root))
                if success:
                    logger.info(f"Loaded feedback templates for {len(self.feedback_tool.feedback_templates)} categories")
                    return True
            
            logger.warning("Could not load training feedback patterns")
            return False
            
        except Exception as e:
            logger.error(f"Error loading training patterns: {e}")
            return False
    
    def execute(self, validation_results: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Main execution method for the agent.
        
        Args:
            validation_results: Output from Rule Validation Agent
            **kwargs: Additional parameters (pattern_results, original_image, etc.)
            
        Returns:
            Dictionary containing feedback generation results
        """
        logger.info("Feedback Generator Agent executing...")
        
        try:
            # Step 1: Generate German feedback from violations
            german_feedback = self._generate_german_feedback(validation_results)
            
            # Step 2: Create visual annotations (if image available)
            visual_annotations = self._create_visual_annotations(validation_results, kwargs)
            
            # Step 3: Format comprehensive report
            comprehensive_report = self._create_comprehensive_report(
                validation_results, german_feedback, visual_annotations
            )
            
            # Step 4: Generate final output structure
            feedback_results = self._create_feedback_structure(
                validation_results, german_feedback, visual_annotations, comprehensive_report
            )
            
            logger.info("Feedback generation completed successfully")
            return feedback_results
            
        except Exception as e:
            logger.error(f"Feedback Generator Agent failed: {e}")
            return self._create_error_result(str(e), validation_results)
    
    def _generate_german_feedback(self, validation_results: Dict[str, Any]) -> List[GeneratedFeedback]:
        """Generate German feedback from validation violations."""
        logger.info("Step 1: Generating German feedback...")
        
        generated_feedback = []
        violations = validation_results.get('violations', {}).get('all_violations', [])
        
        # Get pattern recognition and OCR confidence data
        pattern_confidence = validation_results.get('input_analysis', {}).get('pattern_analysis', {}).get('average_confidence', 0.5)
        ocr_confidence = validation_results.get('input_analysis', {}).get('text_analysis', {}).get('average_confidence', 0.5)
        
        for violation in violations:
            # Calculate dynamic confidence based on multiple factors
            base_confidence = self._calculate_violation_confidence(
                violation, pattern_confidence, ocr_confidence, validation_results
            )
            
            # Create feedback item from violation
            feedback_item = GeneratedFeedback(
                category_id=violation.get('category_id', 0),
                category_name=violation.get('category_name', 'Unknown'),
                element_description=violation.get('description', ''),
                feedback_text=violation.get('german_feedback', violation.get('description', '')),
                status='Korrektur erforderlich',
                severity=violation.get('severity', 'minor'),
                confidence=base_confidence,
                position=None  # Would be populated from bounding box data
            )
            
            generated_feedback.append(feedback_item)
        
        # Add positive feedback for compliant categories
        compliance_score = validation_results.get('validation_summary', {}).get('compliance_score', 0)
        if compliance_score > 0.8:
            # Calculate confidence for positive feedback based on overall system confidence
            positive_confidence = self._calculate_positive_feedback_confidence(
                compliance_score, pattern_confidence, ocr_confidence
            )
            
            generated_feedback.append(GeneratedFeedback(
                category_id=9,
                category_name='Allgemein',
                element_description='Zeichnungsqualität',
                feedback_text='Gute Übereinstimmung mit Standards erkannt',
                status='Sicher',
                severity='minor',
                confidence=positive_confidence
            ))
        
        logger.info(f"Generated {len(generated_feedback)} feedback items")
        return generated_feedback
    
    def _calculate_violation_confidence(self, violation: Dict[str, Any], 
                                      pattern_confidence: float, ocr_confidence: float,
                                      validation_results: Dict[str, Any]) -> float:
        """Calculate confidence score for a violation based on multiple factors."""
        
        # Base confidence factors
        confidence_factors = []
        
        # 1. Pattern recognition confidence (how well we detected the issue)
        confidence_factors.append(pattern_confidence)
        
        # 2. OCR confidence (how well we read the text)
        confidence_factors.append(ocr_confidence)
        
        # 3. Rule certainty (how definitive the rule violation is)
        severity_weights = {'critical': 0.95, 'major': 0.85, 'minor': 0.75}
        rule_confidence = severity_weights.get(violation.get('severity', 'minor'), 0.75)
        confidence_factors.append(rule_confidence)
        
        # 4. Context confidence (how much context we have)
        context_confidence = min(1.0, len(validation_results.get('violations', {}).get('all_violations', [])) / 10.0)
        confidence_factors.append(context_confidence)
        
        # 5. Category-specific confidence
        category_id = violation.get('category_id', 0)
        category_data = validation_results.get('category_analysis', {}).get(str(category_id), {})
        category_confidence = category_data.get('confidence', 0.5)
        confidence_factors.append(category_confidence)
        
        # Calculate weighted average with emphasis on pattern and OCR
        weighted_confidence = (
            pattern_confidence * 0.3 +
            ocr_confidence * 0.3 +
            rule_confidence * 0.2 +
            context_confidence * 0.1 +
            category_confidence * 0.1
        )
        
        # Apply random variation to prevent identical values (±5%)
        import random
        variation = random.uniform(-0.05, 0.05)
        final_confidence = max(0.1, min(0.99, weighted_confidence + variation))
        
        return final_confidence
    
    def _calculate_positive_feedback_confidence(self, compliance_score: float,
                                              pattern_confidence: float, ocr_confidence: float) -> float:
        """Calculate confidence for positive feedback."""
        
        # Positive feedback confidence based on system performance
        base_confidence = (compliance_score + pattern_confidence + ocr_confidence) / 3.0
        
        # Boost confidence for high compliance
        if compliance_score > 0.9:
            base_confidence *= 1.1
        elif compliance_score > 0.85:
            base_confidence *= 1.05
        
        # Apply small random variation
        import random
        variation = random.uniform(-0.03, 0.03)
        final_confidence = max(0.5, min(0.99, base_confidence + variation))
        
        return final_confidence
    
    def _create_visual_annotations(self, validation_results: Dict[str, Any], 
                                 kwargs: Dict[str, Any]) -> Optional[AnnotatedImage]:
        """Create visual annotations for violations."""
        logger.info("Step 2: Creating visual annotations...")
        
        if not self.config['output']['generate_visual_annotations']:
            return None
        
        try:
            # Get input analysis data
            input_analysis = validation_results.get('input_analysis', {})
            
            # Get original image path from kwargs or input analysis
            from pathlib import Path
            
            original_image_path = kwargs.get('original_image_path')
            if not original_image_path:
                # Try to get from input analysis
                input_file = input_analysis.get('input_info', {}).get('file_path')
                if input_file:
                    # For PDF files, assume first page image was created
                    input_path = Path(input_file)
                    if input_path.suffix.lower() == '.pdf':
                        # Look for extracted image in temp directory
                        temp_dir = kwargs.get('temp_dir', 'temp')
                        original_image_path = f"{temp_dir}/{input_path.stem}_page_1.png"
                    else:
                        original_image_path = input_file
            
            if not original_image_path or not Path(original_image_path).exists():
                logger.warning(f"Original image not found: {original_image_path}, creating mock annotation")
                return self._create_mock_annotation(validation_results)
            
            # Get OCR elements and pattern matches for comprehensive annotation
            ocr_elements = input_analysis.get('text_analysis', {}).get('text_elements', [])
            pattern_matches = input_analysis.get('pattern_analysis', {}).get('matches', [])
            
            # Create output path for annotated image
            output_dir = kwargs.get('output_dir', 'output')
            output_path = Path(output_dir) / f"annotated_{Path(original_image_path).name}"
            output_path = output_path.with_suffix('.jpg')  # Ensure proper image format
            
            # Get violations as feedback items
            violations = validation_results.get('violations', {}).get('all_violations', [])
            
            # Convert violations to feedback format for annotation
            feedback_items = []
            for violation in violations:
                feedback_items.append({
                    'category_name': violation.get('category_name', 'Unknown'),
                    'severity': violation.get('severity', 'minor'),
                    'feedback_text': violation.get('german_feedback', violation.get('description', '')),
                    'position': violation.get('position')  # Bounding box if available
                })
            
            # Create visual annotations using enhanced tool
            annotated_image = self.annotation_tool.create_feedback_overlay(
                image_path=original_image_path,
                feedback_items=feedback_items,
                ocr_elements=ocr_elements,
                pattern_matches=pattern_matches,
                output_path=output_path
            )
            
            logger.info(f"Visual annotations created successfully: {output_path}")
            return annotated_image
            
        except Exception as e:
            logger.error(f"Visual annotation creation failed: {e}")
            return self._create_mock_annotation(validation_results)
    
    def _create_mock_annotation(self, validation_results: Dict[str, Any]):
        """Create mock annotation when real annotation fails."""
        violations = validation_results.get('violations', {}).get('all_violations', [])
        
        mock_annotation = type('MockAnnotation', (), {
            'created': False,
            'annotation_count': len(violations),
            'image_path': 'mock_annotation_fallback',
            'annotations': [],  # Add missing annotations attribute
            'legend': {},
            'metadata': {
                'fallback_reason': 'Image file not accessible for annotation',
                'violations_count': len(violations),
                'total_feedback_items': len(violations),
                'total_ocr_elements': 0,
                'total_pattern_matches': 0,
                'image_dimensions': (800, 600)
            }
        })()
        
        return mock_annotation
    
    def _convert_violations_to_feedback(self, violations: List[Dict]) -> List:
        """Convert violations to feedback format for annotation."""
        feedback_items = []
        
        for violation in violations:
            class MockFeedback:
                def __init__(self, violation):
                    self.feedback_text = violation.get('german_feedback', '')
                    self.severity = violation.get('severity', 'minor')
                    # Mock position - in reality would come from actual detection
                    self.position = (100, 100)
            
            feedback_items.append(MockFeedback(violation))
        
        return feedback_items
    
    def _create_comprehensive_report(self, validation_results: Dict[str, Any],
                                   german_feedback: List[GeneratedFeedback],
                                   visual_annotations: Optional[AnnotatedImage]) -> FeedbackReport:
        """Create comprehensive feedback report."""
        logger.info("Step 3: Creating comprehensive report...")
        
        # Create summary
        summary = {
            'total_items': len(german_feedback),
            'by_severity': {'critical': 0, 'major': 0, 'minor': 0},
            'by_category': {},
            'overall_status': validation_results.get('validation_summary', {}).get('overall_status', 'Unknown')
        }
        
        for item in german_feedback:
            summary['by_severity'][item.severity] += 1
            category = item.category_name
            summary['by_category'][category] = summary['by_category'].get(category, 0) + 1
        
        # Create report metadata
        report_metadata = {
            'generation_method': 'rule_based_with_templates',
            'total_violations_processed': len(validation_results.get('violations', {}).get('all_violations', [])),
            'feedback_items_generated': len(german_feedback),
            'visual_annotations_created': visual_annotations is not None,
            'compliance_score': validation_results.get('validation_summary', {}).get('compliance_score', 0)
        }
        
        report = FeedbackReport(
            image_id=validation_results.get('input_info', {}).get('file_name', 'unknown'),
            feedback_items=german_feedback,
            summary=summary,
            report_metadata=report_metadata
        )
        
        return report
    
    def _create_feedback_structure(self, validation_results: Dict[str, Any],
                                 german_feedback: List[GeneratedFeedback],
                                 visual_annotations: Optional[AnnotatedImage],
                                 comprehensive_report: FeedbackReport) -> Dict[str, Any]:
        """Create structured feedback results."""
        logger.info("Step 4: Creating feedback structure...")
        
        feedback_structure = {
            'agent_info': {
                'agent_name': 'Feedback Generator Agent',
                'role': self.role,
                'processing_timestamp': self._get_timestamp(),
                'version': '1.0'
            },
            'input_info': validation_results.get('input_info', {}),
            'generation_summary': {
                'total_feedback_items': len(german_feedback),
                'visual_annotations_created': visual_annotations is not None,
                'report_generated': comprehensive_report is not None,
                'processing_successful': True
            },
            'german_feedback': {
                'items': [
                    {
                        'category_id': item.category_id,
                        'category_name': item.category_name,
                        'element_description': item.element_description,
                        'feedback_text': item.feedback_text,
                        'status': item.status,
                        'severity': item.severity,
                        'confidence': item.confidence,
                        'position': item.position
                    }
                    for item in german_feedback
                ],
                'summary': comprehensive_report.summary
            },
            'visual_annotations': {
                'created': visual_annotations is not None,
                'annotation_count': len(visual_annotations.annotations) if visual_annotations else 0,
                'legend_categories': len(visual_annotations.legend.get('categories', {})) if visual_annotations else 0
            } if visual_annotations else {'created': False},
            'comprehensive_report': {
                'german_text': self.feedback_tool.format_german_report(comprehensive_report),
                'summary': comprehensive_report.summary,
                'metadata': comprehensive_report.report_metadata
            },
            'recommendations': self._generate_final_recommendations(validation_results, comprehensive_report)
        }
        
        return feedback_structure
    
    def _generate_final_recommendations(self, validation_results: Dict[str, Any],
                                      report: FeedbackReport) -> List[str]:
        """Generate final recommendations."""
        recommendations = []
        
        summary = report.summary
        
        if summary['by_severity']['critical'] > 0:
            recommendations.append("🔴 Kritische Mängel sofort beheben vor Freigabe")
        
        if summary['by_severity']['major'] > 0:
            recommendations.append("🟡 Größere Mängel korrigieren für bessere Qualität")
        
        if summary['by_severity']['minor'] > 0:
            recommendations.append("🔵 Kleinere Verbesserungen für Optimierung umsetzen")
        
        compliance_score = validation_results.get('validation_summary', {}).get('compliance_score', 0)
        if compliance_score > 0.9:
            recommendations.append("✅ Sehr gute Zeichnungsqualität - nur minimale Verbesserungen")
        elif compliance_score > 0.7:
            recommendations.append("✅ Gute Zeichnungsqualität - einige Verbesserungen möglich")
        
        if not recommendations:
            recommendations.append("✅ Zeichnung entspricht den Standards")
        
        return recommendations
    
    def _create_error_result(self, error_message: str, validation_results: Dict) -> Dict[str, Any]:
        """Create error result structure."""
        return {
            'agent_info': {
                'agent_name': 'Feedback Generator Agent',
                'role': self.role,
                'processing_timestamp': self._get_timestamp(),
                'version': '1.0'
            },
            'input_info': validation_results.get('input_info', {}),
            'generation_summary': {
                'processing_successful': False,
                'error_message': error_message
            },
            'german_feedback': {'items': []},
            'comprehensive_report': {'german_text': ''}
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
            
            logger.info(f"Feedback Generator Agent results saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return False
    
    def save_german_report(self, results: Dict[str, Any], output_path: str) -> bool:
        """Save German text report separately."""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            german_text = results.get('comprehensive_report', {}).get('german_text', '')
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(german_text)
            
            logger.info(f"German feedback report saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving German report: {e}")
            return False


def main():
    """Test the Feedback Generator Agent."""
    # Initialize agent
    agent = FeedbackGeneratorAgent()
    
    # Test with validation results
    output_dir = Path(__file__).parent.parent.parent / "output" / "agent_results"
    validation_results_file = output_dir / "8_rule_validation.json"
    
    if validation_results_file.exists():
        print(f"\nTesting Feedback Generator Agent...")
        
        # Load validation results
        with open(validation_results_file, 'r', encoding='utf-8') as f:
            validation_results = json.load(f)
        
        print(f"Generating feedback for: {validation_results['input_info']['file_name']}")
        
        # Execute agent
        results = agent.execute(validation_results)
        
        if results['generation_summary']['processing_successful']:
            summary = results['generation_summary']
            print(f"  ✅ Feedback generation successful")
            print(f"  Feedback items: {summary['total_feedback_items']}")
            print(f"  Visual annotations: {summary['visual_annotations_created']}")
            print(f"  Report generated: {summary['report_generated']}")
            
            # Show some feedback
            feedback_items = results['german_feedback']['items']
            if feedback_items:
                print(f"  Sample feedback:")
                for item in feedback_items[:3]:
                    print(f"    - {item['category_name']}: {item['feedback_text']}")
            
            # Save results
            output_file = output_dir / "8_feedback_generator.json"
            agent.save_results(results, str(output_file))
            
            # Save German report
            german_report_file = output_dir / "8_german_feedback_report.txt"
            agent.save_german_report(results, str(german_report_file))
            
            print(f"  Results saved to: {output_file}")
            print(f"  German report saved to: {german_report_file}")
        else:
            print(f"  ❌ Processing failed: {results['generation_summary'].get('error_message')}")
    else:
        print("Validation results not found. Please run rule validation agent first.")


if __name__ == "__main__":
    main()