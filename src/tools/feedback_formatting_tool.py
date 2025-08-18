#!/usr/bin/env python3
"""
Feedback Formatting Tool for Technical Drawing Feedback System

This tool generates German technical feedback using your training data patterns
and formats comprehensive feedback reports with severity rankings.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import json
import random

logger = logging.getLogger(__name__)


@dataclass
class FeedbackTemplate:
    """Template for generating feedback."""
    category_id: int
    category_name: str
    element_patterns: List[str]
    feedback_patterns: List[str] 
    status_patterns: List[str]
    severity: str  # critical, major, minor


@dataclass
class GeneratedFeedback:
    """Generated feedback item."""
    category_id: int
    category_name: str
    element_description: str
    feedback_text: str
    status: str
    severity: str
    confidence: float
    position: Optional[Tuple[int, int]] = None


@dataclass
class FeedbackReport:
    """Complete feedback report."""
    image_id: str
    feedback_items: List[GeneratedFeedback]
    summary: Dict
    report_metadata: Dict


class FeedbackFormattingTool:
    """Tool for generating German technical feedback."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize feedback formatting tool.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.feedback_templates: Dict[int, FeedbackTemplate] = {}
        self.training_patterns: Dict[int, List[str]] = {}
        
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'language': 'de',
            'severity_mapping': {
                'critical': ['fehlt', 'falsch', 'unvollständig'],
                'major': ['überprüfen', 'korrigieren', 'anpassen'],
                'minor': ['verbessern', 'optimieren', 'ergänzen']
            },
            'confidence_threshold': 0.6,
            'max_feedback_items': 20
        }
    
    def load_training_feedback(self, dataset_root: str) -> bool:
        """
        Load German feedback patterns from your training data.
        
        Args:
            dataset_root: Root directory of dataset
            
        Returns:
            True if loading successful
        """
        try:
            # Import required modules
            import sys
            sys.path.append(str(Path(__file__).parent.parent / "utils"))
            from data_pipeline import DataPipeline
            
            # Load dataset
            pipeline = DataPipeline(dataset_root)
            drawings_data = pipeline.load_all_drawings()
            
            logger.info(f"Loading feedback patterns from {len(drawings_data)} drawings")
            
            # Collect patterns by category
            category_feedback = {}
            
            for drawing_id, drawing_data in drawings_data.items():
                for feedback_item in drawing_data.feedback_items:
                    category_id = feedback_item.category_id
                    
                    if category_id not in category_feedback:
                        category_feedback[category_id] = {
                            'elements': [],
                            'feedback': [],
                            'status': []
                        }
                    
                    category_feedback[category_id]['elements'].append(feedback_item.element_description)
                    category_feedback[category_id]['feedback'].append(feedback_item.feedback_text)
                    category_feedback[category_id]['status'].append(feedback_item.status)
            
            # Create templates
            for category_id, patterns in category_feedback.items():
                template = self._create_feedback_template(category_id, patterns)
                if template:
                    self.feedback_templates[category_id] = template
            
            logger.info(f"Created templates for {len(self.feedback_templates)} categories")
            return True
            
        except Exception as e:
            logger.error(f"Error loading training feedback: {e}")
            return False
    
    def _create_feedback_template(self, category_id: int, patterns: Dict) -> Optional[FeedbackTemplate]:
        """Create feedback template from training patterns."""
        try:
            from ..utils.data_pipeline import TechnicalDrawingCategory
            
            category_name = TechnicalDrawingCategory.get_category_name(category_id)
            
            # Analyze patterns to determine severity
            feedback_texts = patterns['feedback']
            severity = self._determine_severity(feedback_texts)
            
            template = FeedbackTemplate(
                category_id=category_id,
                category_name=category_name,
                element_patterns=list(set(patterns['elements'])),
                feedback_patterns=list(set(patterns['feedback'])),
                status_patterns=list(set(patterns['status'])),
                severity=severity
            )
            
            return template
            
        except Exception as e:
            logger.debug(f"Failed to create template for category {category_id}: {e}")
            return None
    
    def _determine_severity(self, feedback_texts: List[str]) -> str:
        """Determine severity based on feedback text patterns."""
        severity_keywords = self.config['severity_mapping']
        
        critical_count = 0
        major_count = 0
        minor_count = 0
        
        for text in feedback_texts:
            text_lower = text.lower()
            
            for keyword in severity_keywords['critical']:
                if keyword in text_lower:
                    critical_count += 1
                    break
            else:
                for keyword in severity_keywords['major']:
                    if keyword in text_lower:
                        major_count += 1
                        break
                else:
                    minor_count += 1
        
        # Determine predominant severity
        if critical_count > major_count and critical_count > minor_count:
            return 'critical'
        elif major_count > minor_count:
            return 'major'
        else:
            return 'minor'
    
    def generate_feedback_for_matches(self, pattern_matches: List, 
                                    image_id: str = "unknown") -> FeedbackReport:
        """
        Generate German feedback based on pattern matches.
        
        Args:
            pattern_matches: List of MatchResult objects
            image_id: Identifier for the image
            
        Returns:
            FeedbackReport with generated feedback
        """
        feedback_items = []
        
        for match in pattern_matches:
            pattern = match.pattern
            category_id = pattern.category_id
            
            # Generate feedback for this match
            if category_id in self.feedback_templates:
                template = self.feedback_templates[category_id]
                
                generated_feedback = self._generate_feedback_from_template(
                    template, match
                )
                
                if generated_feedback:
                    feedback_items.append(generated_feedback)
        
        # Generate summary
        summary = self._generate_summary(feedback_items)
        
        # Compile metadata
        report_metadata = {
            'generation_method': 'template_based',
            'total_matches_processed': len(pattern_matches),
            'feedback_items_generated': len(feedback_items),
            'templates_used': len([t for t in self.feedback_templates.keys() 
                                 if any(f.category_id == t for f in feedback_items)])
        }
        
        return FeedbackReport(
            image_id=image_id,
            feedback_items=feedback_items,
            summary=summary,
            report_metadata=report_metadata
        )
    
    def _generate_feedback_from_template(self, template: FeedbackTemplate, 
                                       match) -> Optional[GeneratedFeedback]:
        """Generate feedback from template and match."""
        try:
            # Select random patterns (or use more sophisticated selection)
            element_pattern = random.choice(template.element_patterns)
            feedback_pattern = random.choice(template.feedback_patterns)
            status_pattern = random.choice(template.status_patterns)
            
            # Adapt patterns to current context
            element_description = self._adapt_element_pattern(element_pattern, match)
            feedback_text = self._adapt_feedback_pattern(feedback_pattern, match)
            status = self._adapt_status_pattern(status_pattern, match)
            
            generated_feedback = GeneratedFeedback(
                category_id=template.category_id,
                category_name=template.category_name,
                element_description=element_description,
                feedback_text=feedback_text,
                status=status,
                severity=template.severity,
                confidence=match.confidence,
                position=(match.matched_region[0] + match.matched_region[2]//2,
                         match.matched_region[1] + match.matched_region[3]//2)
            )
            
            return generated_feedback
            
        except Exception as e:
            logger.debug(f"Failed to generate feedback from template: {e}")
            return None
    
    def _adapt_element_pattern(self, pattern: str, match) -> str:
        """Adapt element pattern to current context."""
        # Simple adaptation - in practice this would be more sophisticated
        return pattern
    
    def _adapt_feedback_pattern(self, pattern: str, match) -> str:
        """Adapt feedback pattern to current context."""
        # Add confidence-based modifications
        if match.confidence < 0.7:
            return f"{pattern} (Überprüfung empfohlen)"
        return pattern
    
    def _adapt_status_pattern(self, pattern: str, match) -> str:
        """Adapt status pattern to current context."""
        # Adjust status based on match confidence
        if match.confidence > 0.8:
            return "Sicher"
        elif "Sicher" in pattern:
            return "Wahrscheinlich"
        return pattern
    
    def _generate_summary(self, feedback_items: List[GeneratedFeedback]) -> Dict:
        """Generate summary of feedback items."""
        summary = {
            'total_items': len(feedback_items),
            'by_severity': {'critical': 0, 'major': 0, 'minor': 0},
            'by_category': {},
            'overall_status': 'unknown'
        }
        
        for item in feedback_items:
            # Count by severity
            summary['by_severity'][item.severity] += 1
            
            # Count by category
            category = item.category_name
            summary['by_category'][category] = summary['by_category'].get(category, 0) + 1
        
        # Determine overall status
        if summary['by_severity']['critical'] > 0:
            summary['overall_status'] = 'Kritische Probleme gefunden'
        elif summary['by_severity']['major'] > 0:
            summary['overall_status'] = 'Größere Probleme gefunden'
        elif summary['by_severity']['minor'] > 0:
            summary['overall_status'] = 'Kleinere Verbesserungen möglich'
        else:
            summary['overall_status'] = 'Keine Probleme erkannt'
        
        return summary
    
    def format_german_report(self, feedback_report: FeedbackReport) -> str:
        """
        Format feedback report in German.
        
        Args:
            feedback_report: FeedbackReport to format
            
        Returns:
            Formatted German report as string
        """
        report_lines = []
        
        # Header
        report_lines.append(f"=== TECHNISCHE ZEICHNUNG ANALYSE ===")
        report_lines.append(f"Zeichnung: {feedback_report.image_id}")
        report_lines.append(f"Status: {feedback_report.summary['overall_status']}")
        report_lines.append("")
        
        # Summary
        report_lines.append("=== ZUSAMMENFASSUNG ===")
        report_lines.append(f"Gefundene Probleme: {feedback_report.summary['total_items']}")
        
        severity_summary = feedback_report.summary['by_severity']
        report_lines.append(f"  - Kritisch: {severity_summary['critical']}")
        report_lines.append(f"  - Größer: {severity_summary['major']}")
        report_lines.append(f"  - Kleiner: {severity_summary['minor']}")
        report_lines.append("")
        
        # Detailed feedback by category
        report_lines.append("=== DETAILLIERTE RÜCKMELDUNG ===")
        
        # Group by category
        by_category = {}
        for item in feedback_report.feedback_items:
            category = item.category_name
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(item)
        
        # Sort by severity
        severity_order = {'critical': 0, 'major': 1, 'minor': 2}
        
        for category, items in by_category.items():
            report_lines.append(f"\n{category.upper()}:")
            
            # Sort items by severity within category
            sorted_items = sorted(items, key=lambda x: severity_order[x.severity])
            
            for i, item in enumerate(sorted_items, 1):
                severity_marker = {
                    'critical': '🔴',
                    'major': '🟡', 
                    'minor': '🔵'
                }.get(item.severity, '⚪')
                
                report_lines.append(f"  {i}. {severity_marker} {item.element_description}")
                report_lines.append(f"     {item.feedback_text}")
                report_lines.append(f"     Status: {item.status}")
                if item.confidence:
                    report_lines.append(f"     Vertrauen: {item.confidence:.1%}")
                report_lines.append("")
        
        # Footer
        report_lines.append("=== EMPFEHLUNGEN ===")
        
        if severity_summary['critical'] > 0:
            report_lines.append("• Kritische Probleme sofort beheben")
        if severity_summary['major'] > 0:
            report_lines.append("• Größere Probleme vor Freigabe korrigieren")
        if severity_summary['minor'] > 0:
            report_lines.append("• Kleinere Verbesserungen für Qualitätssteigerung")
        
        report_lines.append("")
        report_lines.append("🤖 Generiert mit Technical Drawing Feedback System")
        
        return "\n".join(report_lines)
    
    def save_feedback_report(self, feedback_report: FeedbackReport, 
                           output_dir: Union[str, Path],
                           formats: List[str] = ['json', 'txt']) -> bool:
        """
        Save feedback report in specified formats.
        
        Args:
            feedback_report: Report to save
            output_dir: Directory to save reports
            formats: List of formats ('json', 'txt', 'html')
            
        Returns:
            True if save successful
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            base_filename = f"{feedback_report.image_id}_feedback"
            
            # Save JSON format
            if 'json' in formats:
                json_data = {
                    'image_id': feedback_report.image_id,
                    'feedback_items': [
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
                        for item in feedback_report.feedback_items
                    ],
                    'summary': feedback_report.summary,
                    'report_metadata': feedback_report.report_metadata
                }
                
                json_file = output_dir / f"{base_filename}.json"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Saved JSON report to {json_file}")
            
            # Save text format
            if 'txt' in formats:
                german_report = self.format_german_report(feedback_report)
                
                txt_file = output_dir / f"{base_filename}.txt"
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write(german_report)
                
                logger.info(f"Saved text report to {txt_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving feedback report: {e}")
            return False


def main():
    """Test the feedback formatting tool."""
    # Initialize tool
    tool = FeedbackFormattingTool()
    
    # Test with dataset
    dataset_root = Path(__file__).parent.parent.parent / "dataset"
    
    if dataset_root.exists():
        print(f"\nTesting Feedback Formatting Tool...")
        
        # Load training feedback patterns
        print("Loading German feedback patterns...")
        success = tool.load_training_feedback(str(dataset_root))
        
        if success:
            print(f"Created templates for {len(tool.feedback_templates)} categories:")
            for category_id, template in tool.feedback_templates.items():
                print(f"  Category {category_id} ({template.category_name}): {template.severity}")
                print(f"    Elements: {len(template.element_patterns)}")
                print(f"    Feedback: {len(template.feedback_patterns)}")
            
            # Test generation with dummy matches
            print("\nTesting feedback generation with sample data...")
            
            # This would normally come from pattern matching results
            dummy_matches = []  # Would need actual MatchResult objects
            
            # For demonstration, show what the tool can do
            print("Tool ready for generating German technical feedback!")
            
        else:
            print("Failed to load training feedback")
    else:
        print("Dataset directory not found. Please run from project root.")


if __name__ == "__main__":
    main()