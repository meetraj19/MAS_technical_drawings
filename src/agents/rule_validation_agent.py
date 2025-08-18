#!/usr/bin/env python3
"""
Rule Validation Agent for Technical Drawing Feedback System

This CrewAI agent validates technical drawings against DIN/ISO standards
and your specific 9-category validation rules.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import json

logger = logging.getLogger(__name__)


class RuleValidationAgent:
    """
    CrewAI Agent for validating technical drawing compliance.
    
    Role: Check compliance with DIN/ISO standards and drawing rules
    Goal: Identify violations and classify by severity
    Tools: DIN Standards Checker, Category-specific validation rules
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize Rule Validation Agent."""
        self.config = config or self._get_default_config()
        self.validation_rules = self._load_validation_rules()
        
        # Agent metadata
        self.role = "Technical Drawing Standards Validator"
        self.goal = "Ensure compliance with DIN/ISO standards and drawing conventions"
        self.backstory = """You are an expert in German technical drawing standards (DIN/ISO), 
        specializing in mechanical engineering drawings. You validate elements like 
        Schraubenkopf, Scheibe, Platte, and other components against established standards."""
        
        logger.info(f"Initialized {self.role}")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for the agent."""
        return {
            'validation': {
                'strict_mode': False,
                'check_completeness': True,
                'check_accuracy': True,
                'check_standards_compliance': True
            },
            'severity_thresholds': {
                'critical': ['missing_required', 'incorrect_standard', 'safety_violation'],
                'major': ['incomplete_annotation', 'wrong_tolerance', 'unclear_dimension'],
                'minor': ['formatting_issue', 'optimization_possible', 'style_improvement']
            }
        }
    
    def _load_validation_rules(self) -> Dict:
        """Load validation rules for the 9 technical drawing categories."""
        return {
            0: {  # Unknown
                'name': 'Unknown Elements',
                'rules': ['element_should_be_categorized'],
                'severity': 'major'
            },
            2: {  # Schraubenkopf (Screw Head)
                'name': 'Schraubenkopf',
                'rules': [
                    'head_dimensions_specified',
                    'thread_specification_present',
                    'head_shape_defined',
                    'material_specification'
                ],
                'required_dimensions': ['diameter', 'height'],
                'severity': 'critical'
            },
            3: {  # Scheibe (Washer)
                'name': 'Scheibe', 
                'rules': [
                    'inner_diameter_specified',
                    'outer_diameter_specified',
                    'thickness_specified',
                    'material_grade_defined'
                ],
                'required_dimensions': ['inner_diameter', 'outer_diameter', 'thickness'],
                'severity': 'major'
            },
            4: {  # Platte (Plate)
                'name': 'Platte',
                'rules': [
                    'dimensions_complete',
                    'hole_specifications_complete',
                    'surface_finish_specified',
                    'edge_conditions_defined'
                ],
                'required_dimensions': ['length', 'width', 'thickness'],
                'severity': 'critical'
            },
            5: {  # Gewindereserve (Thread Reserve)
                'name': 'Gewindereserve',
                'rules': [
                    'reserve_length_calculated',
                    'thread_pitch_considered',
                    'minimum_engagement_met'
                ],
                'calculations': ['X = 3*P'],
                'severity': 'critical'
            },
            6: {  # Grundloch (Pilot Hole)
                'name': 'Grundloch',
                'rules': [
                    'pilot_diameter_correct',
                    'depth_specified',
                    'chamfer_defined',
                    'drill_angle_specified'
                ],
                'required_specifications': ['diameter', 'depth', 'chamfer'],
                'severity': 'critical'
            },
            7: {  # Gewindedarstellung (Thread Representation)
                'name': 'Gewindedarstellung',
                'rules': [
                    'thread_lines_correct',
                    'pitch_representation_accurate',
                    'end_conditions_shown',
                    'thread_designation_complete'
                ],
                'standards': ['DIN_13', 'ISO_4762'],
                'severity': 'major'
            },
            8: {  # Schraffur (Hatching)
                'name': 'Schraffur',
                'rules': [
                    'hatch_pattern_consistent',
                    'hatch_spacing_uniform',
                    'material_representation_correct',
                    'section_boundaries_clear'
                ],
                'standards': ['DIN_201'],
                'severity': 'minor'
            },
            9: {  # Schriftfeld (Title Block)
                'name': 'Schriftfeld',
                'rules': [
                    'all_fields_completed',
                    'drawing_number_present',
                    'revision_tracking_current',
                    'approval_signatures_present'
                ],
                'required_fields': ['title', 'drawing_number', 'scale', 'date'],
                'severity': 'critical'
            }
        }
    
    def execute(self, recognition_results: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Main execution method for the agent.
        
        Args:
            recognition_results: Output from Pattern Recognition Agent
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing validation results
        """
        logger.info("Rule Validation Agent executing...")
        
        try:
            # Step 1: Validate pattern matches against rules
            pattern_violations = self._validate_pattern_matches(recognition_results)
            
            # Step 2: Check dimension completeness
            dimension_violations = self._validate_dimensions(recognition_results)
            
            # Step 3: Check standards compliance
            standards_violations = self._validate_standards_compliance(recognition_results)
            
            # Step 4: Create comprehensive validation report
            validation_results = self._create_validation_structure(
                recognition_results, pattern_violations, dimension_violations, standards_violations
            )
            
            logger.info("Rule validation completed successfully")
            return validation_results
            
        except Exception as e:
            logger.error(f"Rule Validation Agent failed: {e}")
            return self._create_error_result(str(e), recognition_results)
    
    def _validate_pattern_matches(self, recognition_results: Dict[str, Any]) -> List[Dict]:
        """Validate detected patterns against category-specific rules using content analysis."""
        logger.info("Step 1: Validating pattern matches...")
        
        violations = []
        pattern_matches = recognition_results.get('pattern_matches', {})
        matches_by_category = pattern_matches.get('by_category', {})
        input_analysis = recognition_results.get('input_analysis', {})
        
        # Get actual content data for context-aware validation
        text_elements = input_analysis.get('text_analysis', {}).get('text_elements', [])
        visual_analysis = input_analysis.get('visual_analysis', {})
        line_count = visual_analysis.get('line_count', 0)
        text_count = len(text_elements)
        
        # If no pattern matches found but significant content exists, provide content-based feedback
        if not matches_by_category and line_count > 50:
            violations.append({
                'violation_id': f'no_patterns_detected_{line_count}_{text_count}',
                'category_id': 0,
                'category_name': 'Pattern_Recognition',
                'violation_type': 'detection_limitation',
                'severity': 'minor',
                'description': f'Automatische Mustererkennung in Zeichnung mit {line_count} Linien und {text_count} Texten begrenzt',
                'rule_reference': 'System Analysis',
                'german_feedback': f'Zeichnung erkannt ({line_count} Linien, {text_count} Texte) - Mustererkennung kann verbessert werden'
            })
        
        # Process detected categories with content-aware validation
        for category_id_str, category_data in matches_by_category.items():
            category_id = int(category_id_str)
            category_rules = self.validation_rules.get(category_id, {})
            
            if not category_rules:
                continue
                
            category_name = category_rules['name']
            rules = category_rules.get('rules', [])
            matches = category_data.get('matches', [])
            confidence = category_data.get('average_confidence', 0)
            
            # Content-aware rule checking
            for rule in rules:
                violation = self._check_rule_compliance_content_aware(
                    category_id, category_name, rule, matches, category_data, 
                    text_elements, visual_analysis
                )
                if violation:
                    violations.append(violation)
        
        logger.info(f"Found {len(violations)} content-aware pattern-based violations")
        return violations
    
    def _validate_dimensions(self, recognition_results: Dict[str, Any]) -> List[Dict]:
        """Validate dimension completeness and accuracy."""
        logger.info("Step 2: Validating dimensions...")
        
        violations = []
        specifications = recognition_results.get('extracted_specifications', {})
        dimensions = specifications.get('dimensions', [])
        pattern_matches = recognition_results.get('pattern_matches', {}).get('by_category', {})
        
        for category_id_str, category_data in pattern_matches.items():
            category_id = int(category_id_str)
            category_rules = self.validation_rules.get(category_id, {})
            
            if not category_rules:
                continue
                
            required_dims = category_rules.get('required_dimensions', [])
            category_name = category_rules['name']
            
            # Check if required dimensions are present
            found_dim_types = [dim.get('dimension_type', '') for dim in dimensions]
            
            for required_dim in required_dims:
                if required_dim not in found_dim_types:
                    violations.append({
                        'violation_id': f"missing_dimension_{category_id}_{required_dim}",
                        'category_id': category_id,
                        'category_name': category_name,
                        'violation_type': 'missing_required_dimension',
                        'severity': category_rules.get('severity', 'major'),
                        'description': f"Fehlende {required_dim} Bemaßung für {category_name}",
                        'rule_reference': f"Required dimension: {required_dim}",
                        'german_feedback': f"{category_name}: {required_dim} Bemaßung fehlt"
                    })
        
        logger.info(f"Found {len(violations)} dimension-related violations")
        return violations
    
    def _validate_standards_compliance(self, recognition_results: Dict[str, Any]) -> List[Dict]:
        """Validate compliance with DIN/ISO standards based on actual content analysis."""
        logger.info("Step 3: Validating standards compliance...")
        
        violations = []
        specifications = recognition_results.get('extracted_specifications', {})
        input_analysis = recognition_results.get('input_analysis', {})
        
        # Get actual content analysis data
        text_elements = input_analysis.get('text_analysis', {}).get('text_elements', [])
        visual_analysis = input_analysis.get('visual_analysis', {})
        pattern_matches = recognition_results.get('pattern_matches', {})
        
        # Content-aware tolerance validation
        violations.extend(self._validate_tolerances_content_aware(text_elements, visual_analysis))
        
        # Content-aware dimension validation 
        violations.extend(self._validate_dimensions_content_aware(text_elements, visual_analysis))
        
        # Content-aware surface finish validation
        violations.extend(self._validate_surface_finish_content_aware(text_elements, pattern_matches))
        
        # Content-aware drawing completeness validation
        violations.extend(self._validate_drawing_completeness(input_analysis, pattern_matches))
        
        logger.info(f"Found {len(violations)} content-aware standards compliance violations")
        return violations
    
    def _validate_tolerances_content_aware(self, text_elements: List, visual_analysis: Dict) -> List[Dict]:
        """Validate tolerance specifications based on actual OCR content."""
        violations = []
        
        # Look for tolerance indicators in OCR text
        tolerance_indicators = ['±', '+/-', 'h6', 'H7', 'ISO 2768', 'f', 'm', 'c', 'v']
        found_tolerances = []
        
        for element in text_elements:
            text_content = str(element.get('text', '')).lower()
            for indicator in tolerance_indicators:
                if indicator.lower() in text_content:
                    found_tolerances.append(indicator)
        
        # Only flag missing tolerances if this appears to be a technical drawing with dimensions
        line_count = visual_analysis.get('line_count', 0)
        has_dimensions = any('mm' in str(elem.get('text', '')) or 
                           any(char.isdigit() for char in str(elem.get('text', ''))) 
                           for elem in text_elements)
        
        # Generate violation based on actual content analysis
        if line_count > 100 and has_dimensions and len(found_tolerances) == 0:
            violations.append({
                'violation_id': f'missing_tolerances_{len(text_elements)}',
                'category_id': 0,
                'category_name': 'General',
                'violation_type': 'missing_tolerances',
                'severity': 'major',
                'description': f'Toleranzangaben in technischer Zeichnung mit {line_count} Linien nicht gefunden',
                'rule_reference': 'ISO 2768',
                'german_feedback': f'Allgemeintoleranzen empfohlen für Zeichnung mit {line_count} Linien'
            })
        elif len(found_tolerances) > 0:
            # Positive feedback for found tolerances
            violations.append({
                'violation_id': f'tolerances_found_{len(found_tolerances)}',
                'category_id': 0,
                'category_name': 'General',
                'violation_type': 'standards_compliance',
                'severity': 'minor',
                'description': f'{len(found_tolerances)} Toleranzangaben erkannt',
                'rule_reference': 'ISO 2768',
                'german_feedback': f'Toleranzangaben vorhanden: {", ".join(found_tolerances[:3])}'
            })
        
        return violations
    
    def _validate_dimensions_content_aware(self, text_elements: List, visual_analysis: Dict) -> List[Dict]:
        """Validate dimension specifications based on actual content."""
        violations = []
        
        # Look for dimension indicators
        dimension_patterns = ['mm', 'cm', 'ø', 'R', 'M', '°']
        found_dimensions = []
        
        for element in text_elements:
            text_content = str(element.get('text', ''))
            for pattern in dimension_patterns:
                if pattern in text_content:
                    found_dimensions.append(f"{pattern}: {text_content[:20]}")
        
        line_count = visual_analysis.get('line_count', 0)
        
        # Generate content-aware dimension feedback
        if line_count > 200 and len(found_dimensions) < 3:
            violations.append({
                'violation_id': f'insufficient_dimensions_{len(found_dimensions)}',
                'category_id': 0,
                'category_name': 'Dimensioning',
                'violation_type': 'incomplete_dimensioning',
                'severity': 'major',
                'description': f'Nur {len(found_dimensions)} Bemaßungen in komplexer Zeichnung gefunden',
                'rule_reference': 'DIN 406',
                'german_feedback': f'Bemaßung unvollständig: {len(found_dimensions)} von erwarteten ~{line_count//50} Bemaßungen'
            })
        elif len(found_dimensions) > 5:
            violations.append({
                'violation_id': f'good_dimensioning_{len(found_dimensions)}',
                'category_id': 0,
                'category_name': 'Dimensioning',
                'violation_type': 'good_practice',
                'severity': 'minor',
                'description': f'Umfangreiche Bemaßung mit {len(found_dimensions)} Elementen',
                'rule_reference': 'DIN 406',
                'german_feedback': f'Gute Bemaßung erkannt: {len(found_dimensions)} Dimensionselemente'
            })
        
        return violations
    
    def _validate_surface_finish_content_aware(self, text_elements: List, pattern_matches: Dict) -> List[Dict]:
        """Validate surface finish specifications based on content."""
        violations = []
        
        # Look for surface finish indicators
        surface_indicators = ['Ra', 'Rz', 'µm', '▽', '∇']
        found_surfaces = []
        
        for element in text_elements:
            text_content = str(element.get('text', ''))
            for indicator in surface_indicators:
                if indicator in text_content:
                    found_surfaces.append(text_content[:30])
        
        # Check pattern matches for critical components
        critical_patterns = pattern_matches.get('by_category', {})
        critical_found = len([cat for cat in critical_patterns.keys() if int(cat) in [2, 4, 6]])
        
        if critical_found > 0 and len(found_surfaces) == 0:
            violations.append({
                'violation_id': f'missing_surface_critical_{critical_found}',
                'category_id': 4,
                'category_name': 'Surface_Finish',
                'violation_type': 'missing_surface_specification',
                'severity': 'minor',
                'description': f'Oberflächenangaben bei {critical_found} kritischen Elementen fehlen',
                'rule_reference': 'ISO 1302',
                'german_feedback': f'Oberflächenrauheit für {critical_found} kritische Elemente empfohlen'
            })
        
        return violations
    
    def _validate_drawing_completeness(self, input_analysis: Dict, pattern_matches: Dict) -> List[Dict]:
        """Validate overall drawing completeness based on content analysis."""
        violations = []
        
        # Get analysis data
        text_analysis = input_analysis.get('text_analysis', {})
        visual_analysis = input_analysis.get('visual_analysis', {})
        
        text_count = len(text_analysis.get('text_elements', []))
        line_count = visual_analysis.get('line_count', 0)
        pattern_count = len(pattern_matches.get('by_category', {}))
        
        # Generate context-aware completeness feedback
        if line_count > 500 and text_count < 5:
            violations.append({
                'violation_id': f'missing_text_complex_{text_count}_{line_count}',
                'category_id': 9,
                'category_name': 'Completeness',
                'violation_type': 'missing_annotations',
                'severity': 'major',
                'description': f'Komplexe Zeichnung ({line_count} Linien) mit nur {text_count} Textangaben',
                'rule_reference': 'DIN 6771',
                'german_feedback': f'Beschriftung unvollständig: {text_count} Texte für {line_count} Linien'
            })
        elif text_count > 10 and line_count > 100:
            violations.append({
                'violation_id': f'well_documented_{text_count}_{line_count}',
                'category_id': 9,
                'category_name': 'Documentation',
                'violation_type': 'good_practice',
                'severity': 'minor',
                'description': f'Gut dokumentierte Zeichnung: {text_count} Textelemente, {line_count} Linien',
                'rule_reference': 'DIN 6771',
                'german_feedback': f'Umfangreiche Dokumentation: {text_count} Beschriftungen erkannt'
            })
        
        return violations
    
    def _check_rule_compliance_content_aware(self, category_id: int, category_name: str, 
                                           rule: str, matches: List[Dict], category_data: Dict,
                                           text_elements: List, visual_analysis: Dict) -> Optional[Dict]:
        """Check compliance with rules using content analysis."""
        avg_confidence = category_data.get('average_confidence', 0)
        line_count = visual_analysis.get('line_count', 0)
        text_count = len(text_elements)
        
        # Content-aware rule validation
        if rule == 'element_should_be_categorized' and category_id == 0:
            # Only flag if there's significant content but no categorization
            if line_count > 100:
                return {
                    'violation_id': f'uncategorized_content_{line_count}_{text_count}',
                    'category_id': category_id,
                    'category_name': category_name,
                    'violation_type': 'uncategorized_element',
                    'severity': 'minor',
                    'description': f'Zeichnungsinhalt ({line_count} Linien, {text_count} Texte) teilweise unkategorisiert',
                    'rule_reference': f'{rule} - Content: {line_count} lines',
                    'german_feedback': f'Automatische Kategorisierung für komplexe Zeichnung ({line_count} Linien) begrenzt'
                }
        
        # Confidence-based validation with content context
        elif avg_confidence < 0.7 and line_count > 0:
            severity = self.validation_rules[category_id].get('severity', 'minor')
            # Adjust severity based on content complexity
            if line_count > 300:
                severity = 'minor'  # Lower severity for complex drawings
            
            return {
                'violation_id': f'detection_uncertainty_{category_id}_{line_count}',
                'category_id': category_id,
                'category_name': category_name,
                'violation_type': 'detection_uncertainty',
                'severity': severity,
                'description': f'Unsichere {category_name}-Erkennung in Zeichnung mit {line_count} Linien',
                'rule_reference': f'Confidence: {avg_confidence:.1%} for {line_count} lines',
                'german_feedback': f'{category_name}: Manuelle Überprüfung bei komplexer Zeichnung empfohlen (Vertrauen: {avg_confidence:.1%})'
            }
        
        # Positive feedback for good detection with sufficient content
        elif avg_confidence > 0.8 and len(matches) > 0 and line_count > 50:
            return {
                'violation_id': f'good_detection_{category_id}_{len(matches)}',
                'category_id': category_id,
                'category_name': category_name,
                'violation_type': 'good_detection',
                'severity': 'minor',
                'description': f'Gute {category_name}-Erkennung: {len(matches)} Elemente mit {avg_confidence:.1%} Vertrauen',
                'rule_reference': f'Confidence: {avg_confidence:.1%}',
                'german_feedback': f'{category_name}: {len(matches)} Elemente erfolgreich erkannt (Vertrauen: {avg_confidence:.1%})'
            }
        
        return None
    
    def _check_rule_compliance(self, category_id: int, category_name: str, 
                             rule: str, matches: List[Dict], category_data: Dict) -> Optional[Dict]:
        """Check compliance with a specific rule."""
        avg_confidence = category_data.get('average_confidence', 0)
        
        # Rule-specific checks
        if rule == 'element_should_be_categorized' and category_id == 0:
            return {
                'violation_id': f'uncategorized_elements_{len(matches)}',
                'category_id': category_id,
                'category_name': category_name,
                'violation_type': 'uncategorized_element',
                'severity': 'major',
                'description': f'{len(matches)} Elemente nicht kategorisiert',
                'rule_reference': rule,
                'german_feedback': f'{len(matches)} Elemente konnten nicht kategorisiert werden'
            }
        
        elif avg_confidence < 0.7:
            severity = self.validation_rules[category_id].get('severity', 'minor')
            return {
                'violation_id': f'low_confidence_{category_id}',
                'category_id': category_id,
                'category_name': category_name,
                'violation_type': 'low_confidence_detection',
                'severity': severity,
                'description': f'Niedrige Erkennungsqualität für {category_name}',
                'rule_reference': f'Confidence threshold: 70%, detected: {avg_confidence:.1%}',
                'german_feedback': f'{category_name}: Überprüfung empfohlen (Vertrauen: {avg_confidence:.1%})'
            }
        
        return None
    
    def _create_validation_structure(self, recognition_results: Dict[str, Any],
                                   pattern_violations: List[Dict],
                                   dimension_violations: List[Dict],
                                   standards_violations: List[Dict]) -> Dict[str, Any]:
        """Create structured validation results."""
        logger.info("Step 4: Creating validation structure...")
        
        all_violations = pattern_violations + dimension_violations + standards_violations
        
        # Categorize by severity
        violations_by_severity = {'critical': [], 'major': [], 'minor': []}
        for violation in all_violations:
            severity = violation.get('severity', 'minor')
            violations_by_severity[severity].append(violation)
        
        # Calculate overall compliance score
        total_checks = len(all_violations) + 10  # Base checks
        violations_count = len(all_violations)
        compliance_score = max(0, (total_checks - violations_count) / total_checks)
        
        validation_structure = {
            'agent_info': {
                'agent_name': 'Rule Validation Agent',
                'role': self.role,
                'processing_timestamp': self._get_timestamp(),
                'version': '1.0'
            },
            'input_info': recognition_results.get('input_info', {}),
            'validation_summary': {
                'total_violations': len(all_violations),
                'critical_violations': len(violations_by_severity['critical']),
                'major_violations': len(violations_by_severity['major']),
                'minor_violations': len(violations_by_severity['minor']),
                'compliance_score': compliance_score,
                'overall_status': self._determine_overall_status(violations_by_severity),
                'processing_successful': True
            },
            'violations': {
                'by_severity': violations_by_severity,
                'all_violations': all_violations,
                'by_category': self._group_violations_by_category(all_violations)
            },
            'compliance_analysis': {
                'standards_checked': ['ISO_2768', 'ISO_1302', 'DIN_406'],
                'categories_validated': list(recognition_results.get('pattern_matches', {}).get('by_category', {}).keys()),
                'recommendations': self._generate_recommendations(violations_by_severity)
            }
        }
        
        return validation_structure
    
    def _determine_overall_status(self, violations_by_severity: Dict) -> str:
        """Determine overall compliance status."""
        if violations_by_severity['critical']:
            return 'Kritische Mängel - Überarbeitung erforderlich'
        elif violations_by_severity['major']:
            return 'Größere Mängel - Korrekturen empfohlen'
        elif violations_by_severity['minor']:
            return 'Kleinere Verbesserungen möglich'
        else:
            return 'Standardkonform'
    
    def _group_violations_by_category(self, violations: List[Dict]) -> Dict:
        """Group violations by category."""
        by_category = {}
        
        for violation in violations:
            category_id = violation.get('category_id', 0)
            category_name = violation.get('category_name', 'Unknown')
            
            if category_id not in by_category:
                by_category[category_id] = {
                    'category_name': category_name,
                    'violations': []
                }
            
            by_category[category_id]['violations'].append(violation)
        
        return by_category
    
    def _generate_recommendations(self, violations_by_severity: Dict) -> List[str]:
        """Generate recommendations based on violations."""
        recommendations = []
        
        if violations_by_severity['critical']:
            recommendations.append("Kritische Mängel sofort beheben vor Freigabe")
        
        if violations_by_severity['major']:
            recommendations.append("Größere Mängel korrigieren für bessere Qualität")
        
        if violations_by_severity['minor']:
            recommendations.append("Kleinere Verbesserungen für Optimierung umsetzen")
        
        if not any(violations_by_severity.values()):
            recommendations.append("Zeichnung entspricht den Standards")
        
        return recommendations
    
    def _create_error_result(self, error_message: str, recognition_results: Dict) -> Dict[str, Any]:
        """Create error result structure."""
        return {
            'agent_info': {
                'agent_name': 'Rule Validation Agent',
                'role': self.role,
                'processing_timestamp': self._get_timestamp(),
                'version': '1.0'
            },
            'input_info': recognition_results.get('input_info', {}),
            'validation_summary': {
                'processing_successful': False,
                'error_message': error_message
            },
            'violations': {'all_violations': []}
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
            
            logger.info(f"Rule Validation Agent results saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return False


def main():
    """Test the Rule Validation Agent."""
    # Initialize agent
    agent = RuleValidationAgent()
    
    # Test with pattern recognition results
    output_dir = Path(__file__).parent.parent.parent / "output" / "agent_results"
    pattern_results_file = output_dir / "8_pattern_recognition.json"
    
    if pattern_results_file.exists():
        print(f"\nTesting Rule Validation Agent...")
        
        # Load pattern recognition results
        with open(pattern_results_file, 'r', encoding='utf-8') as f:
            recognition_results = json.load(f)
        
        print(f"Validating document: {recognition_results['input_info']['file_name']}")
        
        # Execute agent
        results = agent.execute(recognition_results)
        
        if results['validation_summary']['processing_successful']:
            summary = results['validation_summary']
            print(f"  ✅ Validation successful")
            print(f"  Total violations: {summary['total_violations']}")
            print(f"  Critical: {summary['critical_violations']}")
            print(f"  Major: {summary['major_violations']}")
            print(f"  Minor: {summary['minor_violations']}")
            print(f"  Compliance score: {summary['compliance_score']:.1%}")
            print(f"  Status: {summary['overall_status']}")
            
            # Save results
            output_file = output_dir / "8_rule_validation.json"
            agent.save_results(results, str(output_file))
            print(f"  Results saved to: {output_file}")
        else:
            print(f"  ❌ Processing failed: {results['validation_summary'].get('error_message')}")
    else:
        print("Pattern recognition results not found. Please run pattern recognition agent first.")


if __name__ == "__main__":
    main()