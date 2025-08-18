#!/usr/bin/env python3
"""
CrewAI System Integration for Technical Drawing Feedback System

This module orchestrates all 5 agents to process technical drawings end-to-end,
from document parsing through learning and improvement.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import json
from datetime import datetime

# Add agents to path
sys.path.append(str(Path(__file__).parent))

from agents.document_parser_agent import DocumentParserAgent
from agents.pattern_recognition_agent import PatternRecognitionAgent  
from agents.rule_validation_agent import RuleValidationAgent
from agents.feedback_generator_agent import FeedbackGeneratorAgent
from agents.learning_agent import LearningAgent

logger = logging.getLogger(__name__)


class TechnicalDrawingCrewAI:
    """
    CrewAI system for technical drawing analysis and feedback.
    
    Orchestrates 5 specialized agents:
    1. Document Parser Agent - Extracts content (PDF/OCR/Image Analysis)
    2. Pattern Recognition Agent - Matches against your bounding box database  
    3. Rule Validation Agent - Validates against DIN/ISO standards
    4. Feedback Generator Agent - Creates German feedback with visual annotations
    5. Learning Agent - Improves system over time
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the CrewAI system with all agents."""
        self.config = config or self._get_default_config()
        
        # Initialize all agents
        logger.info("Initializing CrewAI Technical Drawing System...")
        
        self.document_parser = DocumentParserAgent(self.config.get('document_parser', {}))
        self.pattern_recognition = PatternRecognitionAgent(self.config.get('pattern_recognition', {}))
        self.rule_validation = RuleValidationAgent(self.config.get('rule_validation', {}))
        self.feedback_generator = FeedbackGeneratorAgent(self.config.get('feedback_generator', {}))
        self.learning_agent = LearningAgent(self.config.get('learning_agent', {}))
        
        # System metadata
        self.system_info = {
            'name': 'Technical Drawing Feedback System',
            'version': '1.0',
            'agents_count': 5,
            'capabilities': [
                'German technical drawing analysis',
                'Pattern matching with your 71 bounding box patterns',
                'DIN/ISO standards validation',
                'German feedback generation',
                'Visual annotations',
                'System learning and improvement'
            ]
        }
        
        logger.info(f"CrewAI system initialized with {self.system_info['agents_count']} agents")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for the system."""
        return {
            'workflow': {
                'enable_all_agents': True,
                'save_intermediate_results': True,
                'continue_on_agent_failure': True,
                'max_processing_time_minutes': 10
            },
            'output': {
                'base_path': 'output',
                'save_individual_agent_results': True,
                'save_final_report': True,
                'formats': ['json', 'txt']
            },
            'logging': {
                'level': 'INFO',
                'save_logs': True
            }
        }
    
    def process_technical_drawing(self, input_file: str, **kwargs) -> Dict[str, Any]:
        """
        Process a technical drawing through the complete agent workflow.
        
        Args:
            input_file: Path to technical drawing file (PDF, JPG, PNG)
            **kwargs: Additional parameters (user_feedback, ground_truth, etc.)
            
        Returns:
            Complete analysis results with German feedback and recommendations
        """
        logger.info(f"🚀 Starting technical drawing analysis: {input_file}")
        
        workflow_results = {
            'system_info': self.system_info,
            'input_file': input_file,
            'processing_timestamp': datetime.now().isoformat(),
            'workflow_status': 'started',
            'agent_results': {},
            'final_output': {}
        }
        
        try:
            # Step 1: Document Parser Agent
            logger.info("📄 Step 1: Document parsing...")
            document_results = self.document_parser.execute(input_file)
            workflow_results['agent_results']['document_parser'] = document_results
            
            if not document_results.get('processing_summary', {}).get('processing_successful', False):
                return self._handle_agent_failure('document_parser', document_results, workflow_results)
            
            # Step 2: Pattern Recognition Agent  
            logger.info("🔍 Step 2: Pattern recognition...")
            pattern_results = self.pattern_recognition.execute(document_results)
            workflow_results['agent_results']['pattern_recognition'] = pattern_results
            
            if not pattern_results.get('recognition_summary', {}).get('processing_successful', False):
                return self._handle_agent_failure('pattern_recognition', pattern_results, workflow_results)
            
            # Step 3: Rule Validation Agent
            logger.info("✅ Step 3: Rule validation...")
            validation_results = self.rule_validation.execute(pattern_results)
            workflow_results['agent_results']['rule_validation'] = validation_results
            
            if not validation_results.get('validation_summary', {}).get('processing_successful', False):
                return self._handle_agent_failure('rule_validation', validation_results, workflow_results)
            
            # Step 4: Feedback Generator Agent
            logger.info("📝 Step 4: Feedback generation...")
            feedback_results = self.feedback_generator.execute(validation_results, **kwargs)
            workflow_results['agent_results']['feedback_generator'] = feedback_results
            
            if not feedback_results.get('generation_summary', {}).get('processing_successful', False):
                return self._handle_agent_failure('feedback_generator', feedback_results, workflow_results)
            
            # Step 5: Learning Agent
            logger.info("🧠 Step 5: System learning...")
            learning_results = self.learning_agent.execute(feedback_results, **kwargs)
            workflow_results['agent_results']['learning_agent'] = learning_results
            
            # Step 6: Create final comprehensive output
            logger.info("📊 Step 6: Creating final output...")
            final_output = self._create_final_output(workflow_results)
            workflow_results['final_output'] = final_output
            workflow_results['workflow_status'] = 'completed_successfully'
            
            logger.info("✅ Technical drawing analysis completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Workflow failed: {e}")
            workflow_results['workflow_status'] = 'failed'
            workflow_results['error_message'] = str(e)
        
        return workflow_results
    
    def _handle_agent_failure(self, agent_name: str, agent_results: Dict, 
                            workflow_results: Dict) -> Dict[str, Any]:
        """Handle failure of an individual agent."""
        logger.warning(f"⚠️ Agent {agent_name} failed")
        
        if self.config['workflow']['continue_on_agent_failure']:
            logger.info("Continuing workflow despite agent failure...")
            workflow_results['workflow_status'] = 'partial_completion'
            # Continue with available results
            return workflow_results
        else:
            workflow_results['workflow_status'] = 'failed_at_agent'
            workflow_results['failed_agent'] = agent_name
            workflow_results['error_message'] = agent_results.get('error_message', 'Agent failed')
            return workflow_results
    
    def _create_final_output(self, workflow_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive final output."""
        agent_results = workflow_results['agent_results']
        
        # Extract key information from each agent
        document_info = agent_results.get('document_parser', {}).get('processing_summary', {})
        pattern_info = agent_results.get('pattern_recognition', {}).get('recognition_summary', {})
        validation_info = agent_results.get('rule_validation', {}).get('validation_summary', {})
        feedback_info = agent_results.get('feedback_generator', {}).get('generation_summary', {})
        learning_info = agent_results.get('learning_agent', {}).get('learning_summary', {})
        
        # Create executive summary
        executive_summary = {
            'drawing_processed': workflow_results['input_file'],
            'processing_status': workflow_results['workflow_status'],
            'analysis_timestamp': workflow_results['processing_timestamp'],
            'agents_executed': len([a for a in agent_results.keys() if agent_results[a]]),
            'text_elements_found': document_info.get('total_text_elements', 0),
            'patterns_matched': pattern_info.get('total_patterns_matched', 0),
            'violations_found': validation_info.get('total_violations', 0),
            'compliance_score': validation_info.get('compliance_score', 0),
            'feedback_items_generated': feedback_info.get('total_feedback_items', 0),
            'system_learning_active': learning_info.get('analysis_completed', False)
        }
        
        # Extract German feedback
        german_feedback_items = agent_results.get('feedback_generator', {}).get('german_feedback', {}).get('items', [])
        german_report_text = agent_results.get('feedback_generator', {}).get('comprehensive_report', {}).get('german_text', '')
        
        # Extract violations by severity
        violations = agent_results.get('rule_validation', {}).get('violations', {})
        violations_by_severity = violations.get('by_severity', {'critical': [], 'major': [], 'minor': []})
        
        # Extract learning recommendations
        learning_recommendations = agent_results.get('learning_agent', {}).get('recommendations', {}).get('all_recommendations', [])
        
        # Create actionable recommendations
        actionable_recommendations = []
        
        if violations_by_severity['critical']:
            actionable_recommendations.append({
                'priority': 'high',
                'type': 'critical_violations',
                'count': len(violations_by_severity['critical']),
                'message': f"🔴 {len(violations_by_severity['critical'])} kritische Mängel sofort beheben"
            })
        
        if violations_by_severity['major']:
            actionable_recommendations.append({
                'priority': 'medium', 
                'type': 'major_violations',
                'count': len(violations_by_severity['major']),
                'message': f"🟡 {len(violations_by_severity['major'])} größere Mängel korrigieren"
            })
        
        if violations_by_severity['minor']:
            actionable_recommendations.append({
                'priority': 'low',
                'type': 'minor_improvements', 
                'count': len(violations_by_severity['minor']),
                'message': f"🔵 {len(violations_by_severity['minor'])} kleinere Verbesserungen umsetzen"
            })
        
        if executive_summary['compliance_score'] > 0.8:
            actionable_recommendations.append({
                'priority': 'info',
                'type': 'positive_feedback',
                'message': f"✅ Gute Zeichnungsqualität (Konformität: {executive_summary['compliance_score']:.1%})"
            })
        
        final_output = {
            'executive_summary': executive_summary,
            'german_feedback': {
                'report_text': german_report_text,
                'feedback_items': german_feedback_items,
                'total_items': len(german_feedback_items)
            },
            'technical_analysis': {
                'patterns_detected': pattern_info.get('categories_detected', 0),
                'violations_by_severity': violations_by_severity,
                'compliance_score': executive_summary['compliance_score'],
                'overall_status': validation_info.get('overall_status', 'Unknown')
            },
            'actionable_recommendations': actionable_recommendations,
            'system_learning': {
                'performance_analysis_completed': learning_info.get('analysis_completed', False),
                'improvement_opportunities': learning_info.get('opportunities_identified', 0),
                'recommendations': learning_recommendations[:5]  # Top 5 recommendations
            },
            'processing_metadata': {
                'workflow_version': self.system_info['version'],
                'agents_used': list(agent_results.keys()),
                'total_processing_time': self._calculate_processing_time(workflow_results),
                'success_rate': self._calculate_success_rate(agent_results)
            }
        }
        
        return final_output
    
    def _calculate_processing_time(self, workflow_results: Dict) -> str:
        """Calculate total processing time."""
        start_time = datetime.fromisoformat(workflow_results['processing_timestamp'])
        end_time = datetime.now()
        duration = end_time - start_time
        return f"{duration.total_seconds():.1f} seconds"
    
    def _calculate_success_rate(self, agent_results: Dict) -> float:
        """Calculate success rate of agent execution."""
        total_agents = len(agent_results)
        successful_agents = sum(1 for result in agent_results.values() 
                              if result and any(summary.get('processing_successful', False) or 
                                              summary.get('analysis_completed', False) for 
                                              summary in result.values() if isinstance(summary, dict)))
        return successful_agents / total_agents if total_agents > 0 else 0
    
    def save_workflow_results(self, workflow_results: Dict[str, Any], 
                            output_dir: str) -> Dict[str, str]:
        """Save complete workflow results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        file_stem = Path(workflow_results['input_file']).stem
        saved_files = {}
        
        try:
            # Save complete workflow results
            workflow_file = output_dir / f"{file_stem}_complete_analysis.json"
            with open(workflow_file, 'w', encoding='utf-8') as f:
                json.dump(workflow_results, f, indent=2, ensure_ascii=False)
            saved_files['complete_analysis'] = str(workflow_file)
            
            # Save German feedback report
            german_text = workflow_results.get('final_output', {}).get('german_feedback', {}).get('report_text', '')
            if german_text:
                german_file = output_dir / f"{file_stem}_german_feedback.txt"
                with open(german_file, 'w', encoding='utf-8') as f:
                    f.write(german_text)
                saved_files['german_feedback'] = str(german_file)
            
            # Save executive summary
            summary = workflow_results.get('final_output', {}).get('executive_summary', {})
            if summary:
                summary_file = output_dir / f"{file_stem}_executive_summary.json"
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                saved_files['executive_summary'] = str(summary_file)
            
            # Save individual agent results if requested
            if self.config['output']['save_individual_agent_results']:
                agent_results = workflow_results.get('agent_results', {})
                for agent_name, results in agent_results.items():
                    if results:
                        agent_file = output_dir / f"{file_stem}_{agent_name}.json"
                        with open(agent_file, 'w', encoding='utf-8') as f:
                            json.dump(results, f, indent=2, ensure_ascii=False)
                        saved_files[agent_name] = str(agent_file)
            
            logger.info(f"Workflow results saved: {len(saved_files)} files")
            
        except Exception as e:
            logger.error(f"Error saving workflow results: {e}")
        
        return saved_files
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and capabilities."""
        return {
            'system_info': self.system_info,
            'agents_status': {
                'document_parser': 'ready',
                'pattern_recognition': f'ready (patterns: {len(getattr(self.pattern_recognition.pattern_tool, "pattern_database", []))})',
                'rule_validation': 'ready',
                'feedback_generator': f'ready (templates: {len(self.feedback_generator.feedback_tool.feedback_templates)})',
                'learning_agent': 'ready'
            },
            'configuration': self.config,
            'capabilities_summary': {
                'supports_pdf_processing': True,
                'supports_german_feedback': True,
                'supports_visual_annotations': True,
                'supports_pattern_learning': True,
                'supports_standards_validation': True
            }
        }


def main():
    """Test the complete CrewAI system."""
    # Initialize system
    system = TechnicalDrawingCrewAI()
    
    # Show system status
    status = system.get_system_status()
    print(f"\n🚀 {status['system_info']['name']} v{status['system_info']['version']}")
    print(f"Agents: {status['system_info']['agents_count']}")
    print("Agent Status:")
    for agent, status_msg in status['agents_status'].items():
        print(f"  ✅ {agent}: {status_msg}")
    
    # Test with a dataset image
    dataset_root = Path(__file__).parent.parent / "dataset"
    corrected_dir = dataset_root / "corrected"
    
    if corrected_dir.exists():
        print(f"\n🧪 Testing complete system workflow...")
        
        # Process first image
        image_files = list(corrected_dir.glob("*.jpg"))[:1]
        
        for image_file in image_files:
            print(f"\n📄 Processing: {image_file.name}")
            
            # Execute complete workflow
            workflow_results = system.process_technical_drawing(str(image_file))
            
            if workflow_results['workflow_status'] == 'completed_successfully':
                final_output = workflow_results['final_output']
                exec_summary = final_output['executive_summary']
                
                print(f"  ✅ Workflow completed successfully!")
                print(f"  📊 Processing time: {final_output['processing_metadata']['total_processing_time']}")
                print(f"  🎯 Success rate: {final_output['processing_metadata']['success_rate']:.1%}")
                print(f"  📝 Text elements: {exec_summary['text_elements_found']}")
                print(f"  🔍 Patterns matched: {exec_summary['patterns_matched']}")
                print(f"  ⚠️ Violations found: {exec_summary['violations_found']}")
                print(f"  📈 Compliance score: {exec_summary['compliance_score']:.1%}")
                print(f"  💬 Feedback items: {exec_summary['feedback_items_generated']}")
                
                # Show recommendations
                recommendations = final_output['actionable_recommendations']
                if recommendations:
                    print(f"  📋 Recommendations:")
                    for rec in recommendations[:3]:
                        print(f"    {rec['message']}")
                
                # Save results
                output_dir = Path(__file__).parent.parent / "output" / "complete_workflow"
                saved_files = system.save_workflow_results(workflow_results, str(output_dir))
                
                print(f"  💾 Results saved: {len(saved_files)} files")
                for file_type, file_path in saved_files.items():
                    print(f"    - {file_type}: {Path(file_path).name}")
                    
            else:
                print(f"  ❌ Workflow failed: {workflow_results.get('error_message', 'Unknown error')}")
    else:
        print("Dataset directory not found. Please run from project root.")


if __name__ == "__main__":
    main()