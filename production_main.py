#!/usr/bin/env python3
"""
Technical Drawing Feedback System - Production Script
====================================================

Main production script for analyzing German technical drawings with AI.
Provides comprehensive feedback using 5 specialized CrewAI agents.

Usage:
    python production_main.py --file path/to/drawing.pdf
    python production_main.py --batch path/to/directory/
    python production_main.py --file drawing.jpg --output custom_output/
    python production_main.py --help

Features:
- Single file and batch processing
- PDF, JPG, PNG support
- German feedback generation
- DIN/ISO standards validation
- Visual annotations
- Learning system integration
- Multiple output formats
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import time
from datetime import datetime
import traceback

# Add source to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import system components
from agents.document_parser_agent import DocumentParserAgent
from agents.pattern_recognition_agent import PatternRecognitionAgent
from agents.rule_validation_agent import RuleValidationAgent
from agents.feedback_generator_agent import FeedbackGeneratorAgent
from agents.learning_agent import LearningAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TechnicalDrawingProductionSystem:
    """
    Production system for technical drawing analysis.
    
    Main entry point for processing technical drawings in production environment.
    Coordinates all 5 agents and provides comprehensive German feedback.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize production system."""
        self.config = self._load_configuration(config_file)
        self.agents = {}
        self.session_stats = {
            'files_processed': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'total_processing_time': 0,
            'session_start': datetime.now()
        }
        
        logger.info("🚀 Initializing Technical Drawing Feedback System v1.0")
        self._initialize_agents()
    
    def _load_configuration(self, config_file: Optional[str]) -> Dict:
        """Load system configuration."""
        default_config = {
            'system': {
                'name': 'Technical Drawing Feedback System',
                'version': '1.0.0',
                'language': 'de',
                'max_processing_time_minutes': 10
            },
            'agents': {
                'feedback_generator': {
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
                            'critical': (255, 0, 0),
                            'major': (255, 165, 0),
                            'minor': (255, 255, 0),
                            'correct': (0, 255, 0),
                            'unknown': (128, 128, 128)
                        },
                        'category_colors': {
                            0: (128, 128, 128),  # Unknown
                            1: (255, 0, 255),    # General
                            2: (255, 0, 0),      # Schraubenkopf
                            3: (0, 255, 0),      # Scheibe
                            4: (0, 0, 255),      # Platte
                            5: (255, 255, 0),    # Gewindereserve
                            6: (255, 165, 0),    # Grundloch
                            7: (128, 0, 128),    # Gewindedarstellung
                            8: (0, 255, 255),    # Schraffur
                            9: (255, 192, 203)   # Schriftfeld
                        }
                    },
                    'output': {
                        'generate_visual_annotations': True,
                        'generate_text_report': True,
                        'include_recommendations': True,
                        'format_versions': ['json', 'txt', 'pdf']
                    }
                }
            },
            'output': {
                'base_directory': 'output/production',
                'create_timestamped_folders': True,
                'save_individual_agent_results': True,
                'formats': ['json', 'txt', 'pdf'],
                'include_performance_metrics': True
            },
            'processing': {
                'supported_formats': ['.pdf', '.jpg', '.jpeg', '.png'],
                'max_file_size_mb': 50,
                'parallel_processing': False,
                'continue_on_error': True
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    # Merge configurations (user config overrides defaults)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Could not load config file {config_file}: {e}")
        
        return default_config
    
    def _initialize_agents(self) -> bool:
        """Initialize all processing agents."""
        try:
            logger.info("🔧 Initializing agents...")
            
            # Get agent configurations
            feedback_config = self.config.get('agents', {}).get('feedback_generator', {})
            
            # Initialize agents
            self.agents = {
                'document_parser': DocumentParserAgent(),
                'pattern_recognition': PatternRecognitionAgent(),
                'rule_validation': RuleValidationAgent(),
                'feedback_generator': FeedbackGeneratorAgent(feedback_config),
                'learning_agent': LearningAgent()
            }
            
            logger.info(f"✅ All {len(self.agents)} agents initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Agent initialization failed: {e}")
            return False
    
    def process_single_file(self, file_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a single technical drawing file.
        
        Args:
            file_path: Path to the technical drawing file
            output_dir: Optional custom output directory
            
        Returns:
            Processing results dictionary
        """
        start_time = time.time()
        file_path = Path(file_path)
        
        logger.info(f"📄 Processing: {file_path.name}")
        
        # Validate file
        if not self._validate_file(file_path):
            return self._create_error_result(f"File validation failed: {file_path}")
        
        # Set up output directory
        if output_dir:
            output_path = Path(output_dir)
        else:
            base_output = Path(self.config['output']['base_directory'])
            if self.config['output']['create_timestamped_folders']:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = base_output / f"analysis_{timestamp}"
            else:
                output_path = base_output
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Process through agent pipeline
        try:
            # Step 1: Document Parser Agent
            logger.info("1️⃣ Document parsing...")
            doc_results = self.agents['document_parser'].execute(str(file_path))
            
            if not doc_results.get('processing_summary', {}).get('processing_successful', False):
                raise Exception("Document parsing failed")
            
            # Step 2: Pattern Recognition Agent
            logger.info("2️⃣ Pattern recognition...")
            pattern_results = self.agents['pattern_recognition'].execute(doc_results)
            
            if not pattern_results.get('recognition_summary', {}).get('processing_successful', False):
                raise Exception("Pattern recognition failed")
            
            # Step 3: Rule Validation Agent
            logger.info("3️⃣ Rule validation...")
            validation_results = self.agents['rule_validation'].execute(pattern_results)
            
            if not validation_results.get('validation_summary', {}).get('processing_successful', False):
                raise Exception("Rule validation failed")
            
            # Step 4: Feedback Generator Agent
            logger.info("4️⃣ Feedback generation...")
            
            # Pass additional context for visual annotation
            feedback_kwargs = {
                'output_dir': str(output_path),
                'original_image_path': self._get_image_path_for_annotation(file_path, doc_results),
                'temp_dir': self._get_temp_dir(output_path)
            }
            
            feedback_results = self.agents['feedback_generator'].execute(
                validation_results, **feedback_kwargs
            )
            
            if not feedback_results.get('generation_summary', {}).get('processing_successful', False):
                raise Exception("Feedback generation failed")
            
            # Step 5: Learning Agent
            logger.info("5️⃣ System learning...")
            learning_results = self.agents['learning_agent'].execute(feedback_results)
            
            # Create comprehensive results
            processing_time = time.time() - start_time
            results = self._create_production_results(
                file_path, doc_results, pattern_results, validation_results, 
                feedback_results, learning_results, processing_time
            )
            
            # Save results
            saved_files = self._save_production_results(results, output_path)
            results['output_files'] = saved_files
            
            # Update session stats
            self.session_stats['files_processed'] += 1
            self.session_stats['successful_analyses'] += 1
            self.session_stats['total_processing_time'] += processing_time
            
            logger.info(f"✅ Processing completed in {processing_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"❌ Processing failed for {file_path.name}: {e}")
            
            # Update session stats
            self.session_stats['files_processed'] += 1
            self.session_stats['failed_analyses'] += 1
            
            return self._create_error_result(str(e), file_path)
    
    def process_batch(self, directory_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Process multiple files in a directory.
        
        Args:
            directory_path: Path to directory containing technical drawings
            output_dir: Optional custom output directory
            
        Returns:
            Batch processing results
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists() or not directory_path.is_dir():
            return self._create_error_result(f"Invalid directory: {directory_path}")
        
        # Find supported files
        supported_extensions = self.config['processing']['supported_formats']
        files_to_process = []
        
        for ext in supported_extensions:
            files_to_process.extend(directory_path.glob(f"*{ext}"))
            files_to_process.extend(directory_path.glob(f"*{ext.upper()}"))
        
        if not files_to_process:
            return self._create_error_result(f"No supported files found in {directory_path}")
        
        logger.info(f"📁 Batch processing: {len(files_to_process)} files found")
        
        # Set up batch output directory
        if output_dir:
            batch_output = Path(output_dir)
        else:
            base_output = Path(self.config['output']['base_directory'])
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_output = base_output / f"batch_analysis_{timestamp}"
        
        batch_output.mkdir(parents=True, exist_ok=True)
        
        # Process files
        batch_results = {
            'batch_info': {
                'directory': str(directory_path),
                'total_files': len(files_to_process),
                'start_time': datetime.now().isoformat(),
                'output_directory': str(batch_output)
            },
            'file_results': {},
            'batch_summary': {}
        }
        
        successful_count = 0
        failed_count = 0
        
        for i, file_path in enumerate(files_to_process, 1):
            logger.info(f"📄 Processing file {i}/{len(files_to_process)}: {file_path.name}")
            
            # Create individual file output directory
            file_output = batch_output / file_path.stem
            
            try:
                result = self.process_single_file(str(file_path), str(file_output))
                batch_results['file_results'][file_path.name] = result
                
                if result.get('success', False):
                    successful_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                logger.error(f"❌ Failed to process {file_path.name}: {e}")
                batch_results['file_results'][file_path.name] = self._create_error_result(str(e), file_path)
                failed_count += 1
                
                if not self.config['processing']['continue_on_error']:
                    break
        
        # Create batch summary
        batch_results['batch_summary'] = {
            'total_files': len(files_to_process),
            'successful': successful_count,
            'failed': failed_count,
            'success_rate': successful_count / len(files_to_process) if files_to_process else 0,
            'completion_time': datetime.now().isoformat(),
            'total_processing_time': sum(
                result.get('processing_time', 0) 
                for result in batch_results['file_results'].values()
                if isinstance(result, dict)
            )
        }
        
        # Save batch summary
        batch_summary_file = batch_output / "batch_summary.json"
        with open(batch_summary_file, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Batch processing completed: {successful_count}/{len(files_to_process)} successful")
        return batch_results
    
    def _get_image_path_for_annotation(self, file_path: Path, doc_results: Dict[str, Any]) -> Optional[str]:
        """Get the image path suitable for annotation."""
        
        # For image files (JPG, PNG), use the original file
        if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            return str(file_path)
        
        # For PDF files, look for extracted page images in document results
        if file_path.suffix.lower() == '.pdf':
            page_details = doc_results.get('page_details', [])
            if page_details and len(page_details) > 0:
                # Use first page for annotation
                first_page = page_details[0]
                if 'image_path' in first_page:
                    return first_page['image_path']
                
                # Try to construct path based on common patterns
                temp_dir = self._get_temp_dir()
                potential_path = temp_dir / f"{file_path.stem}_page_1.png"
                if potential_path.exists():
                    return str(potential_path)
        
        # Fallback: return the original file path for the annotation tool to handle
        return str(file_path)
    
    def _get_temp_dir(self, output_path: Optional[Path] = None) -> Path:
        """Get temporary directory for intermediate files."""
        if output_path:
            temp_dir = output_path / "temp"
        else:
            import tempfile
            temp_dir = Path(tempfile.gettempdir()) / "technical_drawing_temp"
        
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir
    
    def _validate_file(self, file_path: Path) -> bool:
        """Validate input file."""
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False
        
        if file_path.suffix.lower() not in self.config['processing']['supported_formats']:
            logger.error(f"Unsupported file format: {file_path.suffix}")
            return False
        
        max_size_mb = self.config['processing']['max_file_size_mb']
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        if file_size_mb > max_size_mb:
            logger.error(f"File too large: {file_size_mb:.1f}MB > {max_size_mb}MB")
            return False
        
        return True
    
    def _create_production_results(self, file_path: Path, doc_results: Dict, 
                                 pattern_results: Dict, validation_results: Dict,
                                 feedback_results: Dict, learning_results: Dict,
                                 processing_time: float) -> Dict[str, Any]:
        """Create comprehensive production results."""
        return {
            'system_info': {
                'name': self.config['system']['name'],
                'version': self.config['system']['version'],
                'processing_timestamp': datetime.now().isoformat(),
                'file_processed': str(file_path)
            },
            'processing_summary': {
                'file_name': file_path.name,
                'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                'processing_time_seconds': processing_time,
                'all_agents_successful': True,
                'agents_executed': 5
            },
            'analysis_results': {
                'document_analysis': {
                    'text_elements': doc_results.get('processing_summary', {}).get('total_text_elements', 0),
                    'visual_elements': doc_results.get('processing_summary', {}).get('total_visual_elements', 0),
                    'processing_successful': doc_results.get('processing_summary', {}).get('processing_successful', False)
                },
                'pattern_analysis': {
                    'patterns_matched': pattern_results.get('recognition_summary', {}).get('total_patterns_matched', 0),
                    'categories_detected': pattern_results.get('recognition_summary', {}).get('categories_detected', 0),
                    'average_confidence': pattern_results.get('recognition_summary', {}).get('average_confidence', 0)
                },
                'compliance_analysis': {
                    'total_violations': validation_results.get('validation_summary', {}).get('total_violations', 0),
                    'critical_violations': validation_results.get('validation_summary', {}).get('critical_violations', 0),
                    'major_violations': validation_results.get('validation_summary', {}).get('major_violations', 0),
                    'minor_violations': validation_results.get('validation_summary', {}).get('minor_violations', 0),
                    'compliance_score': validation_results.get('validation_summary', {}).get('compliance_score', 0),
                    'overall_status': validation_results.get('validation_summary', {}).get('overall_status', 'Unknown')
                }
            },
            'german_feedback': {
                'feedback_items': feedback_results.get('german_feedback', {}).get('items', []),
                'comprehensive_report': feedback_results.get('comprehensive_report', {}).get('german_text', ''),
                'recommendations': feedback_results.get('recommendations', [])
            },
            'learning_insights': {
                'performance_analyzed': learning_results.get('learning_summary', {}).get('analysis_completed', False),
                'improvement_opportunities': learning_results.get('learning_summary', {}).get('opportunities_identified', 0),
                'system_recommendations': learning_results.get('recommendations', {}).get('all_recommendations', [])
            },
            'agent_details': {
                'document_parser': doc_results,
                'pattern_recognition': pattern_results,
                'rule_validation': validation_results,
                'feedback_generator': feedback_results,
                'learning_agent': learning_results
            },
            'success': True
        }
    
    def _save_production_results(self, results: Dict[str, Any], output_path: Path) -> Dict[str, str]:
        """Save production results to files."""
        saved_files = {}
        file_name = results['processing_summary']['file_name']
        file_stem = Path(file_name).stem
        
        try:
            # Save complete results
            complete_file = output_path / f"{file_stem}_complete_analysis.json"
            with open(complete_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            saved_files['complete_analysis'] = str(complete_file)
            
            # Save German feedback report
            german_text = results['german_feedback']['comprehensive_report']
            if german_text:
                german_file = output_path / f"{file_stem}_feedback_report.txt"
                with open(german_file, 'w', encoding='utf-8') as f:
                    f.write(german_text)
                saved_files['german_report'] = str(german_file)
            
            # Save executive summary
            summary = {
                'file_analysis': results['processing_summary'],
                'key_results': results['analysis_results'],
                'compliance_summary': results['analysis_results']['compliance_analysis'],
                'recommendations': results['german_feedback']['recommendations']
            }
            
            summary_file = output_path / f"{file_stem}_executive_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            saved_files['executive_summary'] = str(summary_file)
            
            # Save individual agent results if requested
            if self.config['output']['save_individual_agent_results']:
                agent_results = results['agent_details']
                for agent_name, agent_data in agent_results.items():
                    agent_file = output_path / f"{file_stem}_{agent_name}.json"
                    with open(agent_file, 'w', encoding='utf-8') as f:
                        json.dump(agent_data, f, indent=2, ensure_ascii=False)
                    saved_files[agent_name] = str(agent_file)
            
            # Check for and include annotated image if created
            feedback_details = results.get('agent_details', {}).get('feedback_generator', {})
            visual_annotations = feedback_details.get('visual_annotations')
            
            if visual_annotations and hasattr(visual_annotations, 'created') and visual_annotations.created:
                # Look for annotated image in output directory
                import glob
                annotated_pattern = output_path / f"annotated_*"
                annotated_files = glob.glob(str(annotated_pattern))
                
                if annotated_files:
                    # Use the first found annotated image
                    annotated_path = annotated_files[0]
                    saved_files['annotated_image'] = annotated_path
                    logger.info(f"📸 Annotated image included: {Path(annotated_path).name}")
            
            logger.info(f"💾 Results saved: {len(saved_files)} files in {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
        
        return saved_files
    
    def _create_error_result(self, error_message: str, file_path: Optional[Path] = None) -> Dict[str, Any]:
        """Create error result structure."""
        return {
            'system_info': {
                'name': self.config['system']['name'],
                'version': self.config['system']['version'],
                'processing_timestamp': datetime.now().isoformat(),
                'file_processed': str(file_path) if file_path else 'unknown'
            },
            'error': error_message,
            'success': False
        }
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get current session statistics."""
        session_duration = datetime.now() - self.session_stats['session_start']
        
        return {
            'session_info': {
                'start_time': self.session_stats['session_start'].isoformat(),
                'duration_minutes': session_duration.total_seconds() / 60,
                'system_version': self.config['system']['version']
            },
            'processing_stats': {
                'files_processed': self.session_stats['files_processed'],
                'successful_analyses': self.session_stats['successful_analyses'],
                'failed_analyses': self.session_stats['failed_analyses'],
                'success_rate': (
                    self.session_stats['successful_analyses'] / self.session_stats['files_processed']
                    if self.session_stats['files_processed'] > 0 else 0
                ),
                'total_processing_time_seconds': self.session_stats['total_processing_time'],
                'average_processing_time_seconds': (
                    self.session_stats['total_processing_time'] / self.session_stats['files_processed']
                    if self.session_stats['files_processed'] > 0 else 0
                )
            }
        }


def create_cli_parser() -> argparse.ArgumentParser:
    """Create command line interface parser."""
    parser = argparse.ArgumentParser(
        description="Technical Drawing Feedback System - Production Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python production_main.py --file drawing.pdf
  python production_main.py --file technical_drawing.jpg --output results/
  python production_main.py --batch drawings_folder/ 
  python production_main.py --batch drawings/ --output batch_results/ --config custom_config.json
  python production_main.py --stats
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--file', '-f',
        type=str,
        help='Process a single technical drawing file (PDF, JPG, PNG)'
    )
    input_group.add_argument(
        '--batch', '-b',
        type=str,
        help='Process all supported files in a directory'
    )
    input_group.add_argument(
        '--stats', '-s',
        action='store_true',
        help='Show system statistics and capabilities'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Custom output directory (default: output/production/)'
    )
    
    # Configuration options
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (JSON format)'
    )
    
    # Logging options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress output except errors'
    )
    
    return parser


def main():
    """Main entry point for production system."""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Configure logging based on arguments
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Show system statistics
    if args.stats:
        print("🚀 Technical Drawing Feedback System v1.0")
        print("=" * 50)
        print("📊 System Capabilities:")
        print("  ✅ PDF, JPG, PNG processing")
        print("  ✅ German technical drawing analysis")
        print("  ✅ 9-category classification system")
        print("  ✅ DIN/ISO standards validation")
        print("  ✅ Pattern matching (71 training patterns)")
        print("  ✅ German feedback generation")
        print("  ✅ Visual annotations")
        print("  ✅ Learning system integration")
        print("  ✅ Batch processing")
        print("  ✅ Multi-format output")
        print("\n🤖 Agent Architecture:")
        print("  1. Document Parser Agent")
        print("  2. Pattern Recognition Agent")
        print("  3. Rule Validation Agent")
        print("  4. Feedback Generator Agent")
        print("  5. Learning Agent")
        return
    
    try:
        # Initialize production system
        system = TechnicalDrawingProductionSystem(args.config)
        
        if args.file:
            # Process single file
            logger.info(f"🎯 Single file processing: {args.file}")
            result = system.process_single_file(args.file, args.output)
            
            if result['success']:
                print(f"\n✅ Processing completed successfully!")
                print(f"📄 File: {result['processing_summary']['file_name']}")
                print(f"⏱️ Time: {result['processing_summary']['processing_time_seconds']:.2f}s")
                print(f"📊 Compliance: {result['analysis_results']['compliance_analysis']['compliance_score']:.1%}")
                print(f"⚠️ Violations: {result['analysis_results']['compliance_analysis']['total_violations']}")
                print(f"💾 Results saved to: {Path(result['output_files']['complete_analysis']).parent}")
            else:
                print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
                
        elif args.batch:
            # Process batch
            logger.info(f"📁 Batch processing: {args.batch}")
            result = system.process_batch(args.batch, args.output)
            
            if 'batch_summary' in result:
                summary = result['batch_summary']
                print(f"\n📊 Batch processing completed!")
                print(f"📁 Directory: {result['batch_info']['directory']}")
                print(f"📄 Files processed: {summary['total_files']}")
                print(f"✅ Successful: {summary['successful']}")
                print(f"❌ Failed: {summary['failed']}")
                print(f"📈 Success rate: {summary['success_rate']:.1%}")
                print(f"⏱️ Total time: {summary['total_processing_time']:.2f}s")
                print(f"💾 Results saved to: {result['batch_info']['output_directory']}")
            else:
                print(f"❌ Batch processing failed: {result.get('error', 'Unknown error')}")
        
        # Show session statistics
        stats = system.get_session_statistics()
        if stats['processing_stats']['files_processed'] > 0:
            print(f"\n📈 Session Statistics:")
            print(f"  Files processed: {stats['processing_stats']['files_processed']}")
            print(f"  Success rate: {stats['processing_stats']['success_rate']:.1%}")
            print(f"  Average time: {stats['processing_stats']['average_processing_time_seconds']:.2f}s")
            
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ System error: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()