# Technical Drawing Feedback System 


## System Overview
 A multi-agent AI system using CrewAI that analyzes technical drawings  and provides feedback by comparing them against corrected drawings and DIN/ISO standards.
 
Multi Agent System Overview:
<img width="1887" height="594" alt="Screenshot 2025-08-17 at 09 50 16" src="https://github.com/user-attachments/assets/5918199b-03b0-4877-a49b-0ed0c4b94324" />


Workflow of the system:
<img width="1019" height="883" alt="Screenshot 2025-08-17 at 09 50 46" src="https://github.com/user-attachments/assets/f38c7cdf-165a-4163-abd4-c00b22e73a7c" />


 ## Core Tool Implementation

### 1 PDF Processing Tool
Create a tool that:
- Accepts PDF file path as input
- Converts each page to high-resolution image (300 DPI minimum)
- Extracts text layer if present
- Preserves page dimensions and scale
- Returns structured data with images and metadata

### 2 OCR Tool
Create a tool that:
- Takes image data as input
- Applies preprocessing (denoise, threshold, deskew)
- Runs Tesseract with technical drawing configuration
- Extracts text with bounding box coordinates
- Returns structured text elements with confidence scores

### 3 Image Analysis Tool
Create a tool that:
- Detects lines using Hough transform
- Identifies basic shapes (circles, rectangles)
- Finds dimension lines and arrows
- Locates cross-hatched sections
- Returns geometric elements with coordinates

### 4 Dimension Extraction Tool
Create a tool that:
- Parses dimension text (e.g., "Ø55", "40H7", "Ra 3.2")
- Identifies tolerance notations
- Extracts surface finish symbols
- Links dimensions to their reference lines
- Returns structured dimension data

### 5 DIN Standards Checker Tool
Create a tool that:
- Loads DIN/ISO standards rules
- Validates general tolerances (ISO 2768)
- Checks surface texture notations (ISO 1302)
- Verifies geometric tolerancing (ISO 1101)
- Returns list of violations with references

### 6 Pattern Matching Tool
Create a tool that:
- Loads corrected drawing patterns
- Extracts features from current drawing
- Compares against pattern database
- Calculates similarity scores
- Returns matching patterns with corrections

### 7 Feedback Formatting Tool
Create a tool that:
- Takes validation errors as input
- Categorizes by severity (critical/major/minor)
- Groups related errors
- Generates correction suggestions
- Creates structured feedback report

### 8 Visual Annotation Tool
Create a tool that:
- Takes original drawing and errors
- Adds colored rectangles around errors
- Adds error numbers and labels
- Creates legend for error types
- Outputs annotated PDF

## Agent Implementation

### 1 Document Parser Agent
**Role**: Extract all content from technical drawings
**Tools**: PDF Processing Tool, OCR Tool, Image Analysis Tool
**Tasks**:
- Load PDF and convert to analyzable format
- Extract all text elements
- Identify drawing regions
- Output structured document representation

### 2 Pattern Recognition Agent
**Role**: Identify drawing elements and patterns
**Tools**: Dimension Extraction Tool, Pattern Matching Tool
**Tasks**:
- Find all dimensions and tolerances
- Identify standard symbols
- Match against known patterns
- Output categorized drawing elements

### 3 Rule Validation Agent
**Role**: Check compliance with standards
**Tools**: DIN Standards Checker Tool
**Tasks**:
- Apply relevant standards based on drawing type
- Check each element for compliance
- Identify missing required information
- Output validation errors with severity

### 4 Feedback Generator Agent
**Role**: Create comprehensive feedback
**Tools**: Feedback Formatting Tool, Visual Annotation Tool
**Tasks**:
- Compile all errors from validation
- Generate human-readable explanations
- Create visual annotations
- Output feedback report and annotated drawing

### 5 Learning Agent
**Role**: Improve system over time
**Tools**: Custom learning tools
**Tasks**:
- Store new error patterns
- Update pattern database
- Track correction effectiveness
- Refine detection algorithms

## System Output

### Analysis Results
- **Compliance Scores**: Percentage compliance with standards
- **Violation Detection**: Critical, Major, Minor categorization
- **German Feedback**: Detailed technical feedback in German
- **Pattern Matches**: Recognition of  training patterns
- **Processing Metrics**: Time, confidence, accuracy scores

### Download Formats
- **Complete Analysis**: Full JSON analysis with all data
- **German Report**: Human-readable German feedback (TXT)
- **Executive Summary**: Key metrics and compliance overview (JSON)
  <img width="446" height="553" alt="Screenshot 2025-08-18 at 08 52 15" src="https://github.com/user-attachments/assets/90337491-a9f7-4ed0-9a76-e6984e4d3608" />


