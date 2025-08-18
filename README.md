# Technical Drawing Feedback System 


## System Overview
 A multi-agent AI system using CrewAI that analyzes technical drawings  and provides feedback by comparing them against corrected drawings and DIN/ISO standards.
 
Multi Agent System Overview:
<img width="1887" height="594" alt="Screenshot 2025-08-17 at 09 50 16" src="https://github.com/user-attachments/assets/5918199b-03b0-4877-a49b-0ed0c4b94324" />


Workflow of the system:
<img width="1019" height="883" alt="Screenshot 2025-08-17 at 09 50 46" src="https://github.com/user-attachments/assets/f38c7cdf-165a-4163-abd4-c00b22e73a7c" />

## Data Structure
<img width="1561" height="853" alt="Screenshot 2025-08-18 at 08 57 40" src="https://github.com/user-attachments/assets/0dba30ec-dc56-48a3-9368-c83f8a6d12db" />
<img width="1058" height="684" alt="Screenshot 2025-08-18 at 08 59 14" src="https://github.com/user-attachments/assets/5f1b5b15-a720-413e-a964-2518f1c64015" />

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

<img width="405" height="413" alt="Screenshot 2025-08-18 at 11 59 06" src="https://github.com/user-attachments/assets/64cbbdc1-6363-4009-af14-4793e2d75d34" />

