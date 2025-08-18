# Technical Drawing Feedback System 

## System Overview
 A multi-agent AI system using CrewAI that analyzes technical drawings  and provides feedback by comparing them against corrected drawings and DIN/ISO standards.
 
Multi Agent System Overview:
<img width="1887" height="594" alt="Screenshot 2025-08-17 at 09 50 16" src="https://github.com/user-attachments/assets/5918199b-03b0-4877-a49b-0ed0c4b94324" />


Workflow of the system:
<img width="1019" height="883" alt="Screenshot 2025-08-17 at 09 50 46" src="https://github.com/user-attachments/assets/f38c7cdf-165a-4163-abd4-c00b22e73a7c" />
 Core Tool Implementation

 ## Step 1: Environment Setup

### 1.1 Create Project Structure
```
Create these directories:
- /src (main source code)
- /src/agents (CrewAI agents)
- /src/tools (agent tools)
- /src/utils (utilities)
- /dataset/corrected (corrected PDF samples)
- /dataset/uncorrected (uncorrected PDF samples)
- /output (analysis results)
- /models (ML models and patterns)
- /standards (DIN/ISO rules)
- /tests (test files)

## Step 2: Core Tool Implementation

### 2.1 PDF Processing Tool
Create a tool that:
- Accepts PDF file path as input
- Converts each page to high-resolution image (300 DPI minimum)
- Extracts text layer if present
- Preserves page dimensions and scale
- Returns structured data with images and metadata

### 2.2 OCR Tool
Create a tool that:
- Takes image data as input
- Applies preprocessing (denoise, threshold, deskew)
- Runs Tesseract with technical drawing configuration
- Extracts text with bounding box coordinates
- Returns structured text elements with confidence scores

### 2.3 Image Analysis Tool
Create a tool that:
- Detects lines using Hough transform
- Identifies basic shapes (circles, rectangles)
- Finds dimension lines and arrows
- Locates cross-hatched sections
- Returns geometric elements with coordinates

### 2.4 Dimension Extraction Tool
Create a tool that:
- Parses dimension text (e.g., "Ø55", "40H7", "Ra 3.2")
- Identifies tolerance notations
- Extracts surface finish symbols
- Links dimensions to their reference lines
- Returns structured dimension data

### 2.5 DIN Standards Checker Tool
Create a tool that:
- Loads DIN/ISO standards rules
- Validates general tolerances (ISO 2768)
- Checks surface texture notations (ISO 1302)
- Verifies geometric tolerancing (ISO 1101)
- Returns list of violations with references

### 2.6 Pattern Matching Tool
Create a tool that:
- Loads corrected drawing patterns
- Extracts features from current drawing
- Compares against pattern database
- Calculates similarity scores
- Returns matching patterns with corrections

### 2.7 Feedback Formatting Tool
Create a tool that:
- Takes validation errors as input
- Categorizes by severity (critical/major/minor)
- Groups related errors
- Generates correction suggestions
- Creates structured feedback report

### 2.8 Visual Annotation Tool
Create a tool that:
- Takes original drawing and errors
- Adds colored rectangles around errors
- Adds error numbers and labels
- Creates legend for error types
- Outputs annotated PDF



