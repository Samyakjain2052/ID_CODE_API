from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from typing import Optional, List, Dict, Any
import easyocr
import re
import io
import json
import os
import time
import logging
import requests
import groq
import numpy as np
import nbformat
from PIL import Image
from pydantic import BaseModel, Field
import uuid
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("api")

# Get API key from environment variable or use fallback
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

# App version and build info
APP_VERSION = "1.0.0"
BUILD_DATE = "2025-03-23"

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Purpose API",
    description="API for ID data extraction and code analysis",
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)  # Compress responses larger than 1KB

#############################
# Data Models               #
#############################

class IDExtractResponse(BaseModel):
    request_id: str = Field(..., description="Unique request identifier")
    name: Optional[str] = Field(None, description="Extracted name from ID")
    dob: Optional[str] = Field(None, description="Extracted date of birth")
    raw_text: str = Field(..., description="Raw extracted text from image")
    llm_response: Optional[str] = Field(None, description="Raw LLM API response")
    processing_time: float = Field(..., description="Processing time in seconds")

class CodeAnalysisResponse(BaseModel):
    request_id: str = Field(..., description="Unique request identifier")
    score: int = Field(..., description="AI probability score (0-100)")
    reasoning: str = Field(..., description="Analysis explanation")
    key_indicators: List[str] = Field(..., description="Key features that influenced the decision")
    file_name: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File type (Python or Jupyter Notebook)")
    processing_time: float = Field(..., description="Processing time in seconds")

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: str

class HealthCheckResponse(BaseModel):
    status: str
    version: str
    build_date: str
    endpoints: Dict[str, str]
    timestamp: str

#######################################
# Request Tracking & Error Handling   #
#######################################

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        process_time = time.time() - start_time
        error_response = ErrorResponse(
            error="Internal Server Error",
            detail=str(e),
            timestamp=datetime.now().isoformat()
        )
        return JSONResponse(
            status_code=500,
            content=error_response.dict(),
            headers={"X-Process-Time": str(process_time)}
        )

def generate_request_id():
    """Generate a unique request ID"""
    return str(uuid.uuid4())

#######################################
# ID Data Extraction Functionality    #
#######################################

# Initialize EasyOCR reader (this loads the model once on startup)
# Use lazy initialization to speed up application startup
_ocr_reader = None

def get_ocr_reader():
    """Lazy initialization of OCR reader"""
    global _ocr_reader
    if _ocr_reader is None:
        logger.info("Initializing EasyOCR reader...")
        _ocr_reader = easyocr.Reader(['en'])
        logger.info("EasyOCR reader initialized successfully")
    return _ocr_reader

def extract_text_from_image(image_bytes):
    """Extract all text from an image using EasyOCR"""
    try:
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to numpy array
        image_np = np.array(image)
        
        # Use EasyOCR to extract text
        reader = get_ocr_reader()
        results = reader.readtext(image_np)
        
        # Combine all detected text
        text = ' '.join([result[1] for result in results])
        return text
    except Exception as e:
        logger.error(f"Error extracting text from image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error extracting text: {str(e)}")

def extract_info_with_llm(text):
    """Use Groq API to extract structured information from ID text"""
    api_url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Improved prompt with explicit formatting instructions
    prompt = f"""
    Extract the name and date of birth from the following ID card text.
    
    Text from ID card:
    {text}
    
    Follow these rules exactly:
    1. Return your answer as a valid JSON object with ONLY these two keys: "name" and "dob"
    2. For name, include the full name as it appears on the ID
    3. For dob, use YYYY-MM-DD format if possible, or the original format from the ID
    4. If you cannot find the information, use null for that field
    5. Do not include any explanation, notes, or additional text
    
    Respond ONLY with the JSON object in this exact format:
    {{
      "name": "Person Name",
      "dob": "YYYY-MM-DD"
    }}
    """
    
    payload = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        llm_response = result["choices"][0]["message"]["content"]
        
        # Try to parse JSON
        try:
            # First try direct JSON parsing
            extracted_data = json.loads(llm_response)
            return extracted_data, llm_response
        except json.JSONDecodeError:
            # Try to extract JSON using regex if direct parsing fails
            json_pattern = r'({.*})'
            json_match = re.search(json_pattern, llm_response, re.DOTALL)
            
            if json_match:
                try:
                    extracted_data = json.loads(json_match.group(1))
                    return extracted_data, llm_response
                except:
                    pass
            
            # If all JSON parsing fails, fall back to regex
            return extract_info_with_regex(text), llm_response
                
    except Exception as e:
        logger.error(f"Error with LLM processing: {str(e)}")
        # Fall back to regex extraction
        return extract_info_with_regex(text), f"Error with LLM processing: {str(e)}"

def extract_info_with_regex(text):
    """Enhanced regex method for extracting information from ID text"""
    # More comprehensive regex patterns for different ID formats
    
    # Name patterns - handles various formats
    name_patterns = [
        r"(?:Name|Full Name)[:\s]+([A-Za-z\s\.]+)",
        r"(?:नाम|பெயர்)[:\s]+([A-Za-z\s\.]+)",  # Hindi/Tamil "Name"
        r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)"  # Capitalized full names
    ]
    
    # DOB patterns - handles various date formats
    dob_patterns = [
        r"(?:Date of Birth|DOB|Birth Date|D\.O\.B\.)[:\s]+(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
        r"(?:Date of Birth|DOB|Birth Date|D\.O\.B\.)[:\s]+(\d{2,4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})",
        r"(?:जन्म तिथि|பிறந்த தேதி)[:\s]+(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",  # Hindi/Tamil "DOB"
        r"(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})"  # Generic date format
    ]
    
    # Try to find name using multiple patterns
    name = None
    for pattern in name_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            break
    
    # Try to find DOB using multiple patterns
    dob = None
    for pattern in dob_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            dob = match.group(1).strip()
            break
    
    return {
        "name": name,
        "dob": dob
    }

#######################################
# Code Analysis Functionality         #
#######################################

def read_python_file_content(file_bytes):
    """Read Python file content from bytes"""
    content = file_bytes.decode("utf-8")
    return content

def extract_code_from_notebook(file_bytes):
    """Extract code from Jupyter notebook"""
    content = file_bytes.decode("utf-8")
    notebook = nbformat.reads(content, as_version=4)
    
    code_cells = []
    for cell in notebook.cells:
        if cell.cell_type == "code":
            code_cells.append(cell.source)
    
    return "\n\n".join(code_cells)

def analyze_code_with_groq(code):
    """Analyze code using Groq's LLM API"""
    client = groq.Client(api_key=GROQ_API_KEY)
    
    # Truncate very large code samples to prevent API errors
    max_code_length = 10000
    if len(code) > max_code_length:
        logger.warning(f"Code length ({len(code)}) exceeds maximum. Truncating to {max_code_length} characters.")
        code = code[:max_code_length] + "\n\n# [Code truncated due to length...]"
    
    prompt = f"""
    Please analyze the following code and determine whether it was likely written by a human or an AI. 
    Provide a score from 0 to 100, where 0 means definitely human-written and 100 means definitely AI-written.
    
    Consider these factors:
    - Code structure and organization
    - Variable naming patterns
    - Comment style and frequency
    - Consistency in coding style
    - Error handling approaches
    - Use of idioms specific to the language
    
    CODE TO ANALYZE:
    ```python
    {code}
    ```
    
    Respond with a JSON object containing:
    - score: The AI probability score (0-100)
    - reasoning: Your detailed analysis explaining the score
    - key_indicators: List of specific features that influenced your decision
    """
    
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are an expert at distinguishing between AI-generated and human-written code."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=2000,
            response_format={"type": "json_object"},
            timeout=60  # Increase timeout for large code samples
        )
        
        analysis = json.loads(response.choices[0].message.content)
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing code: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing code: {str(e)}")

###################
# API Endpoints   #
###################

@app.get("/", response_model=HealthCheckResponse, tags=["Health"])
async def read_root():
    """API health check endpoint"""
    return {
        "status": "online",
        "version": APP_VERSION,
        "build_date": BUILD_DATE,
        "endpoints": {
            "/id/extract": "Extract data from ID images",
            "/code/analyze": "Analyze code to detect if it's AI-generated"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """API health check endpoint"""
    return {
        "status": "online",
        "version": APP_VERSION,
        "build_date": BUILD_DATE,
        "endpoints": {
            "/id/extract": "Extract data from ID images",
            "/code/analyze": "Analyze code to detect if it's AI-generated"
        },
        "timestamp": datetime.now().isoformat()
    }

###################
# ID Endpoints    #
###################

@app.post("/id/extract", response_model=IDExtractResponse, tags=["ID Extraction"])
async def extract_id_data(file: UploadFile = File(...)):
    """
    Extract data from ID card image
    
    - **file**: Image file of an ID card
    
    Returns name, date of birth, and raw extracted text.
    """
    start_time = time.time()
    request_id = generate_request_id()
    logger.info(f"Processing ID extraction request {request_id}")
    
    # Validate file
    if not file.content_type.startswith('image/'):
        logger.warning(f"Request {request_id}: Invalid file type {file.content_type}")
        raise HTTPException(status_code=400, detail="Only image files are allowed")
    
    # Read file
    contents = await file.read()
    
    # Extract text from image
    logger.info(f"Request {request_id}: Extracting text from image")
    text = extract_text_from_image(contents)
    
    if not text:
        logger.warning(f"Request {request_id}: No text extracted from image")
        raise HTTPException(
            status_code=422,
            detail="Could not extract text from image. Please try with a clearer image."
        )
    
    # Process with LLM
    logger.info(f"Request {request_id}: Processing text with LLM")
    result, raw_llm_response = extract_info_with_llm(text)
    
    # Add raw text to result
    result["raw_text"] = text
    result["llm_response"] = raw_llm_response
    result["request_id"] = request_id
    result["processing_time"] = time.time() - start_time
    
    logger.info(f"Request {request_id}: Extraction completed in {result['processing_time']:.2f}s")
    return result

###################
# Code Endpoints  #
###################

@app.post("/code/analyze", response_model=CodeAnalysisResponse, tags=["Code Analysis"])
async def analyze_code(file: UploadFile = File(...)):
    """
    Analyze code to determine if it's AI-generated
    
    - **file**: Python (.py) or Jupyter notebook (.ipynb) file
    
    Returns an AI probability score and analysis.
    """
    start_time = time.time()
    request_id = generate_request_id()
    logger.info(f"Processing code analysis request {request_id}")
    
    file_name = file.filename
    
    # Check if file type is supported
    if not file_name.lower().endswith(('.py', '.ipynb')):
        logger.warning(f"Request {request_id}: Unsupported file format {file_name}")
        raise HTTPException(
            status_code=400, 
            detail="Unsupported file format. Only Python (.py) and Jupyter notebook (.ipynb) files are allowed."
        )
    
    # Read file content
    file_content = await file.read()
    
    # Process based on file type
    file_extension = file_name.split('.')[-1].lower()
    
    if file_extension == 'py':
        logger.info(f"Request {request_id}: Processing Python file")
        code_content = read_python_file_content(file_content)
        file_type = "Python"
    elif file_extension == 'ipynb':
        logger.info(f"Request {request_id}: Processing Jupyter notebook")
        try:
            code_content = extract_code_from_notebook(file_content)
            file_type = "Jupyter Notebook"
        except Exception as e:
            logger.error(f"Request {request_id}: Error processing notebook: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Error processing notebook: {str(e)}"
            )
    else:
        # This should never happen due to the earlier check
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    # Analyze the code
    try:
        logger.info(f"Request {request_id}: Analyzing code with AI")
        analysis = analyze_code_with_groq(code_content)
        
        # Add file information to response
        analysis['file_name'] = file_name
        analysis['file_type'] = file_type
        analysis['request_id'] = request_id
        analysis['processing_time'] = time.time() - start_time
        
        logger.info(f"Request {request_id}: Analysis completed in {analysis['processing_time']:.2f}s")
        return analysis
    except Exception as e:
        logger.error(f"Request {request_id}: Error analyzing code: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing code: {str(e)}"
        )

# Run the app
if __name__ == "__main__":
    import uvicorn
    
    # Determine port from environment variable or use default
    port = int(os.environ.get("PORT", 8080))
    
    # Determine whether to enable debug mode
    debug_mode = os.environ.get("DEBUG", "false").lower() == "true"
    
    # Log startup information
    logger.info(f"Starting Multi-Purpose API v{APP_VERSION}")
    logger.info(f"Debug mode: {debug_mode}")
    logger.info(f"Listening on port: {port}")
    
    # Run the API with proper production settings
    uvicorn.run(
        "final_api:app",
        host="0.0.0.0",  # Listen on all interfaces for container compatibility
        port=port,
        reload=debug_mode,  # Only enable reload in debug mode
        log_level="info"
    )