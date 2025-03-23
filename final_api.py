from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, List
import easyocr
import re
import io
import json
import requests
import groq
import numpy as np
import nbformat
from PIL import Image
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Purpose API",
    description="API for ID data extraction and code analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Your Groq API key - replace with your actual key
GROQ_API_KEY = "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"

#############################
# Data Models               #
#############################

class IDExtractResponse(BaseModel):
    name: Optional[str] = None
    dob: Optional[str] = None
    raw_text: str
    llm_response: Optional[str] = None

class CodeAnalysisResponse(BaseModel):
    score: int
    reasoning: str
    key_indicators: List[str]
    file_name: str
    file_type: str

#######################################
# ID Data Extraction Functionality    #
#######################################

# Initialize EasyOCR reader (this loads the model once on startup)
reader = easyocr.Reader(['en'])

def extract_text_from_image(image_bytes):
    """Extract all text from an image using EasyOCR"""
    try:
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to numpy array
        image_np = np.array(image)
        
        # Use EasyOCR to extract text
        results = reader.readtext(image_np)
        
        # Combine all detected text
        text = ' '.join([result[1] for result in results])
        return text
    except Exception as e:
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
        "response_format": {"type": "json_object"}  # Request JSON format if supported
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload)
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
            response_format={"type": "json_object"}
        )
        
        analysis = json.loads(response.choices[0].message.content)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing code: {str(e)}")

###################
# API Endpoints   #
###################

@app.get("/")
async def read_root():
    """API root endpoint"""
    return {
        "name": "Multi-Purpose API",
        "version": "1.0.0",
        "endpoints": {
            "/id/extract": "Extract data from ID images",
            "/code/analyze": "Analyze code to detect if it's AI-generated"
        }
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
    # Validate file
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Only image files are allowed")
    
    # Read file
    contents = await file.read()
    
    # Extract text from image
    text = extract_text_from_image(contents)
    
    if not text:
        raise HTTPException(
            status_code=422,
            detail="Could not extract text from image. Please try with a clearer image."
        )
    
    # Process with LLM
    result, raw_llm_response = extract_info_with_llm(text)
    
    # Add raw text to result
    result["raw_text"] = text
    result["llm_response"] = raw_llm_response
    
    return result

###################
# Code Endpoints  #
###################

@app.post("/code/analyze", response_model=CodeAnalysisResponse, tags=["Code Analysis"])
async def analyze_code(
    file: UploadFile = File(...),
):
    """
    Analyze code to determine if it's AI-generated
    
    - **file**: Python (.py) or Jupyter notebook (.ipynb) file
    
    Returns an AI probability score and analysis.
    """
    file_name = file.filename
    
    # Check if file type is supported
    if not file_name.lower().endswith(('.py', '.ipynb')):
        raise HTTPException(
            status_code=400, 
            detail="Unsupported file format. Only Python (.py) and Jupyter notebook (.ipynb) files are allowed."
        )
    
    # Read file content
    file_content = await file.read()
    
    # Process based on file type
    file_extension = file_name.split('.')[-1].lower()
    
    if file_extension == 'py':
        code_content = read_python_file_content(file_content)
        file_type = "Python"
    elif file_extension == 'ipynb':
        try:
            code_content = extract_code_from_notebook(file_content)
            file_type = "Jupyter Notebook"
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error processing notebook: {str(e)}"
            )
    else:
        # This should never happen due to the earlier check
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    # Analyze the code
    try:
        analysis = analyze_code_with_groq(code_content)
        
        # Add file information to response
        analysis['file_name'] = file_name
        analysis['file_type'] = file_type
        
        return analysis
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing code: {str(e)}"
        )

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("final_api:app", host="127.0.0.1", port=8080, reload=True)