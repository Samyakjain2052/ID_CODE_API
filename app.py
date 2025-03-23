from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import easyocr
from PIL import Image
import io
import cv2
import numpy as np
import json
import re
import requests
import logging
from datetime import datetime
from typing import Optional, Dict, Any
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api.log")
    ]
)
logger = logging.getLogger("api")

# Get API key from environment variable or use fallback
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

# Initialize EasyOCR reader (only once to avoid reloading the model)
# Note: This will download models on first run if they're not already downloaded
try:
    logger.info("Initializing EasyOCR reader...")
    reader = easyocr.Reader(['en'])
    logger.info("EasyOCR reader initialized successfully")
except Exception as e:
    logger.error(f"Error initializing EasyOCR: {str(e)}")
    reader = None

# Initialize FastAPI app
app = FastAPI(
    title="ID Card Analyzer API",
    description="API for extracting information from ID cards and calculating age",
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

# Function to calculate age from date of birth
def calculate_age(dob_str):
    """Calculate age from a date of birth string in various formats"""
    try:
        # Try different date formats
        for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%m-%d-%Y'):
            try:
                dob = datetime.strptime(dob_str, fmt)
                today = datetime.today()
                
                # Calculate age
                age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                return age
            except ValueError:
                continue
                
        # If none of the formats worked
        return None
    except Exception as e:
        logger.error(f"Error calculating age: {str(e)}")
        return None

# Function to extract date of birth using regex
def extract_dob(text):
    """Extract date of birth from text using regular expressions"""
    try:
        # Common date patterns
        date_patterns = [
            r'(?:Date of Birth|DOB|Birth Date)[:\s]+(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
            r'(?:Date of Birth|DOB|Birth Date)[:\s]+(\d{2,4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})',
            r'(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})',  # DD/MM/YYYY or MM/DD/YYYY
            r'(\d{4}[\/\-\.]\d{2}[\/\-\.]\d{2})'   # YYYY/MM/DD
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    except Exception as e:
        logger.error(f"Error extracting DOB: {str(e)}")
        return None

# Function to process ID card with Groq LLM
def process_with_groq(text):
    """Process extracted text with Groq LLM to extract structured information"""
    try:
        api_url = "https://api.groq.com/openai/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Improved prompt with explicit formatting instructions
        prompt = f"""
        Extract the full name and date of birth from the following ID card text.
        
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
        
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        llm_response = result["choices"][0]["message"]["content"]
        
        # Parse the JSON response
        extracted_data = json.loads(llm_response)
        return extracted_data, llm_response
        
    except Exception as e:
        logger.error(f"Error processing with Groq: {str(e)}")
        
        # Fall back to regex extraction if Groq fails
        name = None  # We don't have a simple regex for names
        dob = extract_dob(text)
        
        return {"name": name, "dob": dob}, f"Error with Groq API: {str(e)}"

# Health check endpoint
@app.get("/")
def read_root():
    """API health check endpoint"""
    return {
        "status": "online", 
        "version": "1.0.0",
        "endpoints": {
            "/": "Health check",
            "/id/extract": "Extract information from ID card"
        }
    }

# Main ID extraction endpoint
@app.post("/id/extract")
async def extract_id_data(file: UploadFile = File(...)):
    """
    Extract information from an ID card image
    
    - **file**: Image file of an ID card
    
    Returns name, date of birth, age, and raw extracted text.
    """
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Processing new ID extraction request")
    
    # Validate file
    if not file.content_type.startswith('image/'):
        logger.warning(f"[{request_id}] Invalid file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="Only image files are allowed")
    
    # Check if EasyOCR is initialized
    if reader is None:
        logger.error(f"[{request_id}] EasyOCR reader not initialized")
        raise HTTPException(status_code=500, detail="OCR service not initialized")
    
    # Read file
    try:
        contents = await file.read()
    except Exception as e:
        logger.error(f"[{request_id}] Error reading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")
    
    # Process the image with EasyOCR
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Preprocess image for better OCR results
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to remove noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Apply adaptive threshold to handle different lighting conditions
        thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Use EasyOCR to extract text from both original and preprocessed images
        # Try preprocessed image first
        logger.info(f"[{request_id}] Extracting text with EasyOCR from preprocessed image")
        results_thresh = reader.readtext(thresh)
        
        # If not enough text was found, try the original image
        if len(results_thresh) < 3:  # Arbitrary threshold
            logger.info(f"[{request_id}] Not enough text found, trying original image")
            results = reader.readtext(img)
            
            # Use whichever gave more results
            if len(results) > len(results_thresh):
                text_results = results
            else:
                text_results = results_thresh
        else:
            text_results = results_thresh
        
        # Combine all detected text
        text = ' '.join([result[1] for result in text_results])
        
        if not text:
            logger.warning(f"[{request_id}] No text detected in image")
            return JSONResponse(
                status_code=422,
                content={
                    "error": "No text detected in the image. Please upload a clearer image."
                }
            )
        
        logger.info(f"[{request_id}] Extracted text: {text[:100]}...")
        
        # Process with Groq LLM
        logger.info(f"[{request_id}] Processing with Groq LLM")
        extracted_info, llm_response = process_with_groq(text)
        
        # Extract and clean DOB
        dob = extracted_info.get("dob")
        
        # Calculate age if DOB is available
        age = None
        if dob:
            age = calculate_age(dob)
            
        # Prepare the response
        response_data = {
            "request_id": request_id,
            "name": extracted_info.get("name"),
            "dob": dob,
            "age": age,
            "is_minor": age is not None and age < 18,
            "raw_text": text,
            "llm_processed": llm_response
        }
        
        logger.info(f"[{request_id}] Successfully processed ID card: Name={extracted_info.get('name')}, DOB={dob}, Age={age}")
        return response_data
        
    except Exception as e:
        logger.error(f"[{request_id}] Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# Run the app
if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8080))
    
    # Get host from environment variable or use default
    host = os.environ.get("HOST", "127.0.0.1")  # Use 0.0.0.0 for EC2
    
    logger.info(f"Starting ID Card Analyzer API on {host}:{port}")
    
    uvicorn.run(
        "app:app", 
        host=host, 
        port=port,
        log_level="info",
        access_log=True,
        workers=1  # For development; increase for production
    )