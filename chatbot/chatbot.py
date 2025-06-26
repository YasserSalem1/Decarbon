import os
import re
from typing import List, Dict, Optional
from fastapi import FastAPI, UploadFile, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PyPDF2 import PdfReader
from pydantic import BaseModel
import requests
import torch
from PIL import Image
import open_clip
from sentence_transformers import SentenceTransformer
import openai
import pinecone
from langdetect import detect
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import json
from pathlib import Path
import google.generativeai as genai
from google.generativeai.types import Tool
from google.generativeai.types.generation_types import GenerationConfig
import asyncio
import logging
from livekit import rtc, api
from livekit.rtc.room import Room
import wave
import io
import speech_recognition as sr
load_dotenv()
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import base64

# Import the API functions
from api import get_energy_data
from machine_api import get_eia_industrial_data
import json

# Tracks user sessions and last image metadata shown

# nltk.download('punkt')  #
openai.api_key = os.getenv("OPENAI_API_KEY")
# Initialize Gemini
GOOGLE_API_KEY=os.getenv("GOOGLE_CLOUD_KEY")
GOOGLE_CSE_ID=os.getenv("GOOGLE_CSE_ID")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel(
    model_name='gemini-2.0-flash-001'  # Removed tools configuration
)
# Image handling
IMAGES_DIR = Path("images")
IMAGES_METADATA_FILE = IMAGES_DIR / "metadata.json"

# Initialize CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
clip_model = clip_model.to(device)

class ImageMetadata(BaseModel):
    id: str
    filename: str
    embedding: Optional[List[float]] = None  # Store CLIP embedding

class ImageStore:
    def __init__(self):
        self.images_dir = IMAGES_DIR
        self.metadata_file = IMAGES_METADATA_FILE
        self.images_dir.mkdir(exist_ok=True)
        self.metadata: Dict[str, ImageMetadata] = self._load_metadata()
        # Precompute embeddings for all images
        self._precompute_embeddings()

    def _load_metadata(self) -> Dict[str, ImageMetadata]:
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
                return {k: ImageMetadata(**v) for k, v in data.items()}
        return {}

    def _save_metadata(self):
        with open(self.metadata_file, 'w') as f:
            json.dump({k: v.model_dump() for k, v in self.metadata.items()}, f, indent=2)

    def _precompute_embeddings(self):
        """Precompute CLIP embeddings for all images"""
        for image_id, metadata in self.metadata.items():
            if metadata.embedding is None:
                image_path = self.images_dir / metadata.filename
                if image_path.exists():
                    try:
                        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                        with torch.no_grad():
                            image_features = clip_model.encode_image(image)
                            metadata.embedding = image_features.cpu().numpy().tolist()[0]
                    except Exception as e:
                        print(f"Error computing embedding for {image_id}: {e}")
        self._save_metadata()

    def add_image(self, file: UploadFile, metadata: ImageMetadata):
        if len(self.metadata) >= 6:
            raise HTTPException(status_code=400, detail="Maximum of 6 images allowed")
        
        # Save image file
        file_path = self.images_dir / file.filename
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        
        # Compute CLIP embedding
        try:
            image = preprocess(Image.open(file_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = clip_model.encode_image(image)
                metadata.embedding = image_features.cpu().numpy().tolist()[0]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
        
        # Save metadata with embedding
        self.metadata[metadata.id] = metadata
        self._save_metadata()

    def get_image(self, image_id: str) -> Optional[tuple[Path, ImageMetadata]]:
        if image_id not in self.metadata:
            return None
        metadata = self.metadata[image_id]
        image_path = self.images_dir / metadata.filename
        if not image_path.exists():
            return None
        return image_path, metadata

    def get_relevant_images(self, query: str, top_k: int = 1) -> List[tuple[Path, ImageMetadata]]:
        # Encode the text query using CLIP
        text_tokens = open_clip.tokenize([query]).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text_tokens)
            text_features = text_features.cpu().numpy()[0]

        # Calculate similarities with all images
        image_scores = []
        for metadata in self.metadata.values():
            if metadata.embedding is None:
                continue
            similarity = cosine_similarity([text_features], [metadata.embedding])[0][0]
            image_scores.append((metadata, similarity))
        
        # Sort by similarity score and get top k
        image_scores.sort(key=lambda x: x[1], reverse=True)
        relevant_images = []
        
        for metadata, score in image_scores[:top_k]:
            image_path = self.images_dir / metadata.filename
            if image_path.exists():
                relevant_images.append((image_path, metadata))
        
        return relevant_images

    def get_image_by_filename(self, filename: str) -> Optional[tuple[Path, ImageMetadata]]:
        """Get image by filename"""
        image_path = self.images_dir / filename
        if not image_path.exists():
            return None
        # Find metadata by filename
        for metadata in self.metadata.values():
            if metadata.filename == filename:
                return image_path, metadata
        return None
user_last_image: Dict[str, ImageMetadata] = defaultdict(lambda: None)
# Initialize image store
image_store = ImageStore()

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "quickstart"
if index_name not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=3072,  # for OpenAI `text-embedding-ada-002`
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws", region="us-east-1"  # or your preferred region
        )
    )
# Connect to an existing index
index = pc.Index(index_name)
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Explicitly list allowed methods
    allow_headers=["*"],
    expose_headers=["*"]
)

# LiveKit configuration
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")

# Initialize LiveKit room and API
room = None
lkapi = None

async def initialize_livekit():
    global room, lkapi
    try:
        if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
            raise ValueError("LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set in environment variables")
            
        # Initialize LiveKit API with credentials
        api_url = LIVEKIT_URL.replace("ws://", "http://").replace("wss://", "https://")
        lkapi = api.LiveKitAPI(
            api_url,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET
        )
        
        # Create room if it doesn't exist
        try:
            await lkapi.room.create_room(
                api.CreateRoomRequest(name="voice_room")
            )
        except Exception as e:
            logging.info(f"Room might already exist: {str(e)}")
        
        # Generate access token
        token = api.AccessToken(
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET
        ).with_identity("server") \
         .with_name("Server") \
         .with_grants(api.VideoGrants(
            room_join=True,
            room="voice_room"
        )).to_jwt()
        
        # Initialize and connect room
        room = Room()
        
        # Set up event handlers
        @room.on("participant_connected")
        def on_participant_connected(participant: rtc.RemoteParticipant):
            logging.info(f"Participant connected: {participant.sid} {participant.identity}")
        
        @room.on("track_subscribed")
        def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
            logging.info(f"Track subscribed: {publication.sid}")
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                # Handle audio track subscription
                logging.info(f"Audio track subscribed from {participant.identity}")
        
        # Connect to room
        await room.connect(LIVEKIT_URL, token)
        logging.info(f"Connected to room {room.name}")
        
        return room
    except Exception as e:
        logging.error(f"Error initializing LiveKit: {str(e)}")
        raise

# Create startup event handler
@app.on_event("startup")
async def startup_event():
    await initialize_livekit()

# Create shutdown event handler
@app.on_event("shutdown")
async def shutdown_event():
    global lkapi
    if lkapi:
        await lkapi.aclose()

class QueryRequest(BaseModel):
    question: str
    top_k: int = 30
    include_images: bool = True

class VoiceQueryRequest(BaseModel):
    audio_data: bytes
    sample_rate: int = 16000
    sample_width: int = 2

class UnifiedQueryRequest(BaseModel):
    query: str
    include_images: bool = True

def get_coordinates_for_location(location: str):
    """Get coordinates for a location using a geocoding service"""
    # Common city coordinates (can be expanded or replaced with real geocoding API)
    city_coords = {
        "berlin": (52.5200, 13.4050),
        "new york": (40.7128, -74.0060),
        "london": (51.5074, -0.1278),
        "paris": (48.8566, 2.3522),
        "tokyo": (35.6762, 139.6503),
        "moscow": (55.7558, 37.6176),
        "sydney": (-33.8688, 151.2093),
        "los angeles": (34.0522, -118.2437),
        "chicago": (41.8781, -87.6298),
        "madrid": (40.4168, -3.7038),
        "rome": (41.9028, 12.4964),
        "barcelona": (41.3851, 2.1734),
        "amsterdam": (52.3676, 4.9041),
        "vienna": (48.2082, 16.3738),
        "zurich": (47.3769, 8.5417)
    }
    
    location_lower = location.lower()
    for city, coords in city_coords.items():
        if city in location_lower:
            return coords
    
    # Default to New York if location not found
    return (40.7128, -74.0060)

def extract_energy_parameters(query: str):
    """Use LLM to extract location, time period, and days from user query"""
    extraction_prompt = f"""
Extract the following parameters from this energy/weather query:

1. LOCATION: Extract city/country/region mentioned (e.g., "Berlin", "New York", "London", etc.)
2. DAYS: Extract number of days requested (1-7, default to 3 if not specified)
3. SPECIFIC_QUESTION: What specific aspect they're asking about (e.g., "highest solar day", "wind patterns", "energy forecast")

Query: "{query}"

Respond ONLY with a JSON object in this exact format:
{{
    "location": "extracted_location_or_null",
    "days": number_between_1_and_7,
    "specific_question": "what_they_want_to_know"
}}
"""
    
    try:
        response = model.generate_content(extraction_prompt)
        response_text = response.text.strip()
        
        # Clean up the response to extract JSON
        if response_text.startswith('```json'):
            response_text = response_text.replace('```json', '').replace('```', '').strip()
        elif response_text.startswith('```'):
            response_text = response_text.replace('```', '').strip()
            
        # Parse JSON response
        result = json.loads(response_text)
        
        # Validate and set defaults
        if not result.get("location"):
            result["location"] = "New York"  # Default location
        
        days = result.get("days", 3)
        if not isinstance(days, int) or days < 1 or days > 7:
            result["days"] = 3
            
        if not result.get("specific_question"):
            result["specific_question"] = "energy forecast"
            
        return result
        
    except Exception as e:
        print(f"Error in parameter extraction: {e}")
        return {
            "location": "New York",
            "days": 3,
            "specific_question": "energy forecast"
        }

def analyze_energy_data_comprehensive(data_df, location: str, days: int, specific_question: str, original_query: str):
    """Comprehensive analysis using data_df and actual verified metrics in one step"""
    if data_df is None or data_df.empty:
        return "Sorry, I couldn't fetch energy data for that location."
    
    try:
        # Calculate comprehensive verified metrics directly from data_df
        actual_metrics = calculate_accuracy_metrics(data_df)
        
        if not actual_metrics:
            # Fallback to basic summary if metrics calculation fails
            avg_solar = data_df['solar'].mean() if 'solar' in data_df.columns else 0
            avg_wind = data_df['wind_speed'].mean() if 'wind_speed' in data_df.columns else 0
            return f"Energy data for {location}: Average solar radiation: {avg_solar:.1f} W/m², Average wind speed: {avg_wind:.1f} m/s over {days} days."
        
        # Create focused analysis prompt with verified metrics
        comprehensive_prompt = f"""
You are a renewable energy analyst. Provide a concise, data-driven analysis that directly addresses the user's query using the verified metrics.

QUERY: "{original_query}"
LOCATION: {location}
PERIOD: {days} days

VERIFIED DATA:
{json.dumps(actual_metrics, indent=2)}

Provide a focused analysis that includes:

**DIRECT ANSWER TO QUERY:**
Address the specific question using the actual data metrics provided.

**KEY DATA INSIGHTS:**
- Solar: Average, peak values, and optimal production hours
- Wind: Average, peak values, and optimal production hours  
- Peak Hours: Exact times when solar and wind are highest
- Production Patterns: When to expect maximum renewable energy

**ACTIONABLE RECOMMENDATIONS:**
Based on the verified data, provide 3-5 specific recommendations for:
- Optimal scheduling of energy-intensive operations
- Load management strategies
- Cost reduction opportunities

**QUANTIFIED IMPACT:**
- Potential energy cost savings
- Grid dependency reduction
- Carbon footprint reduction

Keep the response concise and focused on the actual data provided. Use specific numbers from the verified metrics. Avoid generic advice - base everything on the real data patterns shown.
"""

        response = model.generate_content(comprehensive_prompt)
        return response.text if hasattr(response, 'text') else str(response)
        
    except Exception as e:
        print(f"Error in comprehensive analysis: {e}")
        # Fallback to basic summary
        avg_solar = data_df['solar'].mean() if 'solar' in data_df.columns else 0
        avg_wind = data_df['wind_speed'].mean() if 'wind_speed' in data_df.columns else 0
        return f"Energy data for {location}: Average solar radiation: {avg_solar:.1f} W/m², Average wind speed: {avg_wind:.1f} m/s over {days} days."

def get_initial_llm_analysis(data_df, location: str, days: int, specific_question: str, original_query: str):
    """Get initial LLM analysis of the energy data"""
    # Prepare data summary for LLM
    data_summary = {
        "total_hours": len(data_df),
        "solar_stats": {
            "min": float(data_df['solar'].min()),
            "max": float(data_df['solar'].max()),
            "avg": float(data_df['solar'].mean()),
            "peak_hour": str(data_df.loc[data_df['solar'].idxmax()].name) if len(data_df) > 0 else "N/A"
        },
        "wind_stats": {
            "min": float(data_df['wind_speed'].min()),
            "max": float(data_df['wind_speed'].max()),
            "avg": float(data_df['wind_speed'].mean()),
            "peak_hour": str(data_df.loc[data_df['wind_speed'].idxmax()].name) if len(data_df) > 0 else "N/A"
        }
    }
    
    # Group by day to find daily patterns
    daily_data = []
    data_df_reset = data_df.reset_index()
    if 'timestamp' in data_df_reset.columns:
        data_df_reset['date'] = pd.to_datetime(data_df_reset['timestamp']).dt.date
        for date, group in data_df_reset.groupby('date'):
            daily_data.append({
                "date": str(date),
                "avg_solar": float(group['solar'].mean()),
                "peak_solar": float(group['solar'].max()),
                "avg_wind": float(group['wind_speed'].mean()),
                "peak_wind": float(group['wind_speed'].max())
            })

    analysis_prompt = f"""
You are an energy data analyst. Analyze this energy data for {location} over {days} days and answer the user's specific question.

Original question: "{original_query}"
Specific aspect they want to know: "{specific_question}"

Data Summary:
- Total data points: {data_summary['total_hours']} hours
- Solar radiation (W/m²): Min: {data_summary['solar_stats']['min']:.1f}, Max: {data_summary['solar_stats']['max']:.1f}, Avg: {data_summary['solar_stats']['avg']:.1f}
- Peak solar time: {data_summary['solar_stats']['peak_hour']}
- Wind speed (m/s): Min: {data_summary['wind_stats']['min']:.1f}, Max: {data_summary['wind_stats']['max']:.1f}, Avg: {data_summary['wind_stats']['avg']:.1f}
- Peak wind time: {data_summary['wind_stats']['peak_hour']}

Daily breakdown:
{json.dumps(daily_data, indent=2)}

Please provide a comprehensive answer that:
1. Directly answers their specific question
2. Provides relevant context and insights
3. Mentions specific times/days when relevant
4. Gives practical implications
5. Uses clear, conversational language

Focus on what they specifically asked about (e.g., if they asked for "highest solar day", identify which day and when).
"""

    response = model.generate_content(analysis_prompt)
    return response.text if hasattr(response, 'text') else str(response)

def get_enhanced_llm_analysis(initial_analysis: str, actual_metrics: dict, location: str, days: int, specific_question: str, original_query: str):
    """Get enhanced LLM analysis that incorporates accuracy verification"""
    
    enhancement_prompt = f"""
You are an expert energy data analyst. with each query you provide a comprehensive analysis, insights , recommendations and actions and environmental information based on verified metrics. to user in factories and companies that make use of renwable energies

ORIGINAL QUERY: "{original_query}"
SPECIFIC QUESTION: "{specific_question}"
LOCATION: {location}
DAYS: {days}

PREVIOUS ANALYSIS:
{initial_analysis}

ACTUAL VERIFIED METRICS:
{json.dumps(actual_metrics, indent=2)}



Be more precise and accurate this time, using the verified data metrics.
"""
    
    response = model.generate_content(enhancement_prompt)
    return response.text if hasattr(response, 'text') else str(response)

def calculate_accuracy_metrics(data_df):
    """Calculate comprehensive accuracy metrics from the actual data for verification"""
    if data_df is None or data_df.empty:
        return None
    
    try:
        import pandas as pd
        import numpy as np
        
        # Reset index to work with timestamp
        if isinstance(data_df.index, pd.DatetimeIndex):
            df_reset = data_df.reset_index()
            df_reset['hour'] = df_reset['timestamp'].dt.hour
            df_reset['date'] = df_reset['timestamp'].dt.date
        else:
            df_reset = data_df.copy()
            # Try to extract hour from timestamp column if it exists
            if 'timestamp' in df_reset.columns:
                df_reset['timestamp'] = pd.to_datetime(df_reset['timestamp'])
                df_reset['hour'] = df_reset['timestamp'].dt.hour
                df_reset['date'] = df_reset['timestamp'].dt.date
        
        metrics = {
            'data_points': len(df_reset),
            'time_range': {},
            'solar': {},
            'wind': {},
            'hourly_patterns': {},
            'daily_breakdown': []
        }
        
        # Time range information
        if 'timestamp' in df_reset.columns:
            metrics['time_range'] = {
                'start': str(df_reset['timestamp'].min()),
                'end': str(df_reset['timestamp'].max()),
                'total_hours': len(df_reset)
            }
        
        # Handle different data types - industrial vs energy data
        data_type = "unknown"
        if 'solar' in df_reset.columns and 'wind_speed' in df_reset.columns:
            data_type = "energy"
        elif any(col in df_reset.columns for col in ['value', 'demand', 'load', 'consumption']):
            data_type = "industrial"
        
        # Solar metrics with detailed analysis
        if 'solar' in df_reset.columns:
            solar_data = df_reset['solar']
            metrics['solar'] = {
                'avg': round(float(solar_data.mean()), 3),
                'max': round(float(solar_data.max()), 1),
                'min': round(float(solar_data.min()), 1),
                'std': round(float(solar_data.std()), 2),
                'peak_hour': None,
                'peak_hour_formatted': None,
                'zero_hours': int((solar_data == 0).sum()),
                'above_avg_hours': int((solar_data > solar_data.mean()).sum())
            }
            
            # Calculate peak hour
            if 'hour' in df_reset.columns:
                hourly_solar = df_reset.groupby('hour')['solar'].mean()
                peak_hour = int(hourly_solar.idxmax())
                metrics['solar']['peak_hour'] = peak_hour
                metrics['solar']['peak_hour_formatted'] = f"{peak_hour:02d}:00"
                
                # Store hourly patterns
                metrics['hourly_patterns']['solar'] = {
                    str(hour): round(float(avg), 1) 
                    for hour, avg in hourly_solar.items()
                }
        
        # Wind metrics with detailed analysis
        if 'wind_speed' in df_reset.columns:
            wind_data = df_reset['wind_speed']
            metrics['wind'] = {
                'avg': round(float(wind_data.mean()), 3),
                'max': round(float(wind_data.max()), 1),
                'min': round(float(wind_data.min()), 1),
                'std': round(float(wind_data.std()), 2),
                'peak_hour': None,
                'peak_hour_formatted': None,
                'calm_hours': int((wind_data < 1).sum()),
                'high_wind_hours': int((wind_data > 10).sum())
            }
            
            # Calculate peak hour
            if 'hour' in df_reset.columns:
                hourly_wind = df_reset.groupby('hour')['wind_speed'].mean()
                peak_hour = int(hourly_wind.idxmax())
                metrics['wind']['peak_hour'] = peak_hour
                metrics['wind']['peak_hour_formatted'] = f"{peak_hour:02d}:00"
                
                # Store hourly patterns
                metrics['hourly_patterns']['wind'] = {
                    str(hour): round(float(avg), 1) 
                    for hour, avg in hourly_wind.items()
                }
        
        # Daily breakdown
        if 'date' in df_reset.columns and len(df_reset['date'].unique()) > 1:
            for date, group in df_reset.groupby('date'):
                day_data = {
                    'date': str(date),
                    'hours': len(group)
                }
                
                if 'solar' in group.columns:
                    day_data['solar'] = {
                        'avg': round(float(group['solar'].mean()), 1),
                        'max': round(float(group['solar'].max()), 1),
                        'peak_hour': int(group.loc[group['solar'].idxmax(), 'hour']) if len(group) > 0 else None
                    }
                
                if 'wind_speed' in group.columns:
                    day_data['wind'] = {
                        'avg': round(float(group['wind_speed'].mean()), 1),
                        'max': round(float(group['wind_speed'].max()), 1),
                        'peak_hour': int(group.loc[group['wind_speed'].idxmax(), 'hour']) if len(group) > 0 else None
                    }
                
                metrics['daily_breakdown'].append(day_data)
        
        # Industrial data analysis (for EIA electricity demand data)
        if data_type == "industrial":
            # Look for common industrial data columns
            value_col = None
            for col in ['value', 'demand', 'load', 'consumption', 'electricity']:
                if col in df_reset.columns:
                    value_col = col
                    break
            
            if value_col:
                industrial_data = df_reset[value_col]
                metrics['industrial'] = {
                    'avg_demand': round(float(industrial_data.mean()), 2),
                    'peak_demand': round(float(industrial_data.max()), 2),
                    'min_demand': round(float(industrial_data.min()), 2),
                    'demand_variability': round(float(industrial_data.std()), 2),
                    'peak_hour': None,
                    'peak_hour_formatted': None,
                    'low_demand_hours': int((industrial_data < industrial_data.quantile(0.25)).sum()),
                    'high_demand_hours': int((industrial_data > industrial_data.quantile(0.75)).sum())
                }
                
                # Calculate peak demand hour
                if 'hour' in df_reset.columns:
                    hourly_demand = df_reset.groupby('hour')[value_col].mean()
                    peak_hour = int(hourly_demand.idxmax())
                    metrics['industrial']['peak_hour'] = peak_hour
                    metrics['industrial']['peak_hour_formatted'] = f"{peak_hour:02d}:00"
                    
                    # Store hourly patterns
                    metrics['hourly_patterns']['industrial'] = {
                        str(hour): round(float(avg), 2) 
                        for hour, avg in hourly_demand.items()
                    }
        
        print(f"Calculated comprehensive metrics: {len(metrics)} categories, data type: {data_type}")
        return metrics
        
    except Exception as e:
        print(f"Error calculating accuracy metrics: {e}")
        import traceback
        traceback.print_exc()
        return None

def query_energy_data(latitude: float = 40.71, longitude: float = -74.01, days: int = 3):
    """Function to query energy data (solar/wind conditions)"""
    print(f"called function query energy data with parameters latitude: {latitude}, longitude: {longitude}, days: {days}")
    
    try:
        # Get energy data using the API
        energy_df = get_energy_data(latitude, longitude, days)
        
        if energy_df is not None and not energy_df.empty:
            # Save the complete data to CSV
            try:
                energy_df.to_csv("current_data.csv")
                print(f"Data saved to current_data.csv ({len(energy_df)} records)")
            except Exception as e:
                print(f"Error saving to CSV: {e}")
            
            # Format the response
            latest_data = energy_df.head(24)  # Get first 24 hours for response
            avg_solar = latest_data['solar'].mean()
            avg_wind = latest_data['wind_speed'].mean()
            
            summary = f"Energy forecast for coordinates ({latitude}, {longitude}) for {days} days:\n"
            summary += f"Average Solar Radiation: {avg_solar:.2f} W/m²\n"
            summary += f"Average Wind Speed: {avg_wind:.2f} m/s\n"
            summary += f"Data points: {len(energy_df)} hours\n"
            summary += f"Complete data saved to current_data.csv"
            
            return {
                "summary": summary,
                "data": latest_data.to_dict('records'),
                "full_data_saved": True,
                "csv_file": "current_data.csv",
                "total_records": len(energy_df),
                "status": "success"
            }
        else:
            return {
                "summary": "Failed to fetch energy data from all APIs",
                "data": None,
                "status": "error"
            }
            
    except Exception as e:
        print(f"Error in query_energy_data: {str(e)}")
        return {
            "summary": f"Error fetching energy data: {str(e)}",
            "data": None,
            "status": "error"
        }

def query_industrial_data(api_key: str = "4XVxrVQ7H3ddVOvZKhYuggmOf64DCT9JvuKlZ0Sl"):
    """Function to query industrial electricity data"""
    print(f"called function query industrial data with API key")
    
    try:
        # Get industrial data using the EIA API
        industrial_df = get_eia_industrial_data(api_key)
        
        if not industrial_df.empty:
            # Save the complete data to CSV
            try:
                industrial_df.to_csv("current_data.csv", index=False)
                print(f"Industrial data saved to current_data.csv ({len(industrial_df)} records)")
            except Exception as e:
                print(f"Error saving industrial data to CSV: {e}")
            
            # Process and summarize the data
            recent_data = industrial_df.head(10)  # Get recent data
            
            summary = f"Industrial electricity demand data:\n"
            summary += f"Total records: {len(industrial_df)}\n"
            summary += f"Recent demand values available\n"
            summary += f"Grid operators: PJM, MISO\n"
            summary += f"Complete data saved to current_data.csv"
            
            return {
                "summary": summary,
                "data": recent_data.to_dict('records'),
                "full_data_saved": True,
                "csv_file": "current_data.csv",
                "total_records": len(industrial_df),
                "status": "success"
            }
        else:
            return {
                "summary": "Failed to fetch industrial electricity data",
                "data": None,
                "status": "error"
            }
            
    except Exception as e:
        print(f"Error in query_industrial_data: {str(e)}")
        return {
            "summary": f"Error fetching industrial data: {str(e)}",
            "data": None,
            "status": "error"
        }

def optimize():
    """Function to call optimization"""
    print("called optimize")
    return "Optimization function called"

def visualize():
    """Function to show visualization"""
    print("called visualize")
    return "Visualization function called"

def classify_query(query: str) -> str:
    """
    Classify user query into one of the following categories using keyword matching:
    1. regular_conversation - general chatbot conversation
    2. machine_info - asking about machine-specific information (RAG)
    3. weather - asking about solar wind conditions
    4. optimize - requesting optimization
    5. visualize - requesting visualization
    """
    query_lower = query.lower()
    
    # Weather/Solar wind related keywords
    weather_keywords = ["weather", "solar wind", "space weather", "solar", "wind", "conditions", "forecast", "storm", "magnetic", "aurora"]
    
    # Optimization keywords
    optimize_keywords = ["optimize", "optimization", "improve", "enhance", "efficient", "performance", "better"]
    
    # Visualization keywords
    viz_keywords = ["visualize", "visualization", "plot", "chart", "graph", "show", "display", "visual", "diagram"]
    
    # Machine/technical info keywords (for RAG)
    machine_keywords = ["machine", "system", "technical", "specification", "manual", "document", "how to", "configuration", "setup", "installation"]
    
    # Industrial/grid data keywords
    industrial_keywords = ["industrial", "electricity", "grid", "power", "demand", "utility", "factory", "manufacturing", "eia", "electrical"]
    
    # Check for specific function calls - prioritize weather/solar/wind queries
    print(f"Classifying query: '{query_lower}'")
    weather_matches = [kw for kw in weather_keywords if kw in query_lower]
    industrial_matches = [kw for kw in industrial_keywords if kw in query_lower]
    print(f"Weather keywords found: {weather_matches}")
    print(f"Industrial keywords found: {industrial_matches}")
    
    if any(keyword in query_lower for keyword in weather_keywords):
        print("Classified as: weather")
        return "weather"
    elif any(keyword in query_lower for keyword in optimize_keywords):
        return "optimize"
    elif any(keyword in query_lower for keyword in viz_keywords):
        return "visualize"
    elif any(keyword in query_lower for keyword in industrial_keywords):
        # Only classify as industrial if it doesn't contain weather keywords
        if not any(keyword in query_lower for keyword in weather_keywords):
            return "industrial"
        else:
            return "weather"  # Prefer weather classification if both are present
    elif any(keyword in query_lower for keyword in machine_keywords):
        return "machine_info"
    else:
        return "regular_conversation"

def clean_text(text: str) -> str:
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    sentences = [s.strip() for s in sent_tokenize(text) if len(s.split()) > 4]

    sentences = [s for s in sentences if detect(s) == 'en']
    return '. '.join(sentences)

import numpy as np

def chunk_text(text: str, breakpoint_percentile=80, max_sentences=20) -> List[str]:
    sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
    if len(sentences) <= 1:
        return [text]

    # Use OpenAI embeddings instead of SentenceTransformer
    embeddings = [get_embedding(sentence) for sentence in sentences]

    similarities = cosine_similarity(embeddings[:-1], embeddings[1:])
    distances = 1 - np.diagonal(similarities)

    cutoff = float(np.percentile(distances, breakpoint_percentile))

    chunks, current_chunk = [], [sentences[0]]
    for i, dist in enumerate(distances):
        if dist < cutoff and len(current_chunk) < max_sentences:
            current_chunk.append(sentences[i+1])
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentences[i+1]]
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks
def get_embedding(text: str) -> List[float]:
    
    response = openai.Embedding.create(input=[text], model="text-embedding-3-large")

    return response['data'][0]['embedding']

def upsert_chunks(chunks: List[dict], source_id: str, batch_size=50):
    vectors = []
    for i, chunk_info in enumerate(chunks):
        chunk = chunk_info["text"]
        embedding = get_embedding(chunk)
        vectors.append({
            "id": f"{source_id}_{chunk_info['page']}_{chunk_info['index_in_page']}",
            "values": embedding,
            "metadata": {
                "text": chunk,
                "source": source_id,
                "page": chunk_info["page"],
                "index_in_page": chunk_info["index_in_page"],
                "total_in_page": chunk_info["total_in_page"]
            }
        })

    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors[i:i+batch_size])

async def image_follow_up(session_id: str = Form(...), question: str = Form(...)):
    image_meta = user_last_image.get(session_id)
    if not image_meta:
        raise HTTPException(status_code=400, detail="No image context found for this session.")

    image_path = IMAGES_DIR / image_meta.filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found.")

    # Load and prepare the image
    try:
        img = Image.open(image_path)
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize if too large (Gemini has size limits)
        max_size = 1024
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr = img_byte_arr.getvalue()
        
        # Use Gemini to answer the question based on the image
        response = await asyncio.to_thread(
            model.generate_content,
            contents=[
                {"role": "user", "parts": [
                    {"text": question},
                    {"inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64.b64encode(img_byte_arr).decode('utf-8')
                    }}
                ]}
            ],
            generation_config=GenerationConfig(temperature=0.7),
        )
        return {"response": response.text}
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

def hybrid_search(query: str, top_z=1) -> List[dict]:
    # 1. Embed the query
    top_k=80
    embedding = get_embedding(query)

    # 2. Initial dense search
    results = index.query(vector=embedding, top_k=top_k , include_metadata=True)

    # 3. Define keyword match score
    def keyword_score(text: str, query: str):
        q_tokens = set(re.findall(r'\w+', query.lower()))
        t_tokens = set(re.findall(r'\w+', text.lower()))
        return len(q_tokens & t_tokens)

    # 4. Rerank using hybrid score
    reranked = sorted(
        results["matches"],
        key=lambda r: 0.6 * r["score"] + 0.4 * keyword_score(r["metadata"]["text"], query),
        reverse=True
    )

    top_results = reranked[:top_k]

    # 5. Collect all source chunks ONCE to enrich neighbors
    source_chunks_map = {}

    for r in top_results:
        source = r["metadata"]["source"]
        if source in source_chunks_map:
            continue

        # Retrieve all chunks from this source
        all_results = index.query(
            vector=embedding,  # still needs a vector for now
            filter={"source": {"$eq": source}},
            top_k=1000,
            include_metadata=True
        )

        chunks_by_page = {}
        for item in all_results["matches"]:
            meta = item["metadata"]
            key = (meta["page"], meta["index_in_page"])
            chunks_by_page[key] = meta
        source_chunks_map[source] = chunks_by_page

    # 6. Enrich each chunk with neighbors
    enriched = []
    for r in top_results:
        meta = r["metadata"]
        pg = meta["page"]
        idx = meta["index_in_page"]
        src = meta["source"]
        hybrid_score = 0.6 * r["score"] + 0.4 * keyword_score(meta["text"], query)

        enriched.append({
        "chunk": meta["text"],
        "page": pg,
        "source": src,
        "score": hybrid_score,  # <-- Add this line
        "previous": source_chunks_map[src].get((pg, idx - 1), {}).get("text"),
        "next": source_chunks_map[src].get((pg, idx + 1), {}).get("text")
    })

    # 7. Optional: print results nicely
    for i, item in enumerate(enriched, 1):
        print(f"{i}. Page {item['page']} | Source: {item['source']} | Score: {item['score']}")
        if item["previous"]:
            print(f"   Previous: {item['previous']}\n")
        print(f"   Chunk: {item['chunk']}\n")
        if item["next"]:
            print(f"   Next: {item['next']}\n")

    return enriched


@app.post("/upload_pdf/")
def upload_pdf(file: UploadFile, source_id: str = Form(...)):
    reader = PdfReader(file.file)
    
    all_chunks = []

    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if not page_text:
            continue
        cleaned = clean_text(page_text)
        page_chunks = chunk_text(cleaned)
    
        for i, chunk in enumerate(page_chunks):
            all_chunks.append({
            "text": chunk,
            "page": page_num,
            "index_in_page": i,
            "total_in_page": len(page_chunks)
        })

    upsert_chunks(all_chunks, source_id)
    return {"message": f"PDF '{source_id}' processed and stored."}

@app.post("/upload_image/")
async def upload_image(
    file: UploadFile,
    image_id: str = Form(...)
):
    metadata = ImageMetadata(
        id=image_id,
        filename=file.filename
    )
    image_store.add_image(file, metadata)
    return {"message": f"Image '{image_id}' uploaded successfully"}

@app.get("/images/{filename}")
async def get_image(filename: str):
    result = image_store.get_image_by_filename(filename)
    if not result:
        raise HTTPException(status_code=404, detail="Image not found")
    image_path, metadata = result
    return FileResponse(image_path, media_type="image/jpeg")

@app.get("/api/energy-data")
async def get_energy_data_endpoint(latitude: float = 40.71, longitude: float = -74.01, days: int = 3):
    """Dedicated endpoint for energy data (solar/wind)"""
    result = query_energy_data(latitude, longitude, days)
    return result

@app.get("/api/industrial-data")
async def get_industrial_data_endpoint(api_key: str = "4XVxrVQ7H3ddVOvZKhYuggmOf64DCT9JvuKlZ0Sl"):
    """Dedicated endpoint for industrial electricity data"""
    result = query_industrial_data(api_key)
    return result

@app.get("/", response_class=FileResponse)
async def serve_web_interface():
    """Serve the web interface"""
    return FileResponse("web_interface.html")

@app.get("/demo")
async def demo_page():
    """Simple demo page with formatted response"""
    from fastapi.responses import HTMLResponse
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Energy Analysis Demo</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .query-box { background: #f0f8ff; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
            .result-box { background: #f9f9f9; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
            button { background: #4CAF50; color: white; padding: 12px 24px; border: none; border-radius: 4px; cursor: pointer; }
            textarea { width: 100%; height: 100px; padding: 10px; margin: 10px 0; }
            .response { white-space: pre-wrap; line-height: 1.6; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌱 Renewable Energy Analysis Demo</h1>
            
            <div class="query-box">
                <h3>Enter Your Query:</h3>
                <textarea id="query" placeholder="What are the solar and wind conditions in Berlin for energy planning next week?"></textarea>
                <button onclick="analyzeQuery()">Analyze</button>
            </div>
            
            <div id="results" class="result-box" style="display:none;">
                <h3>Analysis Results:</h3>
                <div id="response" class="response"></div>
            </div>
        </div>
        
        <script>
            async function analyzeQuery() {
                const query = document.getElementById('query').value;
                const resultsDiv = document.getElementById('results');
                const responseDiv = document.getElementById('response');
                
                if (!query.trim()) {
                    alert('Please enter a query');
                    return;
                }
                
                responseDiv.innerHTML = 'Analyzing...';
                resultsDiv.style.display = 'block';
                
                try {
                    const response = await fetch('/unified_query/', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query: query, include_images: false })
                    });
                    
                    const data = await response.json();
                    responseDiv.innerHTML = data.answer || 'No response received';
                } catch (error) {
                    responseDiv.innerHTML = 'Error: ' + error.message;
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/unified_query/")
async def unified_query(req: UnifiedQueryRequest):
    """
    Unified query endpoint that routes queries to appropriate handlers based on LLM classification
    """
    # Classify the query using keyword matching
    query_type = classify_query(req.query)
    
    print(f"Query classified as: {query_type}")
    
    if query_type == "weather":
        # Extract parameters from the query using LLM
        params = extract_energy_parameters(req.query)
        print(f"Extracted parameters: {params}")
        
        # Get coordinates for the location
        latitude, longitude = get_coordinates_for_location(params["location"])
        print(f"Using coordinates for {params['location']}: ({latitude}, {longitude})")
        
        # Query energy data with extracted parameters
        result = query_energy_data(latitude=latitude, longitude=longitude, days=params["days"])
        
        if result["status"] == "success" and result["data"]:
            # Convert data back to DataFrame for analysis
            import pandas as pd
            data_df = pd.DataFrame(result["data"])
            if not data_df.empty and 'timestamp' in data_df.columns:
                data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
                data_df = data_df.set_index('timestamp')
            
            # Use single comprehensive analysis with verified metrics
            comprehensive_answer = analyze_energy_data_comprehensive(
                data_df, 
                params["location"], 
                params["days"], 
                params["specific_question"],
                req.query
            )
            
            return {
                "query_type": "weather",
                "answer": comprehensive_answer,
                "location": params["location"],
                "days": params["days"],
                "coordinates": {"latitude": latitude, "longitude": longitude},
                "data": result["data"][:24] if result["data"] else None,  # Limit data in response
                "status": result["status"],
                "source": "energy_apis"
            }
        else:
            return {
                "query_type": "weather", 
                "answer": f"Sorry, I couldn't fetch energy data for {params['location']}. Please try a different location.",
                "location": params["location"],
                "status": "error",
                "source": "energy_apis"
            }
    
    elif query_type == "industrial":
        # Query industrial electricity data
        result = query_industrial_data()
        
        if result["status"] == "success" and result["data"]:
            # Convert industrial data to DataFrame for analysis
            import pandas as pd
            industrial_df = pd.DataFrame(result["data"])
            
            # Use comprehensive analysis for industrial data too
            comprehensive_answer = analyze_energy_data_comprehensive(
                industrial_df,
                "Industrial Grid", 
                7,  # Default analysis period
                "industrial electricity demand patterns",
                req.query
            )
            
            return {
                "query_type": "industrial",
                "answer": comprehensive_answer,
                "location": "Industrial Grid (PJM/MISO)",
                "days": 7,
                "data": result["data"][:10] if result["data"] else None,  # Limit data in response
                "status": result["status"],
                "source": "eia_api"
            }
        else:
            return {
                "query_type": "industrial",
                "answer": result["summary"],
                "data": result["data"],
                "status": result["status"],
                "source": "eia_api"
            }
    
    elif query_type == "optimize":
        result = optimize()
        return {
            "query_type": "optimize",
            "answer": result,
            "source": "optimization_function"
        }
    
    elif query_type == "visualize":
        result = visualize()
        return {
            "query_type": "visualize", 
            "answer": result,
            "source": "visualization_function"
        }
    
    elif query_type == "machine_info":
        # Use RAG for machine-specific information
        query_request = QueryRequest(question=req.query, include_images=req.include_images)
        rag_result = await query_docs(query_request)
        rag_result["query_type"] = "machine_info"
        return rag_result
    
    else:  # regular_conversation
        # Regular LLM conversation
        prompt = f"Answer the following question in a conversational way: {req.query}"
        response = model.generate_content(prompt)
        answer = response.text if hasattr(response, 'text') else str(response)
        
        return {
            "query_type": "regular_conversation",
            "answer": answer,
            "source": "llm_conversation"
        }

@app.post("/query_docs/")
async def query_docs(req: QueryRequest):
    # First try RAG search
    chunks = hybrid_search(req.question, req.top_k)

    # Filter chunks based on a minimum hybrid relevance score
    MIN_SCORE_THRESHOLD = 0.9  # tune this threshold
    relevant_chunks = [c for c in chunks if c.get("score", 0) >= MIN_SCORE_THRESHOLD]

    # Get relevant images using CLIP (do this regardless of text search result)
    relevant_images = []
    if req.include_images:
        relevant_images = image_store.get_relevant_images(req.question, top_k=1)
        if relevant_images:  # Only store if we have images
            _, metadata = relevant_images[0]
            # Store with a default session ID
            user_last_image["default_session"] = metadata
    
    # Prepare image context
    image_context = ""
    if relevant_images:
        image_context = "\nRelevant images:\n" + "\n".join(
            f"- {metadata.filename}"
            for _, metadata in relevant_images
        )

    # Only do classification if we have relevant images
    if relevant_images:
        # Before your main logic:
        clf_prompt = (
            "Is the user's question asking about the image currently displayed? "
            "Respond with YES or NO only.\n\n"
            f"Question: {req.question}"
        )

        # Simple synchronous classification
        classification = model.generate_content(clf_prompt).text.strip().upper()
        print("classification")
        print(classification)
        
        # If classification is YES and we have images, use image processing
        if classification == "YES":
            try:
                # Call image_follow_up with proper parameters
                image_response = await image_follow_up(
                    session_id="default_session",
                    question=req.question
                )
                return {
                    "chunks": [],
                    "answer": image_response["response"],
                    "relevant_images": [
                        {
                            "filename": metadata.filename
                        }
                        for _, metadata in relevant_images
                    ],
                    "source": "image"
                }
            except Exception as e:
                print(f"Error in image follow-up: {e}")
                # Fall through to regular processing if image handling fails

    # Regular text-based processing
    if relevant_chunks:  
        # Only use the first chunk
        context = relevant_chunks[0]["chunk"]
        prompt = f"""Answer the question based on the following context:

{context}
{image_context}
if the context does not contain relative information then perform google search
Question: {req.question}
Answer:"""
        response = model.generate_content(prompt)
        answer = response.text if hasattr(response, 'text') else str(response)
    else:
        # Fallback to direct question
        prompt = f"answer the following question in a summerize way in form of regular sentences : Question: {req.question}\nAnswer:"
        response = model.generate_content(prompt)
        try:
            if hasattr(response, 'text'):
                answer = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content.parts:
                    answer = candidate.content.parts[0].text
                else:
                    answer = str(response)
            else:
                answer = str(response)
        except Exception as e:
            print(f"Error processing response: {e}")
            answer = "I apologize, but I encountered an error processing the response."
    
    return {
        "chunks": [relevant_chunks[0]] if relevant_chunks else [],  # Only return the first chunk
        "answer": answer,
        "relevant_images": [
            {
                "filename": metadata.filename
            }
            for _, metadata in relevant_images
        ] if req.include_images else [],
        "source": "rag" if relevant_chunks else "web"  # Indicate the source of the answer
    }

@app.websocket("/ws/voice")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    recognizer = sr.Recognizer()
    
    try:
        while True:
            # Receive audio data
            data = await websocket.receive_bytes()
            
            # Convert bytes to audio file
            audio_io = io.BytesIO(data)
            with wave.open(audio_io, 'rb') as wav_file:
                # Read audio data
                audio_data = wav_file.readframes(wav_file.getnframes())
                
                # Convert to AudioData for speech recognition
                audio = sr.AudioData(
                    audio_data,
                    sample_rate=wav_file.getframerate(),
                    sample_width=wav_file.getsampwidth()
                )
                
                try:
                    # Perform speech recognition
                    text = recognizer.recognize_google(audio)
                    
                    # Process the query using unified query endpoint
                    query_request = UnifiedQueryRequest(query=text)
                    response = await unified_query(query_request)
                    
                    # Add the transcript to the response
                    response["transcript"] = text
                    
                    # Send back the response
                    await websocket.send_json(response)
                    
                except sr.UnknownValueError:
                    await websocket.send_json({
                        "error": "Could not understand audio",
                        "answer": "I'm sorry, I couldn't understand what you said. Could you please try again?",
                        "transcript": ""
                    })
                except sr.RequestError as e:
                    await websocket.send_json({
                        "error": f"Could not request results; {str(e)}",
                        "answer": "I'm sorry, there was an error processing your voice input. Please try again.",
                        "transcript": ""
                    })
                    
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error in websocket: {str(e)}")
        await websocket.close()

# Add CORS middleware to allow WebSocket connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)
