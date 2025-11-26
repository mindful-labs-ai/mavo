from fastapi import APIRouter, HTTPException, status, Path as FastAPIPath, BackgroundTasks
from typing import Dict, Any
import sys
import os
import json
from pathlib import Path
import traceback

# Add the parent directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logic.models import analysis_jobs, load_analysis_work_from_json
from logic.transcript_analysis.transcript_analysis_util import analyze_word_segments, analyze_sentiment_and_tense, analyze_word_segments_post, make_psycho_timeline_chart_data, render_psycho_timeline_chart
import config
from util.logger import get_logger

# Get logger
logger = get_logger(__name__)

# Create API router for maum endpoints
router = APIRouter(
    prefix="/api/v1/maum",
    tags=["Maum Analysis"],
)

@router.post("/{audio_uuid}", 
    summary="Run Sentiment Analysis",
    description="""
    Run sentiment analysis on the transcribed audio.
    
    This endpoint processes the transcription data to analyze sentiment, tense,
    speech cadence, and other linguistic features. The analysis includes:
    
    - Speaker cadence analysis
    - Speaking patterns
    - Sentiment analysis (positive/negative)
    - Tense analysis (past/future focus)
    
    The result is visualized as a chart and saved for retrieval.
    """,
    response_description="Analysis status"
)
async def run_sentiment_analysis(
    audio_uuid: str = FastAPIPath(..., description="Unique identifier for the audio file"),
) -> Dict[str, Any]:
    """
    Run sentiment analysis on the transcribed audio.
    
    Args:
        audio_uuid: Unique identifier for the audio file
        
    Returns:
        Dict[str, Any]: Status of the sentiment analysis
    """
    try:
        # Check if the analysis job exists in memory
        job = analysis_jobs.get(audio_uuid)
        
        # If not in memory, try to load from disk
        if job is None:
            job = load_analysis_work_from_json(audio_uuid)
            if job:
                # Add to memory
                analysis_jobs[audio_uuid] = job
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"No analysis job found for audio_uuid {audio_uuid}"
                )
        
        # Check if we have the required data
        if not job.result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Analysis job has no result data available"
            )
        
        # Create output directory for this job
        output_dir = config.UPLOADS_DIR / str(audio_uuid) / "analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        consecutive_segment_result_path = config.UPLOADS_DIR / str(audio_uuid) / f"id[{job.audio_uuid}]_consecutive_segments.json"
        consecutive_segment_result_json = json.load(open(consecutive_segment_result_path, "r", encoding="utf-8"))

        
        word_segments = consecutive_segment_result_json['word_segments']
        consecutive_segments = consecutive_segment_result_json['consecutive_segments']
        
        # Prepare data for analysis using dictionaries instead of Segment objects
        data = {
            "word_segments": word_segments,
            "consecutive_segments": consecutive_segments,
        }
        
        # Run word segment analysis
        logger.info(f"Running word segment analysis for {audio_uuid}")
        analyzed_data = analyze_word_segments(data)
        
        # Optionally run sentiment and tense analysis
        logger.info(f"Running sentiment and tense analysis for {audio_uuid}")
        consecutive_segments = analyzed_data.get("consecutive_segments", [])
        analyzed_segments = analyze_sentiment_and_tense(consecutive_segments)
        analyzed_data["consecutive_segments"] = analyzed_segments
        diarization_data_tmp = analyze_word_segments_post(analyzed_data)## add more stats
        analyzed_data['summary'].update(diarization_data_tmp['summary'])
        
        # Create chart data
        logger.info(f"Creating chart data for {audio_uuid}")
        word_segments = analyzed_data.get("word_segments", [])
        chart_data = make_psycho_timeline_chart_data(analyzed_segments, word_segments)
        
        # Save chart data
        chart_data_path = output_dir / "chart_data.json"
        with open(chart_data_path, "w", encoding="utf-8") as f:
            json.dump(chart_data, f, indent=2, ensure_ascii=False)
        
        # Render and save chart
        chart_path = output_dir / "psycho_timeline.png"
        render_psycho_timeline_chart(chart_data, str(chart_path))
        
        # Save analyzed data
        analyzed_data_path = output_dir / "analyzed_data.json"
        with open(analyzed_data_path, "w", encoding="utf-8") as f:
            json.dump(analyzed_data, f, indent=2, ensure_ascii=False)
        
        return {
            "audio_uuid": audio_uuid,
            "status": "success",
            "message": "Sentiment analysis completed successfully",
            "chart_path": f"/api/v1/maum/chart/{audio_uuid}",
            "analyzed_data_path": f"/api/v1/maum/data/{audio_uuid}"
        }
    
    except Exception as e:
        logger.error(f"Error in sentiment analysis for {audio_uuid}: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error running sentiment analysis: {str(e)}"
        )

@router.get("/chart/{audio_uuid}",
    summary="Get Sentiment Analysis Chart",
    description="Get the chart visualization of sentiment analysis",
    response_description="Chart image"
)
async def get_sentiment_chart(
    audio_uuid: str = FastAPIPath(..., description="Unique identifier for the audio file")
):
    """Get the sentiment analysis chart"""
    try:
        # Check if chart exists
        chart_path = config.UPLOADS_DIR / str(audio_uuid) / "analysis" / "psycho_timeline.png"
        
        if not chart_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Sentiment analysis chart not found"
            )
        
        # Return the chart
        return {
            "audio_uuid": audio_uuid,
            "chart_url": f"/api/v1/maum/chart/{audio_uuid}/file",
            "status": "success"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting sentiment chart for {audio_uuid}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting sentiment chart: {str(e)}"
        )

@router.get("/chart/{audio_uuid}/file",
    summary="Download Sentiment Analysis Chart",
    description="Download the chart visualization of sentiment analysis"
)
async def download_sentiment_chart(
    audio_uuid: str = FastAPIPath(..., description="Unique identifier for the audio file")
):
    """Download the sentiment analysis chart"""
    try:
        # Check if chart exists
        chart_path = config.UPLOADS_DIR / str(audio_uuid) / "analysis" / "psycho_timeline.png"
        
        if not chart_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Sentiment analysis chart not found"
            )
        
        # Return the chart
        from fastapi.responses import FileResponse
        return FileResponse(
            chart_path,
            media_type="image/png",
            filename=f"sentiment_analysis_{audio_uuid}.png"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading sentiment chart for {audio_uuid}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error downloading sentiment chart: {str(e)}"
        )

@router.get("/data/{audio_uuid}",
    summary="Get Sentiment Analysis Data",
    description="Get the raw data from the sentiment analysis",
    response_description="Analysis data"
)
async def get_sentiment_data(
    audio_uuid: str = FastAPIPath(..., description="Unique identifier for the audio file")
) -> Dict[str, Any]:
    """Get the raw sentiment analysis data"""
    try:
        # Check if data exists
        data_path = config.UPLOADS_DIR / str(audio_uuid) / "analysis" / "analyzed_data.json"
        
        if not data_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Sentiment analysis data not found"
            )
        
        # Read and return the data
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return {
            "audio_uuid": audio_uuid,
            "status": "success",
            "data": data
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting sentiment data for {audio_uuid}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting sentiment data: {str(e)}"
        ) 