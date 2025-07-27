"""
API server for multilingual alignment evaluation.
Provides RESTful endpoints for evaluation services.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import json
import pandas as pd
import tempfile
import asyncio
from pathlib import Path
from datetime import datetime
import uuid

from scripts.evaluate import MultilingualEvaluator
from scripts.data_processing import DataValidator, DataPreprocessor
from scripts.prompt_manager import MultilingualPromptManager
from scripts.model_adapters import ModelFactory, ModelConfig, MultiModelEvaluator
from scripts.visualization import EvaluationVisualizer, ReportGenerator

app = FastAPI(
    title="Multilingual Alignment Evaluation API",
    description="API for evaluating language model alignment across multiple languages",
    version="0.1.0"
)

# Store job results temporarily
job_results = {}


class EvaluationRequest(BaseModel):
    """Request model for evaluation."""
    data: List[Dict[str, Any]]
    config: Optional[Dict[str, Any]] = None
    generate_report: bool = True
    visualization_type: str = "dashboard"


class PromptGenerationRequest(BaseModel):
    """Request model for prompt generation."""
    languages: List[str] = Field(..., example=["en", "sw", "hi"])
    categories: Optional[List[str]] = None
    samples_per_template: int = Field(3, ge=1, le=10)


class ModelTestRequest(BaseModel):
    """Request model for model testing."""
    prompts: List[str]
    models: Dict[str, Dict[str, Any]]  # model_name -> config
    parallel: bool = True


class JobResponse(BaseModel):
    """Response model for async jobs."""
    job_id: str
    status: str
    message: str


class EvaluationResponse(BaseModel):
    """Response model for evaluation results."""
    metrics: Dict[str, Any]
    report_url: Optional[str] = None
    visualization_url: Optional[str] = None


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Multilingual Alignment Evaluation API",
        "version": "0.1.0",
        "endpoints": {
            "POST /evaluate": "Evaluate alignment metrics",
            "POST /evaluate/file": "Evaluate from uploaded file",
            "POST /prompts/generate": "Generate multilingual prompts",
            "POST /models/test": "Test multiple models",
            "GET /jobs/{job_id}": "Get job status",
            "GET /health": "Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_data(request: EvaluationRequest):
    """Evaluate alignment metrics on provided data."""
    try:
        # Convert to DataFrame
        df = pd.DataFrame(request.data)
        
        # Initialize evaluator
        evaluator = MultilingualEvaluator(config=request.config)
        
        # Validate data
        validator = DataValidator(config=request.config)
        df, validation_report = validator.validate_dataset(df, strict=False)
        
        # Calculate metrics
        metrics = evaluator.calculate_metrics(df)
        
        response = {"metrics": metrics}
        
        # Generate report if requested
        if request.generate_report:
            report_gen = ReportGenerator(config=request.config)
            report_path = f"temp/report_{uuid.uuid4()}.html"
            report_gen.generate_html_report(metrics, df, report_path)
            response["report_url"] = f"/files/{Path(report_path).name}"
        
        # Generate visualization
        if request.visualization_type:
            visualizer = EvaluationVisualizer()
            vis_path = f"temp/vis_{uuid.uuid4()}.html"
            
            if request.visualization_type == "dashboard":
                visualizer.create_overview_dashboard(metrics, vis_path)
            elif request.visualization_type == "heatmap" and 'language' in df.columns:
                visualizer.plot_language_heatmap(df, save_path=vis_path)
                
            response["visualization_url"] = f"/files/{Path(vis_path).name}"
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/evaluate/file")
async def evaluate_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    generate_report: bool = True
):
    """Evaluate alignment metrics from uploaded file."""
    # Generate job ID
    job_id = str(uuid.uuid4())
    job_results[job_id] = {"status": "processing", "result": None}
    
    # Save uploaded file
    temp_path = f"temp/upload_{job_id}.jsonl"
    content = await file.read()
    
    with open(temp_path, 'wb') as f:
        f.write(content)
    
    # Process in background
    background_tasks.add_task(
        process_evaluation_job, 
        job_id, 
        temp_path, 
        generate_report
    )
    
    return JobResponse(
        job_id=job_id,
        status="processing",
        message="Evaluation job started"
    )


async def process_evaluation_job(job_id: str, file_path: str, generate_report: bool):
    """Process evaluation job in background."""
    try:
        # Load data
        df = pd.read_json(file_path, lines=True)
        
        # Evaluate
        evaluator = MultilingualEvaluator()
        metrics = evaluator.calculate_metrics(df)
        
        result = {"metrics": metrics}
        
        # Generate report
        if generate_report:
            report_gen = ReportGenerator()
            report_path = f"temp/report_{job_id}.html"
            report_gen.generate_html_report(metrics, df, report_path)
            result["report_url"] = f"/files/{Path(report_path).name}"
        
        job_results[job_id] = {"status": "completed", "result": result}
        
    except Exception as e:
        job_results[job_id] = {"status": "failed", "error": str(e)}
    
    finally:
        # Clean up temp file
        Path(file_path).unlink(missing_ok=True)


@app.post("/prompts/generate")
async def generate_prompts(request: PromptGenerationRequest):
    """Generate multilingual prompt test suite."""
    try:
        manager = MultilingualPromptManager()
        
        test_suite = manager.generate_test_suite(
            languages=request.languages,
            categories=request.categories,
            samples_per_template=request.samples_per_template
        )
        
        # Get statistics
        stats = manager.get_statistics()
        
        return {
            "prompts": test_suite,
            "count": len(test_suite),
            "statistics": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/models/test")
async def test_models(request: ModelTestRequest):
    """Test multiple models with prompts."""
    try:
        # Initialize models
        model_adapters = {}
        for name, config in request.models.items():
            provider = config.pop("provider", "openai")
            model_config = ModelConfig(**config)
            adapter = ModelFactory.create(provider, model_config)
            model_adapters[name] = adapter
        
        # Create evaluator
        evaluator = MultiModelEvaluator(model_adapters)
        
        # Test prompts
        if request.parallel:
            results = await asyncio.gather(*[
                evaluator.evaluate_prompt_async(prompt)
                for prompt in request.prompts
            ])
        else:
            results = []
            for prompt in request.prompts:
                result = await evaluator.evaluate_prompt_async(prompt)
                results.append(result)
        
        # Format results
        formatted_results = []
        for i, prompt in enumerate(request.prompts):
            formatted_results.append({
                "prompt": prompt,
                "responses": {
                    model: resp.to_dict() 
                    for model, resp in results[i].items()
                }
            })
        
        return {
            "results": formatted_results,
            "summary": {
                "total_prompts": len(request.prompts),
                "models_tested": list(model_adapters.keys())
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get status of an async job."""
    if job_id not in job_results:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = job_results[job_id]
    
    if job_info["status"] == "completed":
        return {
            "status": "completed",
            "result": job_info["result"]
        }
    elif job_info["status"] == "failed":
        return {
            "status": "failed",
            "error": job_info.get("error", "Unknown error")
        }
    else:
        return {
            "status": "processing",
            "message": "Job is still being processed"
        }


@app.get("/files/{filename}")
async def get_file(filename: str):
    """Serve generated files."""
    file_path = f"temp/{filename}"
    
    if not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    if filename.endswith('.html'):
        return FileResponse(file_path, media_type="text/html")
    else:
        return FileResponse(file_path)


@app.on_event("startup")
async def startup_event():
    """Initialize temp directory on startup."""
    Path("temp").mkdir(exist_ok=True)


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up temp files on shutdown."""
    import shutil
    shutil.rmtree("temp", ignore_errors=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)