# app/main.py
from fastapi import FastAPI
from app.api import routes
from app.database.session import init_db

app = FastAPI()

@app.on_event("startup")
def on_startup():
    init_db()

app.include_router(routes.router)

@app.get("/")
def root():
    return {"message": "YOLOv8 FastAPI Inference Service"}

# app/api/routes.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from uuid import uuid4
from app.core.celery_utils import celery_app
from app.models import db_models
from app.database.session import SessionLocal
from app.core.config import settings

router = APIRouter(prefix="/inference")

@router.post("/image")
async def image_inference(file: UploadFile = File(...)):
    task_id = str(uuid4())
    file_path = f"{settings.RESULTS_FOLDER}/{task_id}_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    celery_app.send_task("app.workers.image_worker.image_inference_task", args=[file_path, task_id], queue="image_queue")
    db = SessionLocal()
    db_task = db_models.InferenceTask(task_id=task_id, file_path=file_path, status="PENDING", type="image")
    db.add(db_task)
    db.commit()
    db.close()
    return {"task_id": task_id}

@router.post("/video")
async def video_inference(file: UploadFile = File(...)):
    task_id = str(uuid4())
    file_path = f"{settings.RESULTS_FOLDER}/{task_id}_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    celery_app.send_task("app.workers.video_worker.video_inference_task", args=[file_path, task_id], queue="video_queue")
    db = SessionLocal()
    db_task = db_models.InferenceTask(task_id=task_id, file_path=file_path, status="PENDING", type="video")
    db.add(db_task)
    db.commit()
    db.close()
    return {"task_id": task_id}

@router.get("/task/{task_id}")
def get_task_status(task_id: str):
    db = SessionLocal()
    task = db.query(db_models.InferenceTask).filter(db_models.InferenceTask.task_id == task_id).first()
    db.close()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"task_id": task.task_id, "status": task.status, "output_path": task.output_path}

# app/core/config.py
from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    DB_URL: str = os.getenv("DB_URL")
    REDIS_BROKER_URL: str = os.getenv("REDIS_BROKER_URL")
    RESULTS_FOLDER: str = os.getenv("RESULTS_FOLDER", "/app/static/results")

settings = Settings()

# app/core/celery_utils.py
from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "worker",
    broker=settings.REDIS_BROKER_URL,
    backend=settings.REDIS_BROKER_URL,
)

# app/models/db_models.py
from sqlalchemy import Column, String, Integer
from app.database.session import Base

class InferenceTask(Base):
    __tablename__ = "inference_tasks"
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String, unique=True, index=True)
    file_path = Column(String)
    output_path = Column(String, nullable=True)
    status = Column(String)
    type = Column(String)

# app/database/session.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

engine = create_engine(settings.DB_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

def init_db():
    Base.metadata.create_all(bind=engine)

# app/workers/image_worker.py
from app.core.celery_utils import celery_app
from app.services.inference import run_image_inference
from app.database.session import SessionLocal
from app.models.db_models import InferenceTask

@celery_app.task(name="app.workers.image_worker.image_inference_task")
def image_inference_task(file_path: str, task_id: str):
    db = SessionLocal()
    try:
        task = db.query(InferenceTask).filter(InferenceTask.task_id == task_id).first()
        if task:
            task.status = "PROCESSING"
            db.commit()
        output_path = run_image_inference(file_path, task_id)
        if task:
            task.status = "COMPLETED"
            task.output_path = output_path
            db.commit()
    except Exception:
        if task:
            task.status = "FAILED"
            db.commit()
        raise
    finally:
        db.close()

# app/workers/video_worker.py
from app.core.celery_utils import celery_app
from app.services.inference import run_video_inference
from app.database.session import SessionLocal
from app.models.db_models import InferenceTask

@celery_app.task(name="app.workers.video_worker.video_inference_task")
def video_inference_task(file_path: str, task_id: str):
    db = SessionLocal()
    try:
        task = db.query(InferenceTask).filter(InferenceTask.task_id == task_id).first()
        if task:
            task.status = "PROCESSING"
            db.commit()
        output_path = run_video_inference(file_path, task_id)
        if task:
            task.status = "COMPLETED"
            task.output_path = output_path
            db.commit()
    except Exception:
        if task:
            task.status = "FAILED"
            db.commit()
        raise
    finally:
        db.close()

# app/services/inference.py
from ultralytics import YOLO
import os
from app.core.config import settings

model = YOLO("yolov8n.pt")

def run_image_inference(file_path: str, task_id: str) -> str:
    output_dir = os.path.join(settings.RESULTS_FOLDER, f"{task_id}")
    os.makedirs(output_dir, exist_ok=True)
    model(file_path, save=True, project=output_dir, name="result")
    return os.path.join(output_dir, "result", os.path.basename(file_path))

def run_video_inference(file_path: str, task_id: str) -> str:
    output_dir = os.path.join(settings.RESULTS_FOLDER, f"{task_id}")
    os.makedirs(output_dir, exist_ok=True)
    model(file_path, save=True, project=output_dir, name="result")
    return os.path.join(output_dir, "result", os.path.basename(file_path))
