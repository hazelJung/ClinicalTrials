from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    projects = relationship("Project", back_populates="owner")

class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(String, nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User", back_populates="projects")
    predictions = relationship("Prediction", back_populates="project")

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    smiles = Column(String, index=True)
    
    # Store full prediction results as JSON
    result_json = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    project = relationship("Project", back_populates="predictions")

class Cohort(Base):
    __tablename__ = "cohorts"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    name = Column(String, index=True)
    description = Column(String, nullable=True)
    
    # Validation Inputs (Demographics) stored as JSON
    # { "pop": "EUR", "gender": "BOTH", "female_ratio": 0.5, "age_min": 20, "age_max": 50, "n_subjects": 100 }
    demographics = Column(JSON)
    
    # Drug Parameters stored as JSON
    # { "mw": 151.16, "cl": 2.0, "vd": 1.5, "fu": 0.8, "source": "prediction|manual", "pred_id": 123 }
    drug_params = Column(JSON)
    
    # Simulation Results
    # { "mean_cmax": 80.3, "success_rate": 0.0, "phenotypes": {...} }
    results_json = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    project = relationship("Project", back_populates="cohorts")

# Add backref to Project
Project.cohorts = relationship("Cohort", back_populates="project")


class INDReport(Base):
    """Store generated IND application documents"""
    __tablename__ = "ind_reports"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)
    title = Column(String, index=True)
    file_path = Column(String)  # Path to generated DOCX file
    status = Column(String, default="Completed")  # Completed, Draft, etc.
    meta_data = Column(JSON, nullable=True)  # Store input parameters
    created_at = Column(DateTime, default=datetime.utcnow)
    
    project = relationship("Project", back_populates="ind_reports")

# Add backref to Project for IND Reports
Project.ind_reports = relationship("INDReport", back_populates="project")

