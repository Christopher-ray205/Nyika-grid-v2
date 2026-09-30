"""
Nyika-Grid Predict AI — Database Layer
-------------------------------------------
Logs every prediction made, and lets actual outcomes be recorded later
so the dashboard can show real forecast-vs-actual accuracy over time
(documentation Section 15/17: "trust the model" / evaluation).

Uses SQLite by default (nyika_grid.db) — swap the engine URL for
PostgreSQL in production (Section 13.1).
"""

import datetime
import numpy as np
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()
engine = create_engine("sqlite:///nyika_grid.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)


class PredictionLog(Base):
    __tablename__ = "prediction_log"

    id = Column(Integer, primary_key=True)
    region = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    hour = Column(Integer)
    temperature_c = Column(Float)
    is_weekend = Column(Integer)
    predicted_load_mw = Column(Float)
    predicted_solar_mw = Column(Float, nullable=True)
    actual_load_mw = Column(Float, nullable=True)
    actual_solar_mw = Column(Float, nullable=True)


def init_db():
    Base.metadata.create_all(engine)


def log_prediction(region, hour, temperature_c, is_weekend, predicted_load_mw, predicted_solar_mw=None):
    session = SessionLocal()
    entry = PredictionLog(
        region=region, hour=hour, temperature_c=temperature_c, is_weekend=is_weekend,
        predicted_load_mw=predicted_load_mw, predicted_solar_mw=predicted_solar_mw
    )
    session.add(entry)
    session.commit()
    entry_id = entry.id
    session.close()
    return entry_id


def log_actual(entry_id, actual_load_mw=None, actual_solar_mw=None):
    session = SessionLocal()
    entry = session.get(PredictionLog, entry_id)
    if entry is None:
        session.close()
        return False
    if actual_load_mw is not None:
        entry.actual_load_mw = actual_load_mw
    if actual_solar_mw is not None:
        entry.actual_solar_mw = actual_solar_mw
    session.commit()
    session.close()
    return True


def get_recent_logs(limit=20, region=None):
    session = SessionLocal()
    query = session.query(PredictionLog)
    if region:
        query = query.filter(PredictionLog.region == region)
    logs = query.order_by(PredictionLog.id.desc()).limit(limit).all()
    session.close()
    return logs


def get_accuracy_stats(region=None):
    session = SessionLocal()
    query = session.query(PredictionLog).filter(PredictionLog.actual_load_mw.isnot(None))
    if region:
        query = query.filter(PredictionLog.region == region)
    logs = query.all()
    session.close()

    if not logs:
        return None

    errors = np.array([l.predicted_load_mw - l.actual_load_mw for l in logs])
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    return {"count": len(logs), "rmse": rmse, "mae": mae}


# Create tables on import so any module using this file works immediately
init_db()
