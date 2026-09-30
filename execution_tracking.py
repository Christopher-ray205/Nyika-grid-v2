"""
Nyika-Grid Predict AI — Shedding Execution Tracking
------------------------------------------------------------
Closes a gap the forecast-accuracy tracker (db.py's PredictionLog) doesn't
cover: whether a PLANNED shedding action actually happened as scheduled.

The accuracy tracker asks "was our number right?" This module asks a
different, operationally important question: "did the plan actually get
carried out?" A forecast can be accurate and the shedding schedule still
not get executed as planned — that's a real, distinct failure mode this
tracks separately.

Uses the same SQLite database as db.py (nyika_grid.db), via a separate table.
"""

import datetime
from typing import List, Optional, Dict
from sqlalchemy import Column, Integer, String, Date, DateTime
from sqlalchemy.orm import declarative_base

from db import engine, SessionLocal

Base = declarative_base()


class ShedExecutionLog(Base):
    __tablename__ = "shed_execution_log"

    id = Column(Integer, primary_key=True)
    region = Column(String)
    shed_date = Column(Date)
    hour = Column(Integer)
    zone = Column(String)
    status = Column(String)  # "planned", "executed", "skipped"
    notes = Column(String, nullable=True)
    logged_at = Column(DateTime, default=datetime.datetime.utcnow)


def init_execution_db():
    Base.metadata.create_all(engine)


def log_planned_shedding(region: str, shed_date: datetime.date, hour: int, zone: str) -> int:
    """Called when a shedding schedule is generated — records what was PLANNED."""
    session = SessionLocal()
    entry = ShedExecutionLog(region=region, shed_date=shed_date, hour=hour, zone=zone, status="planned")
    session.add(entry)
    session.commit()
    entry_id = entry.id
    session.close()
    return entry_id


def mark_execution_status(entry_id: int, status: str, notes: Optional[str] = None) -> bool:
    """status: 'executed' or 'skipped'. Called by an operator confirming what actually happened."""
    if status not in ("executed", "skipped"):
        raise ValueError("status must be 'executed' or 'skipped'")

    session = SessionLocal()
    entry = session.get(ShedExecutionLog, entry_id)
    if entry is None:
        session.close()
        return False
    entry.status = status
    if notes is not None:
        entry.notes = notes
    session.commit()
    session.close()
    return True


def get_planned_entries(region: str, shed_date: datetime.date) -> List[ShedExecutionLog]:
    session = SessionLocal()
    entries = (session.query(ShedExecutionLog)
               .filter(ShedExecutionLog.region == region, ShedExecutionLog.shed_date == shed_date)
               .order_by(ShedExecutionLog.hour)
               .all())
    session.close()
    return entries


def get_compliance_summary(region: Optional[str] = None, days_back: int = 30) -> Dict:
    """Plan-vs-reality summary: of everything planned, how much actually happened as scheduled?"""
    cutoff = datetime.date.today() - datetime.timedelta(days=days_back)
    session = SessionLocal()
    query = session.query(ShedExecutionLog).filter(ShedExecutionLog.shed_date >= cutoff)
    if region:
        query = query.filter(ShedExecutionLog.region == region)
    entries = query.all()
    session.close()

    total = len(entries)
    executed = sum(1 for e in entries if e.status == "executed")
    skipped = sum(1 for e in entries if e.status == "skipped")
    still_planned = sum(1 for e in entries if e.status == "planned")

    compliance_pct = round(100 * executed / total, 1) if total > 0 else None

    return {
        "total_logged": total,
        "executed": executed,
        "skipped": skipped,
        "awaiting_confirmation": still_planned,
        "compliance_pct": compliance_pct,
    }


init_execution_db()
