"""
Nyika-Grid Predict AI — Webhook Alerts
------------------------------------------
alerts.py's send_alert() logs to a file — useful for demoing that an alert
*would* fire, but nothing downstream can actually react to it. This lets
an external system (a dashboard, a chatbot, an operations tool) register a
URL and receive a real HTTP POST when a region's load crosses its alert
threshold — actual integration, not just a log line.

Uses the same SQLite database as db.py, via a separate table.
"""

import datetime
import json
from typing import List, Dict, Optional
import requests
from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.orm import declarative_base

from db import engine, SessionLocal

Base = declarative_base()


class WebhookSubscription(Base):
    __tablename__ = "webhook_subscription"

    id = Column(Integer, primary_key=True)
    region = Column(String)  # region key, or "*" for all regions
    url = Column(String)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


def init_webhook_db():
    Base.metadata.create_all(engine)


def register_webhook(region: str, url: str) -> int:
    session = SessionLocal()
    entry = WebhookSubscription(region=region, url=url, active=True)
    session.add(entry)
    session.commit()
    entry_id = entry.id
    session.close()
    return entry_id


def list_webhooks(region: Optional[str] = None) -> List[WebhookSubscription]:
    session = SessionLocal()
    query = session.query(WebhookSubscription).filter(WebhookSubscription.active == True)  # noqa: E712
    if region:
        query = query.filter((WebhookSubscription.region == region) | (WebhookSubscription.region == "*"))
    results = query.all()
    session.close()
    return results


def deactivate_webhook(webhook_id: int) -> bool:
    session = SessionLocal()
    entry = session.get(WebhookSubscription, webhook_id)
    if entry is None:
        session.close()
        return False
    entry.active = False
    session.commit()
    session.close()
    return True


def dispatch_alert_to_webhooks(region: str, alert_payload: Dict, timeout: int = 5) -> List[Dict]:
    """
    Sends alert_payload as JSON to every active webhook subscribed to this
    region (or to "*"). Returns per-webhook delivery results — successes
    and failures both — so the caller can show what actually happened
    rather than assuming delivery succeeded.
    """
    webhooks = list_webhooks(region)
    results = []

    for webhook in webhooks:
        result = {"webhook_id": webhook.id, "url": webhook.url}
        try:
            resp = requests.post(webhook.url, json=alert_payload, timeout=timeout)
            result["success"] = resp.status_code < 400
            result["status_code"] = resp.status_code
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
        results.append(result)

    return results


init_webhook_db()
