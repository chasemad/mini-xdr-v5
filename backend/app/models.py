from sqlalchemy import Column, Integer, String, DateTime, JSON, Text, Index, Boolean
from sqlalchemy.sql import func
from .db import Base


class Event(Base):
    __tablename__ = "events"
    
    id = Column(Integer, primary_key=True)
    ts = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    src_ip = Column(String(64), index=True)
    dst_ip = Column(String(64), nullable=True)
    dst_port = Column(Integer, nullable=True)
    eventid = Column(String(128), index=True)
    message = Column(Text, nullable=True)
    raw = Column(JSON)
    
    __table_args__ = (Index("ix_events_src_ts", "src_ip", "ts"),)


class Incident(Base):
    __tablename__ = "incidents"
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    src_ip = Column(String(64), index=True)
    reason = Column(String(256))
    status = Column(String(32), default="open")  # open|contained|dismissed
    auto_contained = Column(Boolean, default=False)
    triage_note = Column(JSON, nullable=True)  # {summary, severity, recommendation, rationale}


class Action(Base):
    __tablename__ = "actions"
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    incident_id = Column(Integer, index=True)
    action = Column(String(32))   # block|unblock|scheduled_unblock
    result = Column(String(32))   # pending|success|failed|done
    detail = Column(Text, nullable=True)
    params = Column(JSON, nullable=True)
    due_at = Column(DateTime(timezone=True), nullable=True)
