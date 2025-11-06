"""World Model - Knowledge base for research state"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import json


class HypothesisStatus(Enum):
    """Status of a hypothesis"""
    UNTESTED = "untested"
    TESTING = "testing"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"


class Priority(Enum):
    """Task priority levels"""
    HIGH = 3  # Untested hypotheses
    MEDIUM = 2  # Follow-up analyses
    LOW = 1  # Exploratory questions


@dataclass
class Hypothesis:
    """A research hypothesis"""
    id: str
    text: str
    status: HypothesisStatus
    priority: Priority
    created_at: datetime
    tested_at: Optional[datetime] = None
    source: str = "generated"  # "generated", "literature", "data"
    evidence: List[str] = field(default_factory=list)  # List of finding IDs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "tested_at": self.tested_at.isoformat() if self.tested_at else None,
            "source": self.source,
            "evidence": self.evidence
        }


@dataclass
class Finding:
    """A research finding from data analysis"""
    id: str
    text: str
    data: Dict[str, Any]  # Statistics, plots, etc.
    cycle: int
    created_at: datetime
    hypothesis_id: Optional[str] = None
    source: str = "data"  # "data" or "literature"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "data": self.data,
            "cycle": self.cycle,
            "created_at": self.created_at.isoformat(),
            "hypothesis_id": self.hypothesis_id,
            "source": self.source
        }


@dataclass
class Paper:
    """A literature reference"""
    id: str
    title: str
    authors: str
    year: Optional[int]
    abstract: str
    url: Optional[str]
    relevance_score: float
    created_at: datetime
    related_hypothesis_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "abstract": self.abstract,
            "url": self.url,
            "relevance_score": self.relevance_score,
            "created_at": self.created_at.isoformat(),
            "related_hypothesis_ids": self.related_hypothesis_ids
        }


@dataclass
class Task:
    """A research task"""
    id: str
    type: str  # "data_analysis", "literature_search"
    description: str
    priority: Priority
    created_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "pending"  # "pending", "running", "completed", "failed"
    result: Optional[Dict[str, Any]] = None
    hypothesis_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "description": self.description,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status,
            "result": self.result,
            "hypothesis_id": self.hypothesis_id
        }


class WorldModel:
    """Central knowledge base for research state"""

    def __init__(self):
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.findings: Dict[str, Finding] = {}
        self.papers: Dict[str, Paper] = {}
        self.tasks: Dict[str, Task] = {}
        self.current_cycle: int = 0
        self.last_finding_cycle: int = 0
        self._id_counter = 0

    def generate_id(self, prefix: str = "") -> str:
        """Generate a unique ID"""
        self._id_counter += 1
        return f"{prefix}{self._id_counter:04d}"

    def add_hypothesis(self, text: str, priority: Priority = Priority.HIGH,
                      source: str = "generated") -> Hypothesis:
        """Add a new hypothesis"""
        hyp = Hypothesis(
            id=self.generate_id("H"),
            text=text,
            status=HypothesisStatus.UNTESTED,
            priority=priority,
            created_at=datetime.now(),
            source=source
        )
        self.hypotheses[hyp.id] = hyp
        return hyp

    def add_finding(self, text: str, data: Dict[str, Any],
                   hypothesis_id: Optional[str] = None,
                   source: str = "data") -> Finding:
        """Add a new finding"""
        finding = Finding(
            id=self.generate_id("F"),
            text=text,
            data=data,
            cycle=self.current_cycle,
            created_at=datetime.now(),
            hypothesis_id=hypothesis_id,
            source=source
        )
        self.findings[finding.id] = finding
        self.last_finding_cycle = self.current_cycle
        return finding

    def add_paper(self, title: str, authors: str, abstract: str,
                 year: Optional[int] = None, url: Optional[str] = None,
                 relevance_score: float = 0.5,
                 related_hypothesis_ids: List[str] = None) -> Paper:
        """Add a new paper"""
        paper = Paper(
            id=self.generate_id("P"),
            title=title,
            authors=authors,
            year=year,
            abstract=abstract,
            url=url,
            relevance_score=relevance_score,
            created_at=datetime.now(),
            related_hypothesis_ids=related_hypothesis_ids or []
        )
        self.papers[paper.id] = paper
        return paper

    def add_task(self, task_type: str, description: str,
                priority: Priority = Priority.MEDIUM,
                hypothesis_id: Optional[str] = None) -> Task:
        """Add a new task"""
        task = Task(
            id=self.generate_id("T"),
            type=task_type,
            description=description,
            priority=priority,
            created_at=datetime.now(),
            hypothesis_id=hypothesis_id
        )
        self.tasks[task.id] = task
        return task

    def update_hypothesis_status(self, hypothesis_id: str,
                                status: HypothesisStatus,
                                evidence_finding_id: Optional[str] = None):
        """Update hypothesis status"""
        if hypothesis_id in self.hypotheses:
            hyp = self.hypotheses[hypothesis_id]
            hyp.status = status
            hyp.tested_at = datetime.now()
            if evidence_finding_id:
                hyp.evidence.append(evidence_finding_id)

    def get_untested_hypotheses(self, limit: int = 10) -> List[Hypothesis]:
        """Get untested hypotheses ordered by priority"""
        untested = [h for h in self.hypotheses.values()
                   if h.status == HypothesisStatus.UNTESTED]
        return sorted(untested, key=lambda h: h.priority.value, reverse=True)[:limit]

    def get_pending_tasks(self, limit: int = 10) -> List[Task]:
        """Get pending tasks ordered by priority"""
        pending = [t for t in self.tasks.values()
                  if t.status == "pending"]
        return sorted(pending, key=lambda t: t.priority.value, reverse=True)[:limit]

    def get_recent_findings(self, limit: int = 10) -> List[Finding]:
        """Get most recent findings"""
        all_findings = sorted(self.findings.values(),
                            key=lambda f: f.created_at, reverse=True)
        return all_findings[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        return {
            "cycle": self.current_cycle,
            "total_hypotheses": len(self.hypotheses),
            "untested_hypotheses": len([h for h in self.hypotheses.values()
                                       if h.status == HypothesisStatus.UNTESTED]),
            "total_findings": len(self.findings),
            "total_papers": len(self.papers),
            "total_tasks": len(self.tasks),
            "pending_tasks": len([t for t in self.tasks.values()
                                if t.status == "pending"]),
            "last_finding_cycle": self.last_finding_cycle
        }

    def save_to_file(self, filepath: str):
        """Save world model to JSON file"""
        data = {
            "hypotheses": [h.to_dict() for h in self.hypotheses.values()],
            "findings": [f.to_dict() for f in self.findings.values()],
            "papers": [p.to_dict() for p in self.papers.values()],
            "tasks": [t.to_dict() for t in self.tasks.values()],
            "current_cycle": self.current_cycle,
            "last_finding_cycle": self.last_finding_cycle
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"WorldModel(cycle={stats['cycle']}, hypotheses={stats['total_hypotheses']}, findings={stats['total_findings']}, papers={stats['total_papers']})"
