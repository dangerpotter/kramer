"""
Evaluation Interface - Interface for expert review of claims.

This module provides interactive interfaces for domain experts to evaluate
extracted claims, collecting verdicts and storing evaluations in a database.
"""

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.evaluation.claim_extractor import Claim, ClaimType

logger = logging.getLogger(__name__)


class Verdict(Enum):
    """Possible verdicts for claim evaluation."""
    SUPPORTED = "supported"  # Claim is accurate/supported by evidence
    REFUTED = "refuted"  # Claim is inaccurate/contradicted by evidence
    UNCLEAR = "unclear"  # Insufficient evidence or ambiguous
    PARTIALLY_SUPPORTED = "partially_supported"  # Partially accurate


@dataclass
class Evaluation:
    """Represents an expert evaluation of a claim."""
    evaluation_id: Optional[str] = None
    claim_id: str = ""
    verdict: Optional[Verdict] = None
    evaluator_id: str = "default"
    notes: str = ""
    confidence_in_verdict: Optional[float] = None  # Expert's confidence in their verdict
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Set defaults."""
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.evaluation_id is None:
            self.evaluation_id = f"eval_{self.claim_id}_{self.timestamp.strftime('%Y%m%d%H%M%S')}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert evaluation to dictionary."""
        return {
            "evaluation_id": self.evaluation_id,
            "claim_id": self.claim_id,
            "verdict": self.verdict.value if self.verdict else None,
            "evaluator_id": self.evaluator_id,
            "notes": self.notes,
            "confidence_in_verdict": self.confidence_in_verdict,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata,
        }


class EvaluationInterface:
    """
    Interface for expert review of claims.

    Provides methods to:
    1. Present claims with context to evaluators
    2. Collect verdicts and notes
    3. Save evaluations to database
    4. Retrieve and manage evaluations
    """

    def __init__(self, db_path: Path):
        """
        Initialize the evaluation interface.

        Args:
            db_path: Path to SQLite database for storing evaluations
        """
        self.db_path = db_path
        self._init_database()

    def _init_database(self) -> None:
        """Initialize the evaluation database schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Create evaluations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                evaluation_id TEXT PRIMARY KEY,
                claim_id TEXT NOT NULL,
                verdict TEXT,
                evaluator_id TEXT NOT NULL,
                notes TEXT,
                confidence_in_verdict REAL,
                timestamp TEXT NOT NULL,
                metadata TEXT,
                UNIQUE(claim_id, evaluator_id)
            )
        """)

        # Create claims table for reference
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS claims (
                claim_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                claim_type TEXT NOT NULL,
                discovery_title TEXT,
                confidence REAL,
                context TEXT,
                source_section TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL
            )
        """)

        # Create index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_evaluations_claim_id
            ON evaluations(claim_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_evaluations_verdict
            ON evaluations(verdict)
        """)

        conn.commit()
        conn.close()

        logger.info(f"Initialized evaluation database at {self.db_path}")

    def store_claims(self, claims: List[Claim]) -> None:
        """
        Store claims in the database for reference.

        Args:
            claims: List of claims to store
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        for claim in claims:
            cursor.execute("""
                INSERT OR REPLACE INTO claims (
                    claim_id, text, claim_type, discovery_title,
                    confidence, context, source_section, metadata, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                claim.claim_id,
                claim.text,
                claim.claim_type.value,
                claim.discovery_title,
                claim.confidence,
                claim.context,
                claim.source_section,
                str(claim.metadata),
                datetime.now().isoformat(),
            ))

        conn.commit()
        conn.close()

        logger.info(f"Stored {len(claims)} claims in database")

    def present_claim(self, claim: Claim, show_confidence: bool = True) -> None:
        """
        Display a claim with context for evaluation.

        Args:
            claim: The claim to present
            show_confidence: Whether to show the original confidence score
        """
        print("\n" + "=" * 80)
        print(f"CLAIM {claim.claim_id}")
        print("=" * 80)
        print(f"\nDiscovery: {claim.discovery_title}")
        print(f"Type: {claim.claim_type.value}")
        if show_confidence and claim.confidence:
            print(f"Original Confidence: {claim.confidence:.2f}")
        print(f"\nClaim:\n{claim.text}")

        if claim.context:
            print(f"\nContext:\n{claim.context}")

        print("\n" + "-" * 80)

    def collect_verdict(
        self,
        claim: Claim,
        evaluator_id: str = "default",
        interactive: bool = True
    ) -> Evaluation:
        """
        Collect expert verdict for a claim.

        Args:
            claim: The claim to evaluate
            evaluator_id: Identifier for the evaluator
            interactive: Whether to use interactive prompts (or return empty evaluation)

        Returns:
            Evaluation object with verdict
        """
        if not interactive:
            # Return empty evaluation for non-interactive mode
            return Evaluation(claim_id=claim.claim_id, evaluator_id=evaluator_id)

        # Present the claim
        self.present_claim(claim)

        # Collect verdict
        print("\nVerdicts:")
        print("  1. Supported - Claim is accurate/supported by evidence")
        print("  2. Refuted - Claim is inaccurate/contradicted by evidence")
        print("  3. Unclear - Insufficient evidence or ambiguous")
        print("  4. Partially Supported - Claim is partially accurate")

        while True:
            choice = input("\nEnter verdict (1-4) or 's' to skip: ").strip().lower()

            if choice == 's':
                return Evaluation(
                    claim_id=claim.claim_id,
                    evaluator_id=evaluator_id,
                    verdict=None
                )

            verdict_map = {
                '1': Verdict.SUPPORTED,
                '2': Verdict.REFUTED,
                '3': Verdict.UNCLEAR,
                '4': Verdict.PARTIALLY_SUPPORTED,
            }

            if choice in verdict_map:
                verdict = verdict_map[choice]
                break
            else:
                print("Invalid choice. Please enter 1-4 or 's' to skip.")

        # Collect confidence in verdict
        confidence = None
        while True:
            conf_input = input("\nYour confidence in this verdict (0.0-1.0) or press Enter to skip: ").strip()
            if not conf_input:
                break
            try:
                confidence = float(conf_input)
                if 0.0 <= confidence <= 1.0:
                    break
                else:
                    print("Confidence must be between 0.0 and 1.0")
            except ValueError:
                print("Invalid input. Please enter a number between 0.0 and 1.0")

        # Collect notes
        notes = input("\nNotes (optional, press Enter to skip): ").strip()

        # Create evaluation
        evaluation = Evaluation(
            claim_id=claim.claim_id,
            verdict=verdict,
            evaluator_id=evaluator_id,
            notes=notes,
            confidence_in_verdict=confidence,
        )

        return evaluation

    def save_evaluation(self, evaluation: Evaluation) -> None:
        """
        Save a single evaluation to the database.

        Args:
            evaluation: The evaluation to save
        """
        self.save_evaluations([evaluation])

    def save_evaluations(self, evaluations: List[Evaluation]) -> None:
        """
        Save multiple evaluations to the database.

        Args:
            evaluations: List of evaluations to save
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        saved_count = 0
        for evaluation in evaluations:
            # Skip evaluations without verdicts
            if evaluation.verdict is None:
                continue

            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO evaluations (
                        evaluation_id, claim_id, verdict, evaluator_id,
                        notes, confidence_in_verdict, timestamp, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    evaluation.evaluation_id,
                    evaluation.claim_id,
                    evaluation.verdict.value,
                    evaluation.evaluator_id,
                    evaluation.notes,
                    evaluation.confidence_in_verdict,
                    evaluation.timestamp.isoformat() if evaluation.timestamp else datetime.now().isoformat(),
                    str(evaluation.metadata),
                ))
                saved_count += 1
            except Exception as e:
                logger.error(f"Failed to save evaluation {evaluation.evaluation_id}: {e}")

        conn.commit()
        conn.close()

        logger.info(f"Saved {saved_count} evaluations to database")

    def get_evaluations(
        self,
        claim_id: Optional[str] = None,
        evaluator_id: Optional[str] = None,
        verdict: Optional[Verdict] = None
    ) -> List[Evaluation]:
        """
        Retrieve evaluations from the database.

        Args:
            claim_id: Filter by claim ID
            evaluator_id: Filter by evaluator ID
            verdict: Filter by verdict

        Returns:
            List of evaluations matching the filters
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Build query
        query = "SELECT * FROM evaluations WHERE 1=1"
        params = []

        if claim_id:
            query += " AND claim_id = ?"
            params.append(claim_id)

        if evaluator_id:
            query += " AND evaluator_id = ?"
            params.append(evaluator_id)

        if verdict:
            query += " AND verdict = ?"
            params.append(verdict.value)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        conn.close()

        # Convert to Evaluation objects
        evaluations = []
        for row in rows:
            evaluation = Evaluation(
                evaluation_id=row[0],
                claim_id=row[1],
                verdict=Verdict(row[2]) if row[2] else None,
                evaluator_id=row[3],
                notes=row[4] or "",
                confidence_in_verdict=row[5],
                timestamp=datetime.fromisoformat(row[6]) if row[6] else None,
            )
            evaluations.append(evaluation)

        return evaluations

    def get_unevaluated_claims(self, evaluator_id: str = "default") -> List[Claim]:
        """
        Get claims that haven't been evaluated by the specified evaluator.

        Args:
            evaluator_id: Evaluator ID

        Returns:
            List of unevaluated claims
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Find claims without evaluations from this evaluator
        cursor.execute("""
            SELECT c.* FROM claims c
            LEFT JOIN evaluations e ON c.claim_id = e.claim_id AND e.evaluator_id = ?
            WHERE e.evaluation_id IS NULL
            ORDER BY c.created_at
        """, (evaluator_id,))

        rows = cursor.fetchall()
        conn.close()

        # Convert to Claim objects
        claims = []
        for row in rows:
            claim = Claim(
                claim_id=row[0],
                text=row[1],
                claim_type=ClaimType(row[2]),
                discovery_title=row[3],
                confidence=row[4],
                context=row[5],
                source_section=row[6],
            )
            claims.append(claim)

        return claims

    def interactive_evaluation_session(
        self,
        claims: List[Claim],
        evaluator_id: str = "default",
        auto_save: bool = True
    ) -> List[Evaluation]:
        """
        Run an interactive evaluation session for multiple claims.

        Args:
            claims: List of claims to evaluate
            evaluator_id: Evaluator ID
            auto_save: Whether to save evaluations automatically

        Returns:
            List of completed evaluations
        """
        evaluations = []

        print(f"\n{'='*80}")
        print(f"EVALUATION SESSION - {len(claims)} claims to evaluate")
        print(f"Evaluator: {evaluator_id}")
        print(f"{'='*80}\n")

        for i, claim in enumerate(claims, 1):
            print(f"\n[Claim {i}/{len(claims)}]")

            evaluation = self.collect_verdict(claim, evaluator_id, interactive=True)

            if evaluation.verdict:
                evaluations.append(evaluation)

                if auto_save:
                    self.save_evaluation(evaluation)
                    print("✓ Evaluation saved")

            # Ask if user wants to continue
            if i < len(claims):
                continue_input = input("\nContinue to next claim? (y/n): ").strip().lower()
                if continue_input == 'n':
                    print(f"\nSession ended. Evaluated {len(evaluations)}/{len(claims)} claims.")
                    break

        if not auto_save:
            self.save_evaluations(evaluations)

        print(f"\n{'='*80}")
        print(f"SESSION COMPLETE - Evaluated {len(evaluations)} claims")
        print(f"{'='*80}\n")

        return evaluations

    def export_evaluations(self, output_path: Path) -> None:
        """
        Export all evaluations to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        import json

        evaluations = self.get_evaluations()

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump({
                "total_evaluations": len(evaluations),
                "export_timestamp": datetime.now().isoformat(),
                "evaluations": [e.to_dict() for e in evaluations],
            }, f, indent=2)

        logger.info(f"Exported {len(evaluations)} evaluations to {output_path}")
