"""Cycle Manager - Main orchestration loop for autonomous research"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .world_model import WorldModel, Priority, HypothesisStatus
from .priority_queue import TaskPriorityQueue
from .data_agent import DataAgent
from .literature_agent import LiteratureAgent

logger = logging.getLogger(__name__)


class CycleManager:
    """Manages the autonomous research discovery loop"""

    def __init__(self,
                 world_model: WorldModel,
                 data_agent: DataAgent,
                 literature_agent: LiteratureAgent,
                 max_cycles: int = 20,
                 max_time_hours: float = 6.0,
                 stagnation_cycles: int = 3,
                 tasks_per_cycle: int = 10,
                 max_parallel_tasks: int = 4):
        """
        Initialize the cycle manager

        Args:
            world_model: The knowledge base
            data_agent: Agent for data analysis
            literature_agent: Agent for literature search
            max_cycles: Maximum number of cycles to run
            max_time_hours: Maximum time to run in hours
            stagnation_cycles: Stop if no new findings in this many cycles
            tasks_per_cycle: Number of tasks to execute per cycle
            max_parallel_tasks: Maximum tasks to run in parallel
        """
        self.world_model = world_model
        self.data_agent = data_agent
        self.literature_agent = literature_agent
        self.task_queue = TaskPriorityQueue()

        self.max_cycles = max_cycles
        self.max_time = timedelta(hours=max_time_hours)
        self.stagnation_cycles = stagnation_cycles
        self.tasks_per_cycle = tasks_per_cycle
        self.max_parallel_tasks = max_parallel_tasks

        self.start_time: Optional[datetime] = None
        self.cycle_logs: List[Dict[str, Any]] = []

    def _should_stop(self) -> tuple[bool, str]:
        """Check if stopping conditions are met"""
        stats = self.world_model.get_stats()

        # Check max cycles
        if self.world_model.current_cycle >= self.max_cycles:
            return True, f"Reached maximum cycles ({self.max_cycles})"

        # Check max time
        if self.start_time and datetime.now() - self.start_time >= self.max_time:
            return True, f"Reached maximum time ({self.max_time})"

        # Check stagnation
        cycles_since_finding = self.world_model.current_cycle - self.world_model.last_finding_cycle
        if cycles_since_finding >= self.stagnation_cycles and self.world_model.current_cycle > 0:
            return True, f"No new findings in {cycles_since_finding} cycles"

        return False, ""

    def _generate_initial_tasks(self):
        """Generate initial exploratory tasks"""
        logger.info("Generating initial exploratory tasks")

        # Start with dataset exploration
        task = self.world_model.add_task(
            task_type="data_analysis",
            description="Explore the dataset structure",
            priority=Priority.HIGH
        )
        self.task_queue.push(task)

        # Add correlation analysis
        task = self.world_model.add_task(
            task_type="data_analysis",
            description="Test correlations between features",
            priority=Priority.HIGH
        )
        self.task_queue.push(task)

        # Add distribution analysis
        task = self.world_model.add_task(
            task_type="data_analysis",
            description="Test feature distributions",
            priority=Priority.MEDIUM
        )
        self.task_queue.push(task)

        # Add group difference analysis
        task = self.world_model.add_task(
            task_type="data_analysis",
            description="Test for differences between groups",
            priority=Priority.MEDIUM
        )
        self.task_queue.push(task)

        logger.info(f"Generated {self.task_queue.size()} initial tasks")

    def _generate_hypotheses_from_findings(self, findings: List[Dict[str, Any]]):
        """Generate new hypotheses from recent findings"""
        for finding_data in findings:
            finding_text = finding_data.get("text", "")
            finding_text_lower = finding_text.lower()

            # Generate hypotheses based on findings
            if "correlation" in finding_text_lower and "strong" in finding_text_lower:
                hyp_text = f"The correlation observed in '{finding_text}' may be causal"
                hyp = self.world_model.add_hypothesis(hyp_text, Priority.HIGH, source="data")
                logger.info(f"Generated hypothesis: {hyp_text}")

            elif "significant difference" in finding_text_lower:
                hyp_text = f"The group differences in '{finding_text}' persist across different conditions"
                hyp = self.world_model.add_hypothesis(hyp_text, Priority.HIGH, source="data")
                logger.info(f"Generated hypothesis: {hyp_text}")

            elif "not normally distributed" in finding_text_lower:
                hyp_text = f"Non-normal distribution in '{finding_text}' indicates underlying subgroups"
                hyp = self.world_model.add_hypothesis(hyp_text, Priority.MEDIUM, source="data")
                logger.info(f"Generated hypothesis: {hyp_text}")

            elif "variance" in finding_text_lower:
                hyp_text = f"The variance patterns in '{finding_text}' suggest latent structure"
                hyp = self.world_model.add_hypothesis(hyp_text, Priority.MEDIUM, source="data")
                logger.info(f"Generated hypothesis: {hyp_text}")

    def _generate_tasks_from_hypotheses(self):
        """Generate tasks to test untested hypotheses"""
        untested = self.world_model.get_untested_hypotheses(limit=5)

        for hyp in untested:
            # Generate data analysis task
            task = self.world_model.add_task(
                task_type="data_analysis",
                description=f"Test hypothesis: {hyp.text}",
                priority=Priority.HIGH,
                hypothesis_id=hyp.id
            )
            self.task_queue.push(task)

            # Generate literature search task
            task = self.world_model.add_task(
                task_type="literature_search",
                description=f"Search literature for: {hyp.text}",
                priority=Priority.MEDIUM,
                hypothesis_id=hyp.id
            )
            self.task_queue.push(task)

            # Mark hypothesis as testing
            self.world_model.update_hypothesis_status(hyp.id, HypothesisStatus.TESTING)

        logger.info(f"Generated {len(untested) * 2} tasks from {len(untested)} hypotheses")

    def _generate_followup_tasks(self):
        """Generate follow-up exploratory tasks"""
        # Add PCA if not done yet
        pca_tasks = [t for t in self.world_model.tasks.values()
                    if "pca" in t.description.lower() or "component" in t.description.lower()]
        if not pca_tasks:
            task = self.world_model.add_task(
                task_type="data_analysis",
                description="Perform principal component analysis",
                priority=Priority.LOW
            )
            self.task_queue.push(task)

        # Add exploratory literature searches
        if self.world_model.current_cycle % 3 == 0:  # Every 3 cycles
            task = self.world_model.add_task(
                task_type="literature_search",
                description="Search for relevant statistical methods",
                priority=Priority.LOW
            )
            self.task_queue.push(task)

    def _execute_task(self, task) -> Dict[str, Any]:
        """Execute a single task (runs in thread)"""
        logger.info(f"Executing task {task.id}: {task.description}")

        try:
            task.status = "running"

            if task.type == "data_analysis":
                result = self.data_agent.execute_task(task.description)
            elif task.type == "literature_search":
                # Extract hypothesis if present
                hypothesis = None
                if task.hypothesis_id and task.hypothesis_id in self.world_model.hypotheses:
                    hypothesis = self.world_model.hypotheses[task.hypothesis_id].text
                result = self.literature_agent.execute_task(task.description, hypothesis)
            else:
                result = {"error": f"Unknown task type: {task.type}"}

            task.status = "completed"
            task.completed_at = datetime.now()
            task.result = result

            return {
                "task_id": task.id,
                "status": "success",
                "result": result
            }

        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")
            task.status = "failed"
            task.completed_at = datetime.now()
            return {
                "task_id": task.id,
                "status": "failed",
                "error": str(e)
            }

    def _execute_tasks_parallel(self, tasks: List) -> List[Dict[str, Any]]:
        """Execute multiple tasks in parallel"""
        results = []

        with ThreadPoolExecutor(max_workers=self.max_parallel_tasks) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(self._execute_task, task): task
                            for task in tasks}

            # Collect results as they complete
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Task execution exception: {e}")
                    results.append({
                        "task_id": task.id,
                        "status": "failed",
                        "error": str(e)
                    })

        return results

    def _process_results(self, results: List[Dict[str, Any]]):
        """Process task results and update world model"""
        new_findings = []

        for result in results:
            if result["status"] != "success":
                continue

            task_id = result["task_id"]
            task = self.world_model.tasks.get(task_id)
            if not task:
                continue

            task_result = result["result"]
            findings_text = task_result.get("findings", [])

            # Add findings to world model
            for finding_text in findings_text:
                if finding_text and not finding_text.startswith("Error"):
                    finding = self.world_model.add_finding(
                        text=finding_text,
                        data=task_result,
                        hypothesis_id=task.hypothesis_id,
                        source="data" if task.type == "data_analysis" else "literature"
                    )
                    new_findings.append({
                        "id": finding.id,
                        "text": finding_text,
                        "task_id": task_id
                    })

            # Add papers if literature search
            if task.type == "literature_search" and "papers" in task_result:
                for paper_data in task_result["papers"]:
                    paper = self.world_model.add_paper(
                        title=paper_data["title"],
                        authors=paper_data["authors"],
                        year=paper_data.get("year"),
                        abstract=paper_data["abstract"],
                        url=paper_data.get("url"),
                        relevance_score=paper_data.get("relevance_score", 0.5),
                        related_hypothesis_ids=[task.hypothesis_id] if task.hypothesis_id else []
                    )

            # Update hypothesis status if applicable
            if task.hypothesis_id and task.type == "data_analysis":
                # Simple heuristic: if findings mention "significant" or "strong", support hypothesis
                if any("significant" in f.lower() or "strong" in f.lower() for f in findings_text):
                    self.world_model.update_hypothesis_status(
                        task.hypothesis_id,
                        HypothesisStatus.SUPPORTED,
                        evidence_finding_id=new_findings[0]["id"] if new_findings else None
                    )
                elif any("no" in f.lower() or "not" in f.lower() for f in findings_text):
                    self.world_model.update_hypothesis_status(
                        task.hypothesis_id,
                        HypothesisStatus.INCONCLUSIVE
                    )

        return new_findings

    def _log_cycle(self, cycle: int, tasks_executed: int, findings_count: int):
        """Log cycle information"""
        stats = self.world_model.get_stats()
        elapsed = datetime.now() - self.start_time if self.start_time else timedelta(0)

        log_entry = {
            "cycle": cycle,
            "timestamp": datetime.now().isoformat(),
            "elapsed_time": str(elapsed),
            "tasks_executed": tasks_executed,
            "new_findings": findings_count,
            "stats": stats,
            "queue_stats": self.task_queue.get_stats()
        }

        self.cycle_logs.append(log_entry)

        logger.info(f"Cycle {cycle} complete: {tasks_executed} tasks, {findings_count} new findings")
        logger.info(f"World state: {stats['total_hypotheses']} hypotheses, {stats['total_findings']} findings, {stats['total_papers']} papers")

    def run_cycle(self) -> Dict[str, Any]:
        """Run a single cycle of the discovery loop"""
        cycle = self.world_model.current_cycle
        logger.info(f"Starting cycle {cycle}")

        # Get tasks from queue
        tasks = self.task_queue.pop_batch(self.tasks_per_cycle)

        if not tasks:
            logger.info("No tasks in queue, generating new tasks")
            # Generate tasks from hypotheses
            self._generate_tasks_from_hypotheses()

            # Add follow-up tasks if queue is still empty
            if self.task_queue.is_empty():
                self._generate_followup_tasks()

            # Try again
            tasks = self.task_queue.pop_batch(self.tasks_per_cycle)

        if not tasks:
            logger.warning("No tasks to execute in this cycle")
            return {
                "cycle": cycle,
                "tasks_executed": 0,
                "new_findings": 0
            }

        # Execute tasks in parallel
        logger.info(f"Executing {len(tasks)} tasks in parallel")
        results = self._execute_tasks_parallel(tasks)

        # Process results
        new_findings = self._process_results(results)

        # Generate new hypotheses from findings
        self._generate_hypotheses_from_findings(new_findings)

        # Log cycle
        self._log_cycle(cycle, len(tasks), len(new_findings))

        # Increment cycle counter
        self.world_model.current_cycle += 1

        return {
            "cycle": cycle,
            "tasks_executed": len(tasks),
            "new_findings": len(new_findings),
            "stats": self.world_model.get_stats()
        }

    def run(self) -> Dict[str, Any]:
        """Run the complete discovery loop until stopping condition"""
        logger.info("Starting discovery loop")
        self.start_time = datetime.now()

        # Generate initial tasks
        self._generate_initial_tasks()

        # Run cycles
        while True:
            # Check stopping conditions
            should_stop, reason = self._should_stop()
            if should_stop:
                logger.info(f"Stopping: {reason}")
                break

            # Run cycle
            try:
                cycle_result = self.run_cycle()

                # Progress update
                progress = self._estimate_progress()
                logger.info(f"Progress: {progress['completion_percentage']:.1f}%")

            except Exception as e:
                logger.error(f"Cycle failed with error: {e}")
                # Continue to next cycle despite error

            # Small delay between cycles
            time.sleep(0.5)

        # Final summary
        summary = self._generate_summary()
        logger.info("Discovery loop complete")

        return summary

    def _estimate_progress(self) -> Dict[str, Any]:
        """Estimate completion progress"""
        stats = self.world_model.get_stats()
        cycle_progress = self.world_model.current_cycle / self.max_cycles

        # Also consider time
        if self.start_time:
            elapsed = datetime.now() - self.start_time
            time_progress = elapsed.total_seconds() / self.max_time.total_seconds()
        else:
            time_progress = 0

        # Use maximum of cycle and time progress
        completion = max(cycle_progress, time_progress) * 100

        return {
            "current_cycle": self.world_model.current_cycle,
            "max_cycles": self.max_cycles,
            "completion_percentage": min(100, completion),
            "total_hypotheses": stats["total_hypotheses"],
            "total_findings": stats["total_findings"],
            "total_papers": stats["total_papers"]
        }

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate final summary of the discovery loop"""
        stats = self.world_model.get_stats()
        elapsed = datetime.now() - self.start_time if self.start_time else timedelta(0)

        summary = {
            "total_cycles": self.world_model.current_cycle,
            "elapsed_time": str(elapsed),
            "statistics": stats,
            "cycle_logs": self.cycle_logs,
            "key_findings": [
                f.text for f in self.world_model.get_recent_findings(limit=10)
            ],
            "supported_hypotheses": [
                h.text for h in self.world_model.hypotheses.values()
                if h.status == HypothesisStatus.SUPPORTED
            ],
            "top_papers": [
                {
                    "title": p.title,
                    "authors": p.authors,
                    "relevance": p.relevance_score
                }
                for p in sorted(self.world_model.papers.values(),
                              key=lambda x: x.relevance_score, reverse=True)[:5]
            ]
        }

        return summary
