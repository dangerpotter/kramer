"""Configuration for Kramer"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Configuration for the discovery loop"""

    # Discovery loop parameters
    max_cycles: int = 20
    max_time_hours: float = 6.0
    stagnation_cycles: int = 3
    tasks_per_cycle: int = 10
    max_parallel_tasks: int = 4

    # Logging
    log_level: str = "INFO"
    log_file: str = "kramer.log"

    # Output
    output_dir: str = "output"
    save_world_model: bool = True

    # RAG (Retrieval Augmented Generation) settings
    use_full_text: bool = True
    max_papers_to_process: int = 20
    rag_persist_dir: str = "data/rag_db"
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50


def setup_logging(config: Config):
    """Setup logging configuration"""

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, config.log_level.upper()))

    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if config.log_file:
        file_handler = logging.FileHandler(config.log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def create_output_directory(config: Config):
    """Create output directory if it doesn't exist"""
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path
