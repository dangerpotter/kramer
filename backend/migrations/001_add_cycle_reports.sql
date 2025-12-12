-- Migration: Add cycle_reports table
-- Date: 2025-12-11
-- Description: Stores reports generated at the end of each discovery cycle

CREATE TABLE IF NOT EXISTS cycle_reports (
    id VARCHAR(36) PRIMARY KEY,
    cycle_id VARCHAR(36) NOT NULL UNIQUE REFERENCES cycles(id) ON DELETE CASCADE,
    discovery_id VARCHAR(36) NOT NULL REFERENCES discoveries(id) ON DELETE CASCADE,

    -- Report content
    summary TEXT NOT NULL,
    full_content TEXT NOT NULL,

    -- Metrics snapshot
    tasks_completed INTEGER DEFAULT 0,
    findings_count INTEGER DEFAULT 0,
    hypotheses_count INTEGER DEFAULT 0,
    papers_count INTEGER DEFAULT 0,
    budget_used FLOAT DEFAULT 0.0,
    generation_cost FLOAT DEFAULT 0.0,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_cycle_report_discovery ON cycle_reports(discovery_id);
CREATE INDEX IF NOT EXISTS idx_cycle_report_cycle ON cycle_reports(cycle_id);
