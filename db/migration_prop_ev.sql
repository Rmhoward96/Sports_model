-- =============================================================================
-- Migration: probability + EV framing for props (run once in Supabase SQL Editor).
-- =============================================================================
-- Adds the model's raw outcome distribution to prop_predictions so P(over) can be
-- evaluated at the BOOK's actual line (not a fixed default), and the probability /
-- EV columns to prediction_results. Safe to re-run (IF NOT EXISTS).

ALTER TABLE prop_predictions   ADD COLUMN IF NOT EXISTS dist         JSONB;

ALTER TABLE prediction_results ADD COLUMN IF NOT EXISTS model_prob   REAL;
ALTER TABLE prediction_results ADD COLUMN IF NOT EXISTS market_prob  REAL;
ALTER TABLE prediction_results ADD COLUMN IF NOT EXISTS ev           REAL;
