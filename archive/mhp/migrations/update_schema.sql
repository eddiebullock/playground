-- Add new columns for neurodiversity articles
ALTER TABLE articles
ADD COLUMN IF NOT EXISTS neurodiversity_perspective TEXT,
ADD COLUMN IF NOT EXISTS common_strengths_and_challenges TEXT,
ADD COLUMN IF NOT EXISTS prevalence_and_demographics TEXT,
ADD COLUMN IF NOT EXISTS mechanisms_and_understanding TEXT,
ADD COLUMN IF NOT EXISTS evidence_summary TEXT,
ADD COLUMN IF NOT EXISTS common_misconceptions TEXT,
ADD COLUMN IF NOT EXISTS practical_takeaways TEXT,
ADD COLUMN IF NOT EXISTS lived_experience TEXT;

-- Add new columns for psychology articles
ALTER TABLE articles
ADD COLUMN IF NOT EXISTS definition TEXT,
ADD COLUMN IF NOT EXISTS core_principles TEXT,
ADD COLUMN IF NOT EXISTS relevance TEXT,
ADD COLUMN IF NOT EXISTS key_studies_and_theories TEXT,
ADD COLUMN IF NOT EXISTS practical_applications TEXT; 