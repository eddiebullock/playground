-- Create article category enum type if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'article_category') THEN
        CREATE TYPE article_category AS ENUM (
            'mental_health',
            'neuroscience',
            'psychology',
            'brain_health',
            'neurodiversity',
            'interventions',
            'lifestyle_factors',
            'lab_testing'
        );
    END IF;
END $$;

-- Step 1: Add all new columns
ALTER TABLE articles
ADD COLUMN IF NOT EXISTS category article_category,
ADD COLUMN IF NOT EXISTS overview text,
ADD COLUMN IF NOT EXISTS prevalence text,
ADD COLUMN IF NOT EXISTS causes_and_mechanisms text,
ADD COLUMN IF NOT EXISTS symptoms_and_impact text,
ADD COLUMN IF NOT EXISTS evidence_summary text,
ADD COLUMN IF NOT EXISTS common_myths text,
ADD COLUMN IF NOT EXISTS practical_takeaways text,
ADD COLUMN IF NOT EXISTS future_directions text,
ADD COLUMN IF NOT EXISTS references_and_resources text,
ADD COLUMN IF NOT EXISTS definition text,
ADD COLUMN IF NOT EXISTS mechanisms text,
ADD COLUMN IF NOT EXISTS relevance text,
ADD COLUMN IF NOT EXISTS key_studies text,
ADD COLUMN IF NOT EXISTS common_misconceptions text,
ADD COLUMN IF NOT EXISTS practical_implications text,
ADD COLUMN IF NOT EXISTS effectiveness text,
ADD COLUMN IF NOT EXISTS risks_and_limitations text,
ADD COLUMN IF NOT EXISTS how_it_works text,
ADD COLUMN IF NOT EXISTS applications text,
ADD COLUMN IF NOT EXISTS strengths_and_limitations text,
ADD COLUMN IF NOT EXISTS evidence_base text,
ADD COLUMN IF NOT EXISTS practical_applications text;

-- Step 2: Create a function to migrate existing articles
CREATE OR REPLACE FUNCTION migrate_article_categories()
RETURNS void AS $$
BEGIN
    -- First, update categories based on titles
    UPDATE articles
    SET category = CASE
        WHEN title ILIKE '%depression%' OR title ILIKE '%anxiety%' OR title ILIKE '%adhd%' OR title ILIKE '%ptsd%'
            THEN 'mental_health'::article_category
        WHEN title ILIKE '%neuroplasticity%' OR title ILIKE '%neurotransmitter%' OR title ILIKE '%brain circuit%'
            THEN 'neuroscience'::article_category
        WHEN title ILIKE '%cognitive bias%' OR title ILIKE '%personality%' OR title ILIKE '%developmental%'
            THEN 'psychology'::article_category
        WHEN title ILIKE '%sleep%' OR title ILIKE '%nutrition%' OR title ILIKE '%exercise%' OR title ILIKE '%cognitive training%'
            THEN 'brain_health'::article_category
        WHEN title ILIKE '%autism%' OR title ILIKE '%dyslexia%' OR title ILIKE '%dyspraxia%'
            THEN 'neurodiversity'::article_category
        WHEN title ILIKE '%cbt%' OR title ILIKE '%medication%' OR title ILIKE '%mindfulness%' OR title ILIKE '%psychedelic%'
            THEN 'interventions'::article_category
        WHEN title ILIKE '%diet%' OR title ILIKE '%social connection%'
            THEN 'lifestyle_factors'::article_category
        WHEN title ILIKE '%brain imaging%' OR title ILIKE '%cognitive test%'
            THEN 'lab_testing'::article_category
        ELSE 'mental_health'::article_category -- Default category
    END;

    -- Then, update content for each category type
    -- Mental Health Articles
    UPDATE articles
    SET 
        overview = content->>'overview',
        prevalence = content->>'prevalence',
        causes_and_mechanisms = content->>'mechanisms',
        symptoms_and_impact = content->>'symptoms',
        evidence_summary = content->>'keyEvidence',
        practical_takeaways = content->>'practicalTakeaways'
    WHERE category = 'mental_health';

    -- Neuroscience Articles
    UPDATE articles
    SET 
        definition = content->>'overview',
        mechanisms = content->>'mechanisms',
        relevance = content->>'relevance',
        key_studies = content->>'keyEvidence',
        common_misconceptions = content->>'commonMyths',
        practical_implications = content->>'practicalTakeaways'
    WHERE category = 'neuroscience';

    -- Intervention Articles
    UPDATE articles
    SET 
        overview = content->>'overview',
        how_it_works = content->>'mechanisms',
        evidence_base = content->>'keyEvidence',
        effectiveness = content->>'effectiveness',
        practical_applications = content->>'practicalTakeaways',
        common_myths = content->>'commonMyths',
        risks_and_limitations = content->>'risks'
    WHERE category = 'interventions';

    -- Lifestyle Articles
    UPDATE articles
    SET 
        overview = content->>'overview',
        mechanisms = content->>'mechanisms',
        evidence_summary = content->>'keyEvidence',
        practical_takeaways = content->>'practicalTakeaways',
        risks_and_limitations = content->>'risks'
    WHERE category = 'lifestyle_factors';

    -- Lab Testing Articles
    UPDATE articles
    SET 
        overview = content->>'overview',
        how_it_works = content->>'mechanisms',
        applications = content->>'applications',
        strengths_and_limitations = content->>'strengths',
        risks_and_limitations = content->>'risks'
    WHERE category = 'lab_testing';
END;
$$ LANGUAGE plpgsql;

-- Execute the migration
SELECT migrate_article_categories();

-- Drop the migration function
DROP FUNCTION migrate_article_categories();

-- Add NOT NULL constraint to category after migration
ALTER TABLE articles
ALTER COLUMN category SET NOT NULL; 