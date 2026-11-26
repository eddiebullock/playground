-- Add risk_factors category to the article_category enum
ALTER TYPE article_category ADD VALUE IF NOT EXISTS 'risk_factors';

-- Update articles that should be categorized as risk factors
UPDATE articles
SET category = 'risk_factors'::article_category
WHERE (
    title ILIKE '%risk%' OR 
    title ILIKE '%factor%' OR 
    title ILIKE '%cause%' OR 
    title ILIKE '%trigger%' OR
    title ILIKE '%vulnerability%' OR
    title ILIKE '%predisposition%' OR
    title ILIKE '%etiology%'
) AND category IN ('mental_health', 'neuroscience', 'psychology');

-- Add risk factors specific columns
ALTER TABLE articles
ADD COLUMN IF NOT EXISTS risk_level text,
ADD COLUMN IF NOT EXISTS modifiable_factors text,
ADD COLUMN IF NOT EXISTS protective_factors text;

-- Update content mapping for risk factors articles
UPDATE articles
SET 
    overview = COALESCE(content->>'overview', overview),
    risk_level = COALESCE(content->>'riskLevel', risk_level),
    modifiable_factors = COALESCE(content->>'modifiableFactors', modifiable_factors),
    protective_factors = COALESCE(content->>'protectiveFactors', protective_factors),
    causes_and_mechanisms = COALESCE(content->>'mechanisms', causes_and_mechanisms),
    evidence_summary = COALESCE(content->>'keyEvidence', evidence_summary),
    practical_takeaways = COALESCE(content->>'practicalTakeaways', practical_takeaways),
    common_myths = COALESCE(content->>'commonMyths', common_myths),
    future_directions = COALESCE(content->>'futureDirections', future_directions),
    references_and_resources = COALESCE(content->>'references', references_and_resources)
WHERE category = 'risk_factors'; 