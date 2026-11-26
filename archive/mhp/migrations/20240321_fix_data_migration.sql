-- First, ensure all articles have a category
UPDATE articles
SET category = CASE
    WHEN title ILIKE '%neuron%' OR 
         title ILIKE '%synapse%' OR 
         title ILIKE '%neuroplasticity%' OR 
         title ILIKE '%brain circuit%' OR
         title ILIKE '%neuroscience%'
        THEN 'neuroscience'::article_category
    WHEN title ILIKE '%depression%' OR 
         title ILIKE '%anxiety%' OR 
         title ILIKE '%adhd%' OR 
         title ILIKE '%ptsd%' OR
         title ILIKE '%disorder%'
        THEN 'mental_health'::article_category
    WHEN title ILIKE '%cognitive bias%' OR 
         title ILIKE '%personality%' OR 
         title ILIKE '%developmental%'
        THEN 'psychology'::article_category
    WHEN title ILIKE '%sleep%' OR 
         title ILIKE '%nutrition%' OR 
         title ILIKE '%exercise%' OR 
         title ILIKE '%cognitive training%'
        THEN 'brain_health'::article_category
    WHEN title ILIKE '%autism%' OR 
         title ILIKE '%dyslexia%' OR 
         title ILIKE '%dyspraxia%'
        THEN 'neurodiversity'::article_category
    WHEN title ILIKE '%cbt%' OR 
         title ILIKE '%medication%' OR 
         title ILIKE '%mindfulness%' OR 
         title ILIKE '%psychedelic%' OR
         title ILIKE '%therapy%' OR
         title ILIKE '%treatment%'
        THEN 'interventions'::article_category
    WHEN title ILIKE '%diet%' OR 
         title ILIKE '%social connection%' OR
         title ILIKE '%lifestyle%'
        THEN 'lifestyle_factors'::article_category
    WHEN title ILIKE '%brain imaging%' OR 
         title ILIKE '%cognitive test%' OR
         title ILIKE '%diagnosis%' OR
         title ILIKE '%assessment%'
        THEN 'lab_testing'::article_category
    ELSE 'mental_health'::article_category
END;

-- Then, migrate the content for each category type
-- Mental Health Articles
UPDATE articles
SET 
    overview = COALESCE(content->>'overview', overview),
    prevalence = COALESCE(content->>'prevalence', prevalence),
    causes_and_mechanisms = COALESCE(content->>'mechanisms', causes_and_mechanisms),
    symptoms_and_impact = COALESCE(content->>'symptoms', symptoms_and_impact),
    evidence_summary = COALESCE(content->>'keyEvidence', evidence_summary),
    practical_takeaways = COALESCE(content->>'practicalTakeaways', practical_takeaways),
    common_myths = COALESCE(content->>'commonMyths', common_myths),
    future_directions = COALESCE(content->>'futureDirections', future_directions),
    references_and_resources = COALESCE(content->>'references', references_and_resources)
WHERE category = 'mental_health';

-- Neuroscience Articles
UPDATE articles
SET 
    overview = COALESCE(content->>'overview', overview),
    definition = COALESCE(content->>'overview', definition),
    mechanisms = COALESCE(content->>'mechanisms', mechanisms),
    relevance = COALESCE(content->>'relevance', relevance),
    key_studies = COALESCE(content->>'keyEvidence', key_studies),
    common_misconceptions = COALESCE(content->>'commonMyths', common_misconceptions),
    practical_implications = COALESCE(content->>'practicalTakeaways', practical_implications),
    future_directions = COALESCE(content->>'futureDirections', future_directions),
    references_and_resources = COALESCE(content->>'references', references_and_resources)
WHERE category = 'neuroscience';

-- Intervention Articles
UPDATE articles
SET 
    overview = COALESCE(content->>'overview', overview),
    how_it_works = COALESCE(content->>'mechanisms', how_it_works),
    evidence_base = COALESCE(content->>'keyEvidence', evidence_base),
    effectiveness = COALESCE(content->>'effectiveness', effectiveness),
    practical_applications = COALESCE(content->>'practicalTakeaways', practical_applications),
    common_myths = COALESCE(content->>'commonMyths', common_myths),
    risks_and_limitations = COALESCE(content->>'risks', risks_and_limitations),
    future_directions = COALESCE(content->>'futureDirections', future_directions),
    references_and_resources = COALESCE(content->>'references', references_and_resources)
WHERE category = 'interventions';

-- Lab Testing Articles
UPDATE articles
SET 
    overview = COALESCE(content->>'overview', overview),
    how_it_works = COALESCE(content->>'mechanisms', how_it_works),
    applications = COALESCE(content->>'applications', applications),
    strengths_and_limitations = COALESCE(content->>'strengths', strengths_and_limitations),
    risks_and_limitations = COALESCE(content->>'risks', risks_and_limitations),
    future_directions = COALESCE(content->>'futureDirections', future_directions),
    references_and_resources = COALESCE(content->>'references', references_and_resources)
WHERE category = 'lab_testing';

-- Add default content for empty fields
UPDATE articles
SET 
    future_directions = COALESCE(future_directions, 'Research in this area is ongoing. Future studies will help us better understand...'),
    references_and_resources = COALESCE(references_and_resources, 'For more information, please consult the following resources...')
WHERE future_directions IS NULL OR references_and_resources IS NULL; 