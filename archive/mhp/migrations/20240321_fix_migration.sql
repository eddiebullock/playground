-- First, fix the category classification
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
    ELSE category -- Keep existing category if no match
END;

-- Then, fix the content migration
UPDATE articles
SET 
    -- Common fields for all categories
    overview = COALESCE(content->>'overview', overview),
    future_directions = COALESCE(content->>'futureDirections', future_directions),
    references_and_resources = COALESCE(content->>'references', references_and_resources),
    
    -- Mental Health specific fields
    prevalence = CASE 
        WHEN category = 'mental_health' THEN COALESCE(content->>'prevalence', prevalence)
        ELSE prevalence
    END,
    causes_and_mechanisms = CASE 
        WHEN category = 'mental_health' THEN COALESCE(content->>'mechanisms', causes_and_mechanisms)
        ELSE causes_and_mechanisms
    END,
    symptoms_and_impact = CASE 
        WHEN category = 'mental_health' THEN COALESCE(content->>'symptoms', symptoms_and_impact)
        ELSE symptoms_and_impact
    END,
    evidence_summary = CASE 
        WHEN category = 'mental_health' THEN COALESCE(content->>'keyEvidence', evidence_summary)
        ELSE evidence_summary
    END,
    practical_takeaways = CASE 
        WHEN category = 'mental_health' THEN COALESCE(content->>'practicalTakeaways', practical_takeaways)
        ELSE practical_takeaways
    END,
    
    -- Neuroscience specific fields
    definition = CASE 
        WHEN category = 'neuroscience' THEN COALESCE(content->>'overview', definition)
        ELSE definition
    END,
    mechanisms = CASE 
        WHEN category = 'neuroscience' THEN COALESCE(content->>'mechanisms', mechanisms)
        ELSE mechanisms
    END,
    relevance = CASE 
        WHEN category = 'neuroscience' THEN COALESCE(content->>'relevance', relevance)
        ELSE relevance
    END,
    key_studies = CASE 
        WHEN category = 'neuroscience' THEN COALESCE(content->>'keyEvidence', key_studies)
        ELSE key_studies
    END,
    
    -- Intervention specific fields
    how_it_works = CASE 
        WHEN category = 'interventions' THEN COALESCE(content->>'mechanisms', how_it_works)
        ELSE how_it_works
    END,
    evidence_base = CASE 
        WHEN category = 'interventions' THEN COALESCE(content->>'keyEvidence', evidence_base)
        ELSE evidence_base
    END,
    effectiveness = CASE 
        WHEN category = 'interventions' THEN COALESCE(content->>'effectiveness', effectiveness)
        ELSE effectiveness
    END,
    practical_applications = CASE 
        WHEN category = 'interventions' THEN COALESCE(content->>'practicalTakeaways', practical_applications)
        ELSE practical_applications
    END,
    
    -- Lab Testing specific fields
    applications = CASE 
        WHEN category = 'lab_testing' THEN COALESCE(content->>'applications', applications)
        ELSE applications
    END,
    strengths_and_limitations = CASE 
        WHEN category = 'lab_testing' THEN COALESCE(content->>'strengths', strengths_and_limitations)
        ELSE strengths_and_limitations
    END,
    risks_and_limitations = CASE 
        WHEN category = 'lab_testing' THEN COALESCE(content->>'risks', risks_and_limitations)
        ELSE risks_and_limitations
    END; 