-- 1. Fix specific article categorizations
UPDATE articles
SET category = 'neuroscience'::article_category
WHERE title IN (
    'Brain Plasticity and Cognitive Health',
    'Basic Neuroscience: Neurons and Synapses'
);

-- 2. Fix content mapping for neuroscience articles
UPDATE articles
SET 
    overview = content->>'overview',
    definition = content->>'overview',
    mechanisms = content->>'mechanisms',
    relevance = content->>'safety',
    key_studies = (
        SELECT jsonb_agg(jsonb_build_object(
            'title', evidence->>'title',
            'link', evidence->>'link'
        ))
        FROM jsonb_array_elements(content->'key_evidence') evidence
    )::text,
    common_misconceptions = (
        SELECT jsonb_agg(jsonb_build_object(
            'question', faq->>'q',
            'answer', faq->>'a'
        ))
        FROM jsonb_array_elements(content->'faq') faq
    )::text,
    practical_implications = (
        SELECT jsonb_agg(takeaway)
        FROM jsonb_array_elements_text(content->'practical_takeaways') takeaway
    )::text
WHERE category = 'neuroscience';

-- 3. Fix content mapping for mental health articles
UPDATE articles
SET 
    overview = content->>'overview',
    prevalence = content->>'safety',
    causes_and_mechanisms = content->>'mechanisms',
    symptoms_and_impact = content->>'safety',
    evidence_summary = (
        SELECT jsonb_agg(jsonb_build_object(
            'title', evidence->>'title',
            'link', evidence->>'link'
        ))
        FROM jsonb_array_elements(content->'key_evidence') evidence
    )::text,
    practical_takeaways = (
        SELECT jsonb_agg(takeaway)
        FROM jsonb_array_elements_text(content->'practical_takeaways') takeaway
    )::text,
    common_myths = (
        SELECT jsonb_agg(jsonb_build_object(
            'question', faq->>'q',
            'answer', faq->>'a'
        ))
        FROM jsonb_array_elements(content->'faq') faq
    )::text
WHERE category = 'mental_health';

-- 4. Add default content for empty fields
UPDATE articles
SET 
    future_directions = COALESCE(future_directions, 'Research in this area is ongoing. Future studies will help us better understand...'),
    references_and_resources = COALESCE(references_and_resources, 'For more information, please consult the following resources...')
WHERE future_directions IS NULL OR references_and_resources IS NULL;

-- 5. Verify no articles are uncategorized
UPDATE articles
SET category = 'mental_health'::article_category
WHERE category IS NULL; 