-- Check the distribution of categories
SELECT category, COUNT(*) as article_count
FROM articles
GROUP BY category
ORDER BY article_count DESC;

-- Check a sample article from each category to verify content migration
SELECT 
    title,
    category,
    overview,
    CASE 
        WHEN category = 'mental_health' THEN 
            json_build_object(
                'prevalence', prevalence,
                'causes_and_mechanisms', causes_and_mechanisms,
                'symptoms_and_impact', symptoms_and_impact
            )
        WHEN category = 'neuroscience' THEN 
            json_build_object(
                'definition', definition,
                'mechanisms', mechanisms,
                'relevance', relevance
            )
        WHEN category = 'interventions' THEN 
            json_build_object(
                'how_it_works', how_it_works,
                'evidence_base', evidence_base,
                'effectiveness', effectiveness
            )
        ELSE '{}'::json
    END as category_specific_fields
FROM articles
WHERE category IS NOT NULL
LIMIT 5;

-- Verify the migration results
SELECT 
    title,
    category,
    overview,
    CASE 
        WHEN category = 'mental_health' THEN jsonb_build_object(
            'prevalence', prevalence,
            'causes_and_mechanisms', causes_and_mechanisms,
            'symptoms_and_impact', symptoms_and_impact,
            'evidence_summary', evidence_summary,
            'practical_takeaways', practical_takeaways
        )
        WHEN category = 'neuroscience' THEN jsonb_build_object(
            'definition', definition,
            'mechanisms', mechanisms,
            'relevance', relevance,
            'key_studies', key_studies
        )
        WHEN category = 'interventions' THEN jsonb_build_object(
            'how_it_works', how_it_works,
            'effectiveness', effectiveness,
            'evidence_base', evidence_base,
            'practical_applications', practical_applications
        )
        WHEN category = 'lab_testing' THEN jsonb_build_object(
            'how_it_works', how_it_works,
            'applications', applications,
            'strengths_and_limitations', strengths_and_limitations
        )
    END as category_specific_fields
FROM articles
ORDER BY category, title; 