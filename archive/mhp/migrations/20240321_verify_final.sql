-- Verify the final migration results
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
            'practical_takeaways', practical_takeaways,
            'common_myths', common_myths
        )
        WHEN category = 'neuroscience' THEN jsonb_build_object(
            'definition', definition,
            'mechanisms', mechanisms,
            'relevance', relevance,
            'key_studies', key_studies,
            'common_misconceptions', common_misconceptions,
            'practical_implications', practical_implications
        )
    END as category_specific_fields
FROM articles
ORDER BY category, title; 