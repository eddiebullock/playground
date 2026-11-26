-- 1. Check category distribution
SELECT 
    category,
    COUNT(*) as article_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM articles), 2) as percentage
FROM articles
GROUP BY category
ORDER BY article_count DESC;

-- 2. Check for any uncategorized articles
SELECT COUNT(*) as uncategorized_count
FROM articles
WHERE category IS NULL;

-- 3. Check content completeness for each category
SELECT 
    category,
    COUNT(*) as total_articles,
    COUNT(CASE WHEN overview IS NOT NULL THEN 1 END) as has_overview,
    COUNT(CASE WHEN future_directions IS NOT NULL THEN 1 END) as has_future_directions,
    COUNT(CASE WHEN references_and_resources IS NOT NULL THEN 1 END) as has_references,
    -- Category specific fields
    COUNT(CASE 
        WHEN category = 'mental_health' AND prevalence IS NOT NULL THEN 1 
        WHEN category = 'neuroscience' AND definition IS NOT NULL THEN 1
        WHEN category = 'interventions' AND how_it_works IS NOT NULL THEN 1
        WHEN category = 'lab_testing' AND applications IS NOT NULL THEN 1
    END) as has_category_specific_content
FROM articles
GROUP BY category
ORDER BY total_articles DESC;

-- 4. Sample articles from each category to verify content
SELECT 
    title,
    category,
    overview,
    CASE 
        WHEN category = 'mental_health' THEN 
            json_build_object(
                'prevalence', prevalence,
                'causes_and_mechanisms', causes_and_mechanisms,
                'symptoms_and_impact', symptoms_and_impact,
                'evidence_summary', evidence_summary,
                'practical_takeaways', practical_takeaways
            )
        WHEN category = 'neuroscience' THEN 
            json_build_object(
                'definition', definition,
                'mechanisms', mechanisms,
                'relevance', relevance,
                'key_studies', key_studies
            )
        WHEN category = 'interventions' THEN 
            json_build_object(
                'how_it_works', how_it_works,
                'evidence_base', evidence_base,
                'effectiveness', effectiveness,
                'practical_applications', practical_applications
            )
        WHEN category = 'lab_testing' THEN 
            json_build_object(
                'applications', applications,
                'strengths_and_limitations', strengths_and_limitations,
                'risks_and_limitations', risks_and_limitations
            )
        ELSE '{}'::json
    END as category_specific_fields
FROM articles
WHERE category IS NOT NULL
ORDER BY category, title
LIMIT 10; 