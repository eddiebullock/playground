-- First, let's check the table structure
SELECT 
    column_name, 
    data_type, 
    is_nullable,
    column_default
FROM information_schema.columns 
WHERE table_name = 'articles'
ORDER BY ordinal_position;

-- First, clean up any existing test articles
DELETE FROM articles WHERE title LIKE 'Test%';

-- Now let's try inserting test articles for each category
-- Test insert for a psychology article
INSERT INTO articles (
    title,
    slug,
    category,
    overview,
    definition,
    mechanisms,
    relevance,
    key_studies,
    common_misconceptions,
    practical_implications,
    future_directions,
    references_and_resources
) VALUES (
    'Test Psychology Article',
    'test-psychology-article-2024',
    'psychology',
    'This is a test overview for a psychology article.',
    'This is test definition data.',
    'This is test mechanisms data.',
    'This is test relevance data.',
    'This is test key studies data.',
    'This is test common misconceptions data.',
    'This is test practical implications data.',
    'This is test future directions data.',
    'This is test references and resources data.'
);

-- Test insert for a brain health article
INSERT INTO articles (
    title,
    slug,
    category,
    overview,
    definition,
    mechanisms,
    relevance,
    key_studies,
    common_misconceptions,
    practical_implications,
    future_directions,
    references_and_resources
) VALUES (
    'Test Brain Health Article',
    'test-brain-health-article-2024',
    'brain_health',
    'This is a test overview for a brain health article.',
    'This is test definition data.',
    'This is test mechanisms data.',
    'This is test relevance data.',
    'This is test key studies data.',
    'This is test common misconceptions data.',
    'This is test practical implications data.',
    'This is test future directions data.',
    'This is test references and resources data.'
);

-- Test insert for a neurodiversity article
INSERT INTO articles (
    title,
    slug,
    category,
    overview,
    prevalence,
    causes_and_mechanisms,
    symptoms_and_impact,
    evidence_summary,
    practical_takeaways,
    common_myths,
    future_directions,
    references_and_resources
) VALUES (
    'Test Neurodiversity Article',
    'test-neurodiversity-article-2024',
    'neurodiversity',
    'This is a test overview for a neurodiversity article.',
    'This is test prevalence data.',
    'This is test causes and mechanisms data.',
    'This is test symptoms and impact data.',
    'This is test evidence summary data.',
    'This is test practical takeaways data.',
    'This is test common myths data.',
    'This is test future directions data.',
    'This is test references and resources data.'
);

-- Test insert for an interventions article
INSERT INTO articles (
    title,
    slug,
    category,
    overview,
    how_it_works,
    evidence_base,
    effectiveness,
    practical_applications,
    common_myths,
    risks_and_limitations,
    future_directions,
    references_and_resources
) VALUES (
    'Test Interventions Article',
    'test-interventions-article-2024',
    'interventions',
    'This is a test overview for an interventions article.',
    'This is test how it works data.',
    'This is test evidence base data.',
    'This is test effectiveness data.',
    'This is test practical applications data.',
    'This is test common myths data.',
    'This is test risks and limitations data.',
    'This is test future directions data.',
    'This is test references and resources data.'
);

-- Test insert for a lifestyle factors article
INSERT INTO articles (
    title,
    slug,
    category,
    overview,
    how_it_works,
    evidence_base,
    effectiveness,
    practical_applications,
    common_myths,
    risks_and_limitations,
    future_directions,
    references_and_resources
) VALUES (
    'Test Lifestyle Factors Article',
    'test-lifestyle-factors-article-2024',
    'lifestyle_factors',
    'This is a test overview for a lifestyle factors article.',
    'This is test how it works data.',
    'This is test evidence base data.',
    'This is test effectiveness data.',
    'This is test practical applications data.',
    'This is test common myths data.',
    'This is test risks and limitations data.',
    'This is test future directions data.',
    'This is test references and resources data.'
);

-- Test insert for a lab testing article
INSERT INTO articles (
    title,
    slug,
    category,
    overview,
    how_it_works,
    applications,
    strengths_and_limitations,
    risks_and_limitations,
    future_directions,
    references_and_resources
) VALUES (
    'Test Lab Testing Article',
    'test-lab-testing-article-2024',
    'lab_testing',
    'This is a test overview for a lab testing article.',
    'This is test how it works data.',
    'This is test applications data.',
    'This is test strengths and limitations data.',
    'This is test risks and limitations data.',
    'This is test future directions data.',
    'This is test references and resources data.'
);

-- Verify all test inserts
SELECT 
    title,
    category,
    overview,
    CASE 
        WHEN category = 'mental_health' OR category = 'neurodiversity' THEN jsonb_build_object(
            'prevalence', prevalence,
            'causes_and_mechanisms', causes_and_mechanisms,
            'symptoms_and_impact', symptoms_and_impact,
            'evidence_summary', evidence_summary,
            'practical_takeaways', practical_takeaways,
            'common_myths', common_myths
        )
        WHEN category = 'neuroscience' OR category = 'psychology' OR category = 'brain_health' THEN jsonb_build_object(
            'definition', definition,
            'mechanisms', mechanisms,
            'relevance', relevance,
            'key_studies', key_studies,
            'common_misconceptions', common_misconceptions,
            'practical_implications', practical_implications
        )
        WHEN category = 'interventions' OR category = 'lifestyle_factors' THEN jsonb_build_object(
            'how_it_works', how_it_works,
            'evidence_base', evidence_base,
            'effectiveness', effectiveness,
            'practical_applications', practical_applications,
            'common_myths', common_myths,
            'risks_and_limitations', risks_and_limitations
        )
        WHEN category = 'lab_testing' THEN jsonb_build_object(
            'how_it_works', how_it_works,
            'applications', applications,
            'strengths_and_limitations', strengths_and_limitations,
            'risks_and_limitations', risks_and_limitations
        )
    END as category_specific_fields
FROM articles
WHERE title LIKE 'Test%'
ORDER BY category, title; 