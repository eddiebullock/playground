-- Check the content of the JSONB field
SELECT 
    title,
    category,
    content
FROM articles
ORDER BY category, title; 