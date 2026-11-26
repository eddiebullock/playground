-- Drop the existing function
DROP FUNCTION IF EXISTS public.search_articles;

-- Recreate the function with proper type casting
CREATE OR REPLACE FUNCTION public.search_articles(
    query_embedding vector,
    similarity_threshold float DEFAULT 0.7,
    match_count int DEFAULT 5
)
RETURNS TABLE (
    id UUID,
    title TEXT,
    summary TEXT,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        articles.id,
        articles.title,
        articles.summary,
        1 - (articles.title_embedding <=> query_embedding::vector) as similarity
    FROM articles
    WHERE articles.title_embedding IS NOT NULL
    AND 1 - (articles.title_embedding <=> query_embedding::vector) > similarity_threshold
    UNION
    SELECT
        articles.id,
        articles.title,
        articles.summary,
        1 - (articles.content_embedding <=> query_embedding::vector) as similarity
    FROM articles
    WHERE articles.content_embedding IS NOT NULL
    AND 1 - (articles.content_embedding <=> query_embedding::vector) > similarity_threshold
    UNION
    SELECT
        articles.id,
        articles.title,
        articles.summary,
        1 - (articles.summary_embedding <=> query_embedding::vector) as similarity
    FROM articles
    WHERE articles.summary_embedding IS NOT NULL
    AND 1 - (articles.summary_embedding <=> query_embedding::vector) > similarity_threshold
    ORDER BY similarity DESC
    LIMIT match_count;
END;
$$; 