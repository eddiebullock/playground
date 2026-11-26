-- Enable the pgvector extension if not already enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- Add embedding columns to articles table
ALTER TABLE public.articles
    -- Add column for title embedding (1536 dimensions for OpenAI embeddings)
    ADD COLUMN title_embedding vector(1536),
    -- Add column for content embedding
    ADD COLUMN content_embedding vector(1536),
    -- Add column for summary embedding
    ADD COLUMN summary_embedding vector(1536);

-- Create indexes for vector similarity search
CREATE INDEX idx_articles_title_embedding ON public.articles 
    USING ivfflat (title_embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX idx_articles_content_embedding ON public.articles 
    USING ivfflat (content_embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX idx_articles_summary_embedding ON public.articles 
    USING ivfflat (summary_embedding vector_cosine_ops)
    WITH (lists = 100);

-- Create a function to update embeddings
CREATE OR REPLACE FUNCTION public.update_article_embeddings()
RETURNS TRIGGER AS $$
BEGIN
    -- This function will be called by your application code
    -- to update the embeddings when the article is created or updated
    -- The actual embedding generation should happen in your application
    -- using an embedding model (e.g., OpenAI's text-embedding-ada-002)
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create a trigger to update embeddings
CREATE TRIGGER update_article_embeddings_trigger
    BEFORE INSERT OR UPDATE OF title, content, summary
    ON public.articles
    FOR EACH ROW
    EXECUTE FUNCTION public.update_article_embeddings();

-- Add comments
COMMENT ON COLUMN public.articles.title_embedding IS 'Vector embedding of the article title for semantic search';
COMMENT ON COLUMN public.articles.content_embedding IS 'Vector embedding of the article content for semantic search';
COMMENT ON COLUMN public.articles.summary_embedding IS 'Vector embedding of the article summary for semantic search';

-- Create a function to search articles by semantic similarity
CREATE OR REPLACE FUNCTION public.search_articles(
    query_embedding vector(1536),
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
        1 - (articles.title_embedding <=> query_embedding) as similarity
    FROM articles
    WHERE 1 - (articles.title_embedding <=> query_embedding) > similarity_threshold
    UNION
    SELECT
        articles.id,
        articles.title,
        articles.summary,
        1 - (articles.content_embedding <=> query_embedding) as similarity
    FROM articles
    WHERE 1 - (articles.content_embedding <=> query_embedding) > similarity_threshold
    UNION
    SELECT
        articles.id,
        articles.title,
        articles.summary,
        1 - (articles.summary_embedding <=> query_embedding) as similarity
    FROM articles
    WHERE 1 - (articles.summary_embedding <=> query_embedding) > similarity_threshold
    ORDER BY similarity DESC
    LIMIT match_count;
END;
$$;

-- Add comment to the search function
COMMENT ON FUNCTION public.search_articles IS 'Search articles by semantic similarity using vector embeddings'; 