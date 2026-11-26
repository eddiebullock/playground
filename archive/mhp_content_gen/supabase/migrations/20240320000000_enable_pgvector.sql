-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify the extension is enabled
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 
        FROM pg_extension 
        WHERE extname = 'vector'
    ) THEN
        RAISE EXCEPTION 'pgvector extension could not be enabled. Please ensure it is installed in your database.';
    END IF;
END $$; 