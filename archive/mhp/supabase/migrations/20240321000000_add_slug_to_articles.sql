-- Add slug column to articles table
ALTER TABLE articles ADD COLUMN IF NOT EXISTS slug TEXT UNIQUE;

-- Create a function to generate a slug from a title
CREATE OR REPLACE FUNCTION generate_slug(title TEXT) RETURNS TEXT AS $$
BEGIN
  RETURN LOWER(
    REGEXP_REPLACE(
      REGEXP_REPLACE(
        REGEXP_REPLACE(title, '[^a-zA-Z0-9\s-]', '', 'g'),
        '\s+', '-', 'g'
      ),
      '-+', '-', 'g'
    )
  );
END;
$$ LANGUAGE plpgsql;

-- Update existing articles to have slugs
UPDATE articles
SET slug = generate_slug(title)
WHERE slug IS NULL;

-- Make slug column required
ALTER TABLE articles ALTER COLUMN slug SET NOT NULL;

-- Add index for slug lookups
CREATE INDEX IF NOT EXISTS idx_articles_slug ON articles (slug); 