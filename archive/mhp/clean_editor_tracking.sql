-- Create article_edits table to track which editor edited which article
CREATE TABLE IF NOT EXISTS article_edits (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    article_id UUID REFERENCES articles(id) ON DELETE CASCADE NOT NULL,
    editor_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    edited_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_article_edits_article_id ON article_edits(article_id);
CREATE INDEX IF NOT EXISTS idx_article_edits_editor_id ON article_edits(editor_id);
CREATE INDEX IF NOT EXISTS idx_article_edits_edited_at ON article_edits(edited_at);

-- Create article_views table to track article views
CREATE TABLE IF NOT EXISTS article_views (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    article_id UUID REFERENCES articles(id) ON DELETE CASCADE NOT NULL,
    viewer_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    viewed_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_article_views_article_id ON article_views(article_id);
CREATE INDEX IF NOT EXISTS idx_article_views_viewer_id ON article_views(viewer_id);
CREATE INDEX IF NOT EXISTS idx_article_views_viewed_at ON article_views(viewed_at);

-- Enable RLS on new tables
ALTER TABLE article_edits ENABLE ROW LEVEL SECURITY;
ALTER TABLE article_views ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist
DROP POLICY IF EXISTS "Editors can view their own edits" ON article_edits;
DROP POLICY IF EXISTS "Editors can insert their own edits" ON article_edits;
DROP POLICY IF EXISTS "Anyone can view edit counts for ranking" ON article_edits;
DROP POLICY IF EXISTS "Users can view their own views" ON article_views;
DROP POLICY IF EXISTS "Anyone can insert views" ON article_views;
DROP POLICY IF EXISTS "Anyone can view view counts for ranking" ON article_views;

-- Create RLS policies for article_edits
CREATE POLICY "Editors can view their own edits"
    ON article_edits FOR SELECT
    USING (auth.uid() = editor_id);

CREATE POLICY "Editors can insert their own edits"
    ON article_edits FOR INSERT
    WITH CHECK (auth.uid() = editor_id);

CREATE POLICY "Anyone can view edit counts for ranking"
    ON article_edits FOR SELECT
    USING (true);

-- Create RLS policies for article_views
CREATE POLICY "Users can view their own views"
    ON article_views FOR SELECT
    USING (auth.uid() = viewer_id OR viewer_id IS NULL);

CREATE POLICY "Anyone can insert views"
    ON article_views FOR INSERT
    WITH CHECK (true);

CREATE POLICY "Anyone can view view counts for ranking"
    ON article_views FOR SELECT
    USING (true);

-- Drop existing functions if they exist
DROP FUNCTION IF EXISTS get_editor_stats(UUID);
DROP FUNCTION IF EXISTS get_editor_rankings();

-- Create functions for calculating editor stats
CREATE OR REPLACE FUNCTION get_editor_stats(editor_user_id UUID)
RETURNS TABLE (
    total_edits BIGINT,
    unique_articles_edited BIGINT,
    total_views BIGINT,
    total_saves BIGINT,
    editor_score BIGINT
) AS $$
BEGIN
    RETURN QUERY
    WITH editor_articles AS (
        SELECT DISTINCT ae.article_id
        FROM article_edits ae
        WHERE ae.editor_id = editor_user_id
    ),
    article_stats AS (
        SELECT 
            ea.article_id,
            COUNT(DISTINCT av.id) as views,
            COUNT(DISTINCT sa.id) as saves
        FROM editor_articles ea
        LEFT JOIN article_views av ON ea.article_id = av.article_id
        LEFT JOIN saved_articles sa ON ea.article_id = sa.article_id
        GROUP BY ea.article_id
    )
    SELECT 
        (SELECT COUNT(*) FROM article_edits WHERE editor_id = editor_user_id) as total_edits,
        (SELECT COUNT(DISTINCT article_id) FROM article_edits WHERE editor_id = editor_user_id) as unique_articles_edited,
        COALESCE((SELECT SUM(views) FROM article_stats), 0) as total_views,
        COALESCE((SELECT SUM(saves) FROM article_stats), 0) as total_saves,
        COALESCE(
            (SELECT COUNT(*) FROM article_edits WHERE editor_id = editor_user_id) * 1 +
            COALESCE((SELECT SUM(views) FROM article_stats), 0) / 10 +
            COALESCE((SELECT SUM(saves) FROM article_stats), 0) / 2,
            0
        )::BIGINT as editor_score;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create function to get editor rankings
CREATE OR REPLACE FUNCTION get_editor_rankings()
RETURNS TABLE (
    editor_id UUID,
    editor_email TEXT,
    total_edits BIGINT,
    unique_articles_edited BIGINT,
    total_views BIGINT,
    total_saves BIGINT,
    editor_score BIGINT,
    rank_position BIGINT
) AS $$
BEGIN
    RETURN QUERY
    WITH editor_stats AS (
        SELECT 
            ae.editor_id,
            u.email as editor_email,
            COUNT(*) as total_edits,
            COUNT(DISTINCT ae.article_id) as unique_articles_edited,
            COALESCE(SUM(article_views.views), 0) as total_views,
            COALESCE(SUM(article_saves.saves), 0) as total_saves,
            COALESCE(
                COUNT(*) * 1 +
                COALESCE(SUM(article_views.views), 0) / 10 +
                COALESCE(SUM(article_saves.saves), 0) / 2,
                0
            )::BIGINT as editor_score
        FROM article_edits ae
        JOIN auth.users u ON ae.editor_id = u.id
        LEFT JOIN (
            SELECT 
                ae2.article_id,
                COUNT(DISTINCT av.id) as views
            FROM article_edits ae2
            LEFT JOIN article_views av ON ae2.article_id = av.article_id
            GROUP BY ae2.article_id
        ) article_views ON ae.article_id = article_views.article_id
        LEFT JOIN (
            SELECT 
                ae3.article_id,
                COUNT(DISTINCT sa.id) as saves
            FROM article_edits ae3
            LEFT JOIN saved_articles sa ON ae3.article_id = sa.article_id
            GROUP BY ae3.article_id
        ) article_saves ON ae.article_id = article_saves.article_id
        GROUP BY ae.editor_id, u.email
    )
    SELECT 
        es.editor_id,
        es.editor_email,
        es.total_edits,
        es.unique_articles_edited,
        es.total_views,
        es.total_saves,
        es.editor_score,
        ROW_NUMBER() OVER (ORDER BY es.editor_score DESC) as rank_position
    FROM editor_stats es
    ORDER BY es.editor_score DESC;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER; 