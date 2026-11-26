-- Create article_editors table to track which users can edit articles
CREATE TABLE article_editors (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    created_by UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    UNIQUE(user_id)
);

-- Enable RLS
ALTER TABLE article_editors ENABLE ROW LEVEL SECURITY;

-- Create policies
CREATE POLICY "Users can view their own editor status"
    ON article_editors FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Only super admins can manage editors"
    ON article_editors FOR ALL
    USING (
        auth.uid() IN (
            SELECT user_id FROM article_editors
        )
    );

-- Add function to check if a user is an editor
CREATE OR REPLACE FUNCTION is_article_editor(user_id UUID)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1 FROM article_editors WHERE article_editors.user_id = $1
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Add RLS policy to allow editors to update articles
CREATE POLICY "Editors can update articles"
    ON articles FOR UPDATE
    USING (is_article_editor(auth.uid())); 