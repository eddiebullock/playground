-- Create enum for experience types
CREATE TYPE experience_type AS ENUM (
    'anxiety',
    'stress',
    'happiness',
    'exercise',
    'social_interaction',
    'learning',
    'creativity',
    'meditation'
);

-- Create journal_entries table
CREATE TABLE journal_entries (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- Create entry_analyses table
CREATE TABLE entry_analyses (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    entry_id UUID REFERENCES journal_entries(id) ON DELETE CASCADE,
    gpt_response JSONB NOT NULL,
    brain_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- Create experiences table to store individual experiences from analysis
CREATE TABLE experiences (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    analysis_id UUID REFERENCES entry_analyses(id) ON DELETE CASCADE,
    experience_type experience_type NOT NULL,
    intensity FLOAT NOT NULL CHECK (intensity >= 0 AND intensity <= 1),
    evidence_quote TEXT NOT NULL,
    brain_regions TEXT[] NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- Create RLS policies
ALTER TABLE journal_entries ENABLE ROW LEVEL SECURITY;
ALTER TABLE entry_analyses ENABLE ROW LEVEL SECURITY;
ALTER TABLE experiences ENABLE ROW LEVEL SECURITY;

-- Policies for journal_entries
CREATE POLICY "Users can view their own journal entries"
    ON journal_entries FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own journal entries"
    ON journal_entries FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own journal entries"
    ON journal_entries FOR UPDATE
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete their own journal entries"
    ON journal_entries FOR DELETE
    USING (auth.uid() = user_id);

-- Policies for entry_analyses
CREATE POLICY "Users can view their own entry analyses"
    ON entry_analyses FOR SELECT
    USING (EXISTS (
        SELECT 1 FROM journal_entries
        WHERE journal_entries.id = entry_analyses.entry_id
        AND journal_entries.user_id = auth.uid()
    ));

-- Policies for experiences
CREATE POLICY "Users can view their own experiences"
    ON experiences FOR SELECT
    USING (EXISTS (
        SELECT 1 FROM entry_analyses
        JOIN journal_entries ON journal_entries.id = entry_analyses.entry_id
        WHERE entry_analyses.id = experiences.analysis_id
        AND journal_entries.user_id = auth.uid()
    ));

-- Create indexes for better query performance
CREATE INDEX idx_journal_entries_user_id ON journal_entries(user_id);
CREATE INDEX idx_entry_analyses_entry_id ON entry_analyses(entry_id);
CREATE INDEX idx_experiences_analysis_id ON experiences(analysis_id);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = timezone('utc'::text, now());
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for journal_entries
CREATE TRIGGER update_journal_entries_updated_at
    BEFORE UPDATE ON journal_entries
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column(); 