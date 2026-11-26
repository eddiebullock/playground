-- Create the user_prompts table for tracking user interactions and personalization
CREATE TABLE public.user_prompts (
    id UUID NOT NULL DEFAULT extensions.uuid_generate_v4(),
    user_id UUID NOT NULL,
    prompt TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Track which articles were used to generate the response
    referenced_articles UUID[] DEFAULT '{}',
    -- Store the generated response and any metadata
    response JSONB,
    -- Track user feedback, engagement metrics, and any other interaction data
    feedback JSONB DEFAULT '{}',
    
    -- Primary key constraint
    CONSTRAINT user_prompts_pkey PRIMARY KEY (id),
    
    -- Foreign key to users table
    CONSTRAINT user_prompts_user_id_fkey FOREIGN KEY (user_id)
        REFERENCES auth.users (id)
        ON DELETE CASCADE,
    
    -- Ensure response is a valid JSON object when present
    CONSTRAINT user_prompts_response_is_object CHECK (
        response IS NULL OR jsonb_typeof(response) = 'object'
    ),
    
    -- Ensure feedback is a valid JSON object
    CONSTRAINT user_prompts_feedback_is_object CHECK (
        jsonb_typeof(feedback) = 'object'
    )
);

-- Create indexes for common query patterns
CREATE INDEX idx_user_prompts_user_id ON public.user_prompts USING btree (user_id);
CREATE INDEX idx_user_prompts_created_at ON public.user_prompts USING btree (created_at);
CREATE INDEX idx_user_prompts_referenced_articles ON public.user_prompts USING gin (referenced_articles);
CREATE INDEX idx_user_prompts_response ON public.user_prompts USING gin (response);
CREATE INDEX idx_user_prompts_feedback ON public.user_prompts USING gin (feedback);

-- Add RLS policies
ALTER TABLE public.user_prompts ENABLE ROW LEVEL SECURITY;

-- Policy to allow users to view their own prompts
CREATE POLICY "Users can view their own prompts"
    ON public.user_prompts
    FOR SELECT
    USING (auth.uid() = user_id);

-- Policy to allow users to insert their own prompts
CREATE POLICY "Users can insert their own prompts"
    ON public.user_prompts
    FOR INSERT
    WITH CHECK (auth.uid() = user_id);

-- Policy to allow users to update their own prompts
CREATE POLICY "Users can update their own prompts"
    ON public.user_prompts
    FOR UPDATE
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

-- Add comments
COMMENT ON TABLE public.user_prompts IS 'Stores user prompts and their interactions with personalized content';
COMMENT ON COLUMN public.user_prompts.id IS 'Unique identifier for the prompt';
COMMENT ON COLUMN public.user_prompts.user_id IS 'Reference to the user who created the prompt';
COMMENT ON COLUMN public.user_prompts.prompt IS 'The user''s original prompt text';
COMMENT ON COLUMN public.user_prompts.referenced_articles IS 'Array of article IDs used to generate the response';
COMMENT ON COLUMN public.user_prompts.response IS 'The generated personalized response and metadata';
COMMENT ON COLUMN public.user_prompts.feedback IS 'User feedback and engagement metrics'; 