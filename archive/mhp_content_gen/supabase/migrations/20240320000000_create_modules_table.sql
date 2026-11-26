-- Create an enum type for module types to ensure type safety
CREATE TYPE public.module_type AS ENUM ('intervention', 'explanation');

-- Create the modules table
CREATE TABLE public.modules (
    id UUID NOT NULL DEFAULT extensions.uuid_generate_v4(),
    user_id UUID NOT NULL,
    type public.module_type NOT NULL,
    title TEXT NOT NULL,
    content JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    
    -- Primary key constraint
    CONSTRAINT modules_pkey PRIMARY KEY (id),
    
    -- Foreign key to users table
    CONSTRAINT modules_user_id_fkey FOREIGN KEY (user_id)
        REFERENCES auth.users (id)
        ON DELETE CASCADE,
    
    -- Ensure content is not null and is a valid JSON object
    CONSTRAINT modules_content_is_object CHECK (jsonb_typeof(content) = 'object')
);

-- Create indexes for common query patterns
CREATE INDEX idx_modules_user_id ON public.modules USING btree (user_id);
CREATE INDEX idx_modules_type ON public.modules USING btree (type);
CREATE INDEX idx_modules_created_at ON public.modules USING btree (created_at);
CREATE INDEX idx_modules_content ON public.modules USING gin (content);

-- Add RLS policies
ALTER TABLE public.modules ENABLE ROW LEVEL SECURITY;

-- Policy to allow users to view their own modules
CREATE POLICY "Users can view their own modules"
    ON public.modules
    FOR SELECT
    USING (auth.uid() = user_id);

-- Policy to allow users to insert their own modules
CREATE POLICY "Users can insert their own modules"
    ON public.modules
    FOR INSERT
    WITH CHECK (auth.uid() = user_id);

-- Policy to allow users to update their own modules
CREATE POLICY "Users can update their own modules"
    ON public.modules
    FOR UPDATE
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

-- Policy to allow users to delete their own modules
CREATE POLICY "Users can delete their own modules"
    ON public.modules
    FOR DELETE
    USING (auth.uid() = user_id);

-- Create a trigger to automatically update the updated_at timestamp
CREATE OR REPLACE FUNCTION public.handle_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER set_updated_at
    BEFORE UPDATE ON public.modules
    FOR EACH ROW
    EXECUTE FUNCTION public.handle_updated_at();

-- Add comment to table
COMMENT ON TABLE public.modules IS 'Stores modular content pieces that can be of different types (intervention, explanation, etc.)';

-- Add comments to columns
COMMENT ON COLUMN public.modules.id IS 'Unique identifier for the module';
COMMENT ON COLUMN public.modules.user_id IS 'Reference to the user who created the module';
COMMENT ON COLUMN public.modules.type IS 'Type of module (intervention, explanation, etc.)';
COMMENT ON COLUMN public.modules.title IS 'Title/label for the module';
COMMENT ON COLUMN public.modules.content IS 'JSONB object containing the module data';
COMMENT ON COLUMN public.modules.created_at IS 'Timestamp when the module was created';
COMMENT ON COLUMN public.modules.updated_at IS 'Timestamp when the module was last updated'; 