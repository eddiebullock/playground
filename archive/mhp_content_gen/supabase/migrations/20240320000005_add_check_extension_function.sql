-- Create a function to check if an extension is enabled
CREATE OR REPLACE FUNCTION public.check_extension(ext_name text)
RETURNS boolean
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1 
        FROM pg_extension 
        WHERE extname = ext_name
    );
END;
$$; 