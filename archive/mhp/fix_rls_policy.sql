-- Fix the recursive RLS policy on article_editors table
-- Drop the problematic recursive policy
DROP POLICY IF EXISTS "Only super admins can manage editors" ON article_editors;

-- Create a simpler policy that allows authenticated users to view editor status
-- This removes the recursion issue
CREATE POLICY "Authenticated users can view editor status"
    ON article_editors FOR SELECT
    USING (auth.uid() IS NOT NULL);

-- For now, we'll allow service role to manage editors
-- In production, you might want to create a more specific policy
CREATE POLICY "Service role can manage editors"
    ON article_editors FOR ALL
    USING (auth.role() = 'service_role'); 