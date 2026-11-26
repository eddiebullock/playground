-- Temporary fix: Disable RLS on article_editors table to resolve infinite recursion
ALTER TABLE article_editors DISABLE ROW LEVEL SECURITY; 