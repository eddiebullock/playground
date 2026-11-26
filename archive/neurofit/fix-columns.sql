-- Fix Missing Columns Migration
-- Run this FIRST to add missing columns to your existing profiles table
-- This fixes the "Could not find the 'fitness_goal' column" error

DO $$ 
BEGIN
  -- Add 'name' column if it doesn't exist
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                 WHERE table_schema = 'public' 
                 AND table_name='profiles' 
                 AND column_name='name') THEN
    ALTER TABLE profiles ADD COLUMN name TEXT;
    RAISE NOTICE 'Added name column to profiles table';
  ELSE
    RAISE NOTICE 'name column already exists';
  END IF;
  
  -- Add 'fitness_goal' column if it doesn't exist
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                 WHERE table_schema = 'public' 
                 AND table_name='profiles' 
                 AND column_name='fitness_goal') THEN
    ALTER TABLE profiles ADD COLUMN fitness_goal TEXT;
    RAISE NOTICE 'Added fitness_goal column to profiles table';
  ELSE
    RAISE NOTICE 'fitness_goal column already exists';
  END IF;
END $$;

