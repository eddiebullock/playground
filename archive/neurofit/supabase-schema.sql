-- NeuroFit MVP Database Schema
-- Run this in your Supabase SQL Editor

-- Profiles table (stores user preferences)
CREATE TABLE IF NOT EXISTS profiles (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE UNIQUE NOT NULL,
  name TEXT,
  sensory_level TEXT,
  energy_level TEXT,
  environment TEXT,
  fitness_goal TEXT,
  workout_time TEXT,
  equipment TEXT,
  special_considerations TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Workouts table (curated workout library)
CREATE TABLE IF NOT EXISTS workouts (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  title TEXT NOT NULL,
  description TEXT,
  tags TEXT[] DEFAULT '{}',
  sensory_level TEXT,
  duration INTEGER,
  equipment TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Progress table (tracks completed workouts)
CREATE TABLE IF NOT EXISTS progress (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  workout_id UUID REFERENCES workouts(id) ON DELETE CASCADE NOT NULL,
  completed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE workouts ENABLE ROW LEVEL SECURITY;
ALTER TABLE progress ENABLE ROW LEVEL SECURITY;

-- Add missing columns to profiles table if they don't exist
DO $$ 
BEGIN
  -- Add 'name' column if it doesn't exist
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                 WHERE table_name='profiles' AND column_name='name') THEN
    ALTER TABLE profiles ADD COLUMN name TEXT;
  END IF;
  
  -- Add 'fitness_goal' column if it doesn't exist
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                 WHERE table_name='profiles' AND column_name='fitness_goal') THEN
    ALTER TABLE profiles ADD COLUMN fitness_goal TEXT;
  END IF;
END $$;

-- Drop existing policies if they exist (for re-running the script)
DROP POLICY IF EXISTS "Users can view their own profile" ON profiles;
DROP POLICY IF EXISTS "Users can insert their own profile" ON profiles;
DROP POLICY IF EXISTS "Users can update their own profile" ON profiles;
DROP POLICY IF EXISTS "Anyone can view workouts" ON workouts;
DROP POLICY IF EXISTS "Users can view their own progress" ON progress;
DROP POLICY IF EXISTS "Users can insert their own progress" ON progress;

-- RLS Policies for profiles
CREATE POLICY "Users can view their own profile"
  ON profiles FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own profile"
  ON profiles FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own profile"
  ON profiles FOR UPDATE
  USING (auth.uid() = user_id);

-- RLS Policies for workouts (public read)
CREATE POLICY "Anyone can view workouts"
  ON workouts FOR SELECT
  USING (true);

-- RLS Policies for progress
CREATE POLICY "Users can view their own progress"
  ON progress FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own progress"
  ON progress FOR INSERT
  WITH CHECK (auth.uid() = user_id);

-- Seed data: Example workouts (only insert if they don't exist)
INSERT INTO workouts (title, description, tags, sensory_level, duration, equipment)
SELECT * FROM (VALUES
  -- Low Intensity Workouts
  ('Gentle Morning Stretch', 'A calm 10-minute stretching routine to start your day peacefully. Perfect for easing into movement and reducing morning stiffness.', ARRAY['stretching', 'low-intensity', 'morning']::TEXT[], 'low', 10, 'none'),
  ('Calm Yoga Flow', 'A peaceful yoga sequence designed for low sensory input and relaxation. Focuses on gentle movements and breathing.', ARRAY['yoga', 'flexibility', 'mindfulness']::TEXT[], 'low', 15, 'none'),
  ('Walking Meditation', 'Mindful walking exercise combining movement and calm. Perfect for outdoor environments and connecting with nature.', ARRAY['cardio', 'mindfulness', 'outdoor']::TEXT[], 'low', 30, 'none'),
  ('Desk Break Movement', 'Quick 5-minute movements perfect for work breaks. Helps relieve tension and improve focus without leaving your workspace.', ARRAY['stretching', 'quick', 'home']::TEXT[], 'low', 5, 'none'),
  ('Evening Wind-Down', 'Gentle movements to help transition to rest. Designed to calm the nervous system and prepare your body for sleep.', ARRAY['yoga', 'relaxation', 'home']::TEXT[], 'low', 15, 'none'),
  ('Flexibility Focus', 'Deep stretching routine for improved mobility. Targets major muscle groups to increase range of motion and reduce tension.', ARRAY['flexibility', 'stretching']::TEXT[], 'low', 20, 'none'),
  ('Balance & Stability', 'Exercises focused on improving balance and core stability. Great for building confidence in movement and preventing falls.', ARRAY['balance', 'core', 'low-intensity']::TEXT[], 'low', 15, 'none'),
  ('Breathing & Movement', 'Combines controlled breathing with gentle movements. Excellent for stress relief and body awareness.', ARRAY['breathing', 'mindfulness', 'low-intensity']::TEXT[], 'low', 12, 'none'),
  ('Joint Mobility', 'Gentle movements to improve joint health and reduce stiffness. Ideal for those with limited mobility or joint concerns.', ARRAY['mobility', 'low-intensity', 'rehabilitation']::TEXT[], 'low', 18, 'none'),
  ('Restorative Stretch', 'Very gentle stretching routine for recovery days. Promotes relaxation and helps muscles recover.', ARRAY['stretching', 'recovery', 'low-intensity']::TEXT[], 'low', 25, 'none'),
  ('Standing Only Workout', 'Complete workout performed entirely standing. Perfect for small spaces or when floor work is not preferred.', ARRAY['standing', 'full-body', 'home']::TEXT[], 'low', 22, 'none'),
  ('Mindful Movement', 'Slow, intentional movements paired with awareness. Reduces anxiety and improves body connection.', ARRAY['mindfulness', 'low-intensity', 'stress-relief']::TEXT[], 'low', 20, 'none'),
  
  -- Medium Intensity Workouts
  ('Bodyweight Strength', 'Basic strength exercises using only your body weight - no equipment needed. Builds functional strength through push-ups, squats, and planks.', ARRAY['strength', 'bodyweight', 'moderate']::TEXT[], 'medium', 20, 'none'),
  ('Strength Builder', 'Progressive strength training for building muscle. Uses basic equipment like resistance bands or dumbbells for effective workouts.', ARRAY['strength', 'moderate']::TEXT[], 'medium', 30, 'basic'),
  ('Cardio Boost', 'Moderate intensity cardiovascular workout. Improves heart health and endurance through rhythmic movements.', ARRAY['cardio', 'endurance']::TEXT[], 'medium', 25, 'none'),
  ('Full Body Flow', 'A comprehensive workout targeting all major muscle groups. Perfect for overall fitness maintenance.', ARRAY['full-body', 'moderate']::TEXT[], 'medium', 35, 'none'),
  ('Quick Energy Boost', 'A short, energizing routine to combat fatigue. Designed to increase alertness without overstimulation.', ARRAY['quick', 'energy', 'moderate']::TEXT[], 'medium', 8, 'none'),
  ('Core Strengthening', 'Focused core exercises to build abdominal and back strength. Uses bodyweight movements for a strong foundation.', ARRAY['core', 'strength', 'bodyweight']::TEXT[], 'medium', 20, 'none'),
  ('Quick Cardio Burst', 'Short bursts of cardiovascular activity with rest periods. Builds endurance without prolonged intensity.', ARRAY['cardio', 'interval', 'moderate']::TEXT[], 'medium', 15, 'none'),
  ('Progressive Strength', 'Structured strength training that builds over time. Suitable for gym environments with access to equipment.', ARRAY['strength', 'progressive', 'gym']::TEXT[], 'medium', 40, 'moderate'),
  ('HIIT Beginner', 'High-intensity interval training designed for beginners. Short bursts of activity followed by recovery periods.', ARRAY['hiit', 'cardio', 'moderate']::TEXT[], 'medium', 20, 'none'),
  ('Upper Body Focus', 'Targeted upper body strength training. Builds arm, shoulder, and back strength with bodyweight and basic equipment.', ARRAY['strength', 'upper-body', 'moderate']::TEXT[], 'medium', 25, 'basic'),
  ('Lower Body Power', 'Lower body strength and power exercises. Develops leg strength, stability, and functional movement patterns.', ARRAY['strength', 'lower-body', 'moderate']::TEXT[], 'medium', 28, 'none'),
  ('Cardio Dance Flow', 'Fun, rhythmic movements set to music. Improves cardiovascular fitness while being enjoyable and engaging.', ARRAY['cardio', 'dance', 'moderate']::TEXT[], 'medium', 22, 'none'),
  
  -- High Intensity Workouts
  ('Intense Strength Circuit', 'Challenging full-body strength circuit with minimal rest. Builds serious muscle and cardiovascular endurance.', ARRAY['strength', 'high-intensity', 'circuit']::TEXT[], 'high', 35, 'moderate'),
  ('Advanced HIIT', 'High-intensity interval training for experienced exercisers. Pushes cardiovascular limits with challenging intervals.', ARRAY['hiit', 'cardio', 'high-intensity']::TEXT[], 'high', 25, 'none'),
  ('Power Plyometrics', 'Explosive jumping and power movements. Develops athleticism, speed, and explosive strength.', ARRAY['plyometrics', 'power', 'high-intensity']::TEXT[], 'high', 20, 'none'),
  ('Full Body Blast', 'Maximum intensity full-body workout. Combines strength and cardio for a complete fitness challenge.', ARRAY['full-body', 'high-intensity', 'strength']::TEXT[], 'high', 30, 'moderate'),
  ('Endurance Challenge', 'Extended duration high-intensity workout. Tests and builds both mental and physical endurance.', ARRAY['endurance', 'cardio', 'high-intensity']::TEXT[], 'high', 45, 'moderate'),
  ('Strength & Cardio Fusion', 'Combines heavy strength work with intense cardio intervals. The ultimate fitness challenge.', ARRAY['strength', 'cardio', 'high-intensity']::TEXT[], 'high', 40, 'moderate')
) AS v(title, description, tags, sensory_level, duration, equipment)
WHERE NOT EXISTS (SELECT 1 FROM workouts WHERE workouts.title = v.title);
