# NeuroFit - Setup Guide

## Prerequisites

- Node.js 18+ and npm
- Supabase account (free tier works)
- OpenAI API key (for AI Coach feature)

## Installation

1. Install dependencies:
```bash
npm install
```

2. Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

3. Fill in your environment variables in `.env`:
   - `VITE_SUPABASE_URL`: Your Supabase project URL
   - `VITE_SUPABASE_ANON_KEY`: Your Supabase anonymous key
   - `VITE_OPENAI_API_KEY`: Your OpenAI API key

## Supabase Setup

1. Create a new project at [supabase.com](https://supabase.com)

2. Run the SQL schema (see `supabase-schema.sql`) in your Supabase SQL Editor:

```sql
-- User preferences table
CREATE TABLE IF NOT EXISTS user_preferences (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE UNIQUE NOT NULL,
  sensory_level TEXT,
  energy_level TEXT,
  environment TEXT,
  workout_time TEXT,
  equipment TEXT,
  special_considerations TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Workouts table
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

-- Workout progress table
CREATE TABLE IF NOT EXISTS workout_progress (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  workout_id UUID REFERENCES workouts(id) ON DELETE CASCADE NOT NULL,
  completed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE user_preferences ENABLE ROW LEVEL SECURITY;
ALTER TABLE workouts ENABLE ROW LEVEL SECURITY;
ALTER TABLE workout_progress ENABLE ROW LEVEL SECURITY;

-- RLS Policies for user_preferences
CREATE POLICY "Users can view their own preferences"
  ON user_preferences FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own preferences"
  ON user_preferences FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own preferences"
  ON user_preferences FOR UPDATE
  USING (auth.uid() = user_id);

-- RLS Policies for workouts (public read)
CREATE POLICY "Anyone can view workouts"
  ON workouts FOR SELECT
  USING (true);

-- RLS Policies for workout_progress
CREATE POLICY "Users can view their own progress"
  ON workout_progress FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own progress"
  ON workout_progress FOR INSERT
  WITH CHECK (auth.uid() = user_id);
```

3. Insert some sample workouts (optional):

```sql
INSERT INTO workouts (title, description, tags, sensory_level, duration, equipment) VALUES
('Gentle Morning Stretch', 'A calm 10-minute stretching routine to start your day', ARRAY['stretching', 'low-intensity'], 'low', 10, 'none'),
('Bodyweight Strength', 'Basic strength exercises using only your body weight', ARRAY['strength', 'bodyweight'], 'medium', 20, 'none'),
('Calm Yoga Flow', 'A peaceful yoga sequence designed for low sensory input', ARRAY['yoga', 'flexibility'], 'low', 15, 'none'),
('Walking Meditation', 'Mindful walking exercise combining movement and calm', ARRAY['cardio', 'mindfulness'], 'low', 30, 'none');
```

## Running the App

1. Start the development server:
```bash
npm run dev
```

2. Open your browser to `http://localhost:5173`

3. Sign up for an account, complete onboarding, and start using the app!

## Project Structure

```
src/
├── pages/
│   ├── index.jsx          # Landing page
│   ├── auth.jsx           # Sign in/Sign up page
│   ├── dashboard.jsx      # Main dashboard
│   └── onboarding.jsx     # User preferences questionnaire
├── components/
│   ├── Header.jsx          # Navigation header
│   ├── Footer.jsx          # Footer component
│   ├── AIChat.jsx          # AI Coach chat interface
│   ├── WorkoutCard.jsx     # Individual workout card
│   ├── WorkoutList.jsx     # Workout library list
│   └── ProgressTracker.jsx # Progress tracking component
├── services/
│   ├── supabaseClient.js   # Supabase client configuration
│   └── api.js              # Database API helpers
├── utils/
│   ├── prompts.js          # GPT prompt templates
│   └── helpers.js          # Utility functions
└── styles/
    ├── globals.css         # Global Tailwind styles (index.css)
    └── dashboard.css       # Dashboard-specific styles
```

## Features

- **Authentication**: Sign up, sign in, and sign out using Supabase Auth
- **Onboarding**: Questionnaire to capture user preferences
- **AI Coach**: GPT-powered chat interface for fitness guidance
- **Workout Library**: Browse and filter workouts based on preferences
- **Progress Tracking**: Track completed workouts and view statistics

## Notes

- The OpenAI API key is used client-side for MVP purposes. In production, you should proxy API calls through your backend.
- Make sure to configure Supabase RLS policies correctly for security.
- Replace placeholder API keys with your actual keys before deploying.

