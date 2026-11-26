-- Add New Workouts to Existing Database
-- Run this AFTER you've already run the main schema
-- This will add the new workouts without affecting existing ones

INSERT INTO workouts (title, description, tags, sensory_level, duration, equipment)
SELECT * FROM (VALUES
  -- Medium Intensity Workouts (if not already added)
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

