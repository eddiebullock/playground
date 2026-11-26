# How to Add Workouts to Supabase

## Why Nothing Happened

The SQL uses `WHERE NOT EXISTS` which means it only inserts workouts that **don't already exist**. If you already have some workouts in your database, those won't be re-inserted, and only new ones will be added.

## Step-by-Step Instructions

### Option 1: Add Missing Columns First (Fixes Preferences Bug)

1. **Open Supabase SQL Editor**
   - Go to your Supabase project dashboard
   - Click "SQL Editor" in the left sidebar
   - Click "New query"

2. **Run the Column Fix Script**
   - Copy the contents of `fix-columns.sql`
   - Paste into the SQL Editor
   - Click "Run" (or press Cmd+Enter / Ctrl+Enter)
   - You should see messages like "Added fitness_goal column to profiles table"

3. **Verify Columns Were Added**
   - Go to "Table Editor" in the left sidebar
   - Click on the `profiles` table
   - You should now see `name` and `fitness_goal` columns

### Option 2: Add New Workouts

**Method A: Add All Workouts (Recommended)**

1. **Open Supabase SQL Editor**
   - Click "SQL Editor" → "New query"

2. **Copy and Run the Workout Insert**
   - Copy lines 80-119 from `supabase-schema.sql` (the INSERT statement)
   - Paste into SQL Editor
   - Click "Run"
   - This will add all workouts that don't already exist

3. **Check Results**
   - Go to "Table Editor" → `workouts` table
   - You should see all 30 workouts

**Method B: Add Only New Workouts**

If you already have some workouts and only want to add the new ones:

1. Copy the contents of `add-workouts.sql`
2. Run it in SQL Editor
3. This adds only the workouts that weren't in the original 9

### Option 3: Clear and Re-add All Workouts (If You Want Fresh Start)

⚠️ **Warning: This deletes all existing workouts!**

```sql
-- Delete all existing workouts
DELETE FROM workouts;

-- Then run the INSERT statement from supabase-schema.sql (lines 80-119)
```

## Troubleshooting

**If you get "column already exists" errors:**
- That's fine! It means the columns are already there
- The script is safe to run multiple times

**If workouts don't appear:**
- Check the `workouts` table in Table Editor
- Look for any error messages in the SQL Editor output
- Make sure you clicked "Run" after pasting the SQL

**If preferences still don't save:**
- Make sure you ran `fix-columns.sql` first
- Refresh your browser
- Check the browser console for errors

