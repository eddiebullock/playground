import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config({ path: '.env.local' });

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!supabaseUrl || !supabaseKey) {
  console.error('Missing Supabase credentials. Please ensure NEXT_PUBLIC_SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are set in your .env.local file');
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseKey);

const articleIds = [
  '25a4b8ef-02cb-4d21-baac-8c84dc9e8432',  // Understanding Generalized Anxiety Disorder
  '61d62894-b3b2-4f55-9e6e-9453661e41a1',  // Understanding Panic Disorder
  '0132c4c0-17e4-463b-866e-f3d84c3a47e4',  // Understanding Social Anxiety Disorder
  '121080cb-b4bc-4c31-988b-a8d5430afc11',  // Understanding Specific Phobias
  'ad8f574f-4728-41e8-a39a-65c2f316905d',  // Understanding Agoraphobia
  '37684a51-e789-4a5c-890c-22be5495b425',  // Understanding Separation Anxiety Disorder
  'f112ee00-e75a-447b-be58-420d27051caa'   // Understanding Selective Mutism
];

async function updateArticleStatus() {
  try {
    const { data, error } = await supabase
      .from('articles')
      .update({ status: 'published' })
      .in('id', articleIds)
      .select();

    if (error) {
      throw error;
    }

    console.log('Successfully updated articles:', data);
  } catch (error) {
    console.error('Error updating articles:', error);
    process.exit(1);
  }
}

updateArticleStatus(); 