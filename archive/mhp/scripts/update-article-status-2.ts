import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';

// Load environment variables from .env.local
dotenv.config({ path: '.env.local' });

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!supabaseUrl || !supabaseServiceKey) {
  console.error('Missing required environment variables');
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseServiceKey);

const articleIds = [
  'c082f32d-4cf9-437f-a83a-aa3b026fb998', // Understanding Persistent Depressive Disorder
  '735f4317-e0e1-4618-a221-16bef8d48ad1', // Understanding Bipolar I Disorder
  '9dbf7800-acc6-4eb3-95ac-453fb60287c0', // Understanding Bipolar II Disorder
  '5e5302f9-23fa-421f-b26f-72ec6aba2839', // Understanding Cyclothymic Disorder
  'd81bb0ea-c71e-43ae-8659-58da954398f8', // Understanding Seasonal Affective Disorder
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

    console.log('Successfully updated articles:');
    data.forEach(article => {
      console.log(`- ${article.title} (${article.id})`);
    });
  } catch (error) {
    console.error('Error updating articles:', error);
    process.exit(1);
  }
}

updateArticleStatus(); 