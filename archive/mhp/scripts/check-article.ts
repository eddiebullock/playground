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

async function checkArticle() {
  try {
    // Check the specific article
    const { data: article, error: articleError } = await supabase
      .from('articles')
      .select('*')
      .eq('slug', 'understanding-depression')
      .single();

    if (articleError) {
      throw articleError;
    }

    console.log('Article data:', JSON.stringify(article, null, 2));

    // Check all published articles
    const { data: publishedArticles, error: publishedError } = await supabase
      .from('articles')
      .select('*')
      .eq('status', 'published')
      .order('created_at', { ascending: false });

    if (publishedError) {
      throw publishedError;
    }

    console.log('\nPublished articles count:', publishedArticles.length);
    console.log('Published articles:', publishedArticles.map(a => ({
      title: a.title,
      slug: a.slug,
      status: a.status,
      created_at: a.created_at
    })));

  } catch (error) {
    console.error('Error checking article:', error);
    process.exit(1);
  }
}

checkArticle(); 