import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';

// Load environment variables from .env.local
dotenv.config({ path: '.env.local' });

// Initialize Supabase client
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

console.log('Environment check:');
console.log('SUPABASE_URL:', supabaseUrl ? 'Set' : 'Missing');
console.log('SUPABASE_KEY:', supabaseKey ? 'Set' : 'Missing');

if (!supabaseUrl || !supabaseKey) {
  console.error('Missing required environment variables');
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseKey);

async function inspectArticles() {
  console.log('Inspecting articles in database...\n');

  try {
    // Get all published articles
    const { data: articles, error } = await supabase
      .from('articles')
      .select('*')
      .eq('status', 'published')
      .order('title');

    if (error) {
      console.error('Error fetching articles:', error);
      return;
    }

    console.log(`Found ${articles.length} published articles\n`);

    // Analyze categories
    const categories = {};
    const tags = new Set();
    const contentBlocks = new Set();

    articles.forEach(article => {
      // Count categories
      const category = article.category || 'uncategorized';
      categories[category] = (categories[category] || 0) + 1;

      // Collect tags
      if (article.tags) {
        article.tags.forEach(tag => tags.add(tag));
      }

      // Collect content block types
      if (article.content_blocks) {
        Object.keys(article.content_blocks).forEach(block => contentBlocks.add(block));
      }
    });

    console.log('Categories:');
    Object.entries(categories).forEach(([category, count]) => {
      console.log(`  ${category}: ${count} articles`);
    });

    console.log('\nAll tags found:');
    Array.from(tags).sort().forEach(tag => {
      console.log(`  - ${tag}`);
    });

    console.log('\nContent block types:');
    Array.from(contentBlocks).sort().forEach(block => {
      console.log(`  - ${block}`);
    });

    // Show sample articles
    console.log('\nSample articles:');
    articles.slice(0, 5).forEach(article => {
      console.log(`\n  Title: ${article.title}`);
      console.log(`  Category: ${article.category}`);
      console.log(`  Tags: ${(article.tags || []).join(', ')}`);
      console.log(`  Summary: ${article.summary?.substring(0, 100)}...`);
      console.log(`  Content blocks: ${Object.keys(article.content_blocks || {}).length}`);
    });

    // Test search terms
    const testQueries = ['depression', 'anxiety', 'therapy', 'brain', 'sleep', 'exercise'];
    
    console.log('\nTesting search terms:');
    testQueries.forEach(query => {
      const matchingArticles = articles.filter(article => {
        const title = article.title.toLowerCase();
        const summary = article.summary.toLowerCase();
        const tags = (article.tags || []).map(t => t.toLowerCase());
        const contentText = JSON.stringify(article.content_blocks || {}).toLowerCase();
        
        return title.includes(query) || 
               summary.includes(query) || 
               tags.some(tag => tag.includes(query)) ||
               contentText.includes(query);
      });
      
      console.log(`  "${query}": ${matchingArticles.length} matches`);
      if (matchingArticles.length > 0) {
        console.log(`    Examples: ${matchingArticles.slice(0, 3).map(a => a.title).join(', ')}`);
      }
    });

  } catch (error) {
    console.error('Script error:', error);
  }
}

inspectArticles(); 