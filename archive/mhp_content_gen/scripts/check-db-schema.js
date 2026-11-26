import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import chalk from 'chalk';

// Get __dirname equivalent in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load environment variables
dotenv.config({ path: path.resolve(__dirname, '..', '.env') });

// Initialize Supabase client
const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

async function checkDatabaseSchema() {
  console.log(chalk.blue('Checking database schema...'));

  try {
    // Try to select a single article to see the structure
    const { data: articles, error } = await supabase
      .from('articles')
      .select('*')
      .limit(1);

    if (error) {
      console.error(chalk.red('Error fetching articles:', JSON.stringify(error, null, 2)));
      return;
    }

    if (articles && articles.length > 0) {
      const article = articles[0];
      console.log(chalk.green('Database connection successful!'));
      console.log(chalk.blue('Article columns:'));
      console.log(Object.keys(article));
      
      if ('reliability_score' in article) {
        console.log(chalk.green('✅ reliability_score column exists'));
      } else {
        console.log(chalk.red('❌ reliability_score column does NOT exist'));
        console.log(chalk.yellow('You need to add the reliability_score column to your articles table'));
      }
    } else {
      console.log(chalk.yellow('No articles found in database'));
    }

  } catch (error) {
    console.error(chalk.red('Error checking schema:', error));
  }
}

// Run the script
checkDatabaseSchema().catch(console.error); 