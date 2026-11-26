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

async function addReliabilityScoreColumn() {
  console.log(chalk.blue('Adding reliability_score column to articles table...'));

  try {
    // Add the reliability_score column
    const { error } = await supabase.rpc('exec_sql', {
      sql: `
        ALTER TABLE articles 
        ADD COLUMN IF NOT EXISTS reliability_score DECIMAL(3,2) CHECK (reliability_score >= 0 AND reliability_score <= 1);
      `
    });

    if (error) {
      console.error(chalk.red('Error adding column:', JSON.stringify(error, null, 2)));
      
      // Try alternative approach using direct SQL
      console.log(chalk.yellow('Trying alternative approach...'));
      const { error: altError } = await supabase
        .from('articles')
        .select('id')
        .limit(1);
      
      if (altError) {
        console.error(chalk.red('Database connection issue:', JSON.stringify(altError, null, 2)));
      } else {
        console.log(chalk.yellow('Database connection works, but RPC function may not be available.'));
        console.log(chalk.yellow('You may need to add the column manually in your Supabase dashboard:'));
        console.log(chalk.blue('ALTER TABLE articles ADD COLUMN reliability_score DECIMAL(3,2) CHECK (reliability_score >= 0 AND reliability_score <= 1);'));
      }
      return;
    }

    console.log(chalk.green('✅ reliability_score column added successfully!'));

    // Verify the column was added
    const { data: articles, error: verifyError } = await supabase
      .from('articles')
      .select('*')
      .limit(1);

    if (verifyError) {
      console.error(chalk.red('Error verifying column:', JSON.stringify(verifyError, null, 2)));
      return;
    }

    if (articles && articles.length > 0) {
      const article = articles[0];
      if ('reliability_score' in article) {
        console.log(chalk.green('✅ Column verified - reliability_score now exists!'));
      } else {
        console.log(chalk.red('❌ Column not found after addition'));
      }
    }

  } catch (error) {
    console.error(chalk.red('Error in addReliabilityScoreColumn:', error));
  }
}

// Run the script
addReliabilityScoreColumn().catch(console.error); 