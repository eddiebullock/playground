import { OpenAI } from 'openai';
import dotenv from 'dotenv';
import { createClient } from '@supabase/supabase-js';
import chalk from 'chalk';
import { program } from 'commander';

// Load environment variables
dotenv.config();

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

// Initialize Supabase client
const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

// Initialize CLI
program
  .name('update-db-summaries')
  .description('Update summaries of articles in the database')
  .version('1.0.0')
  .option('-c, --count <number>', 'Number of most recent articles to update', '10')
  .option('-m, --model <string>', 'GPT model to use', 'gpt-4-turbo-preview')
  .parse(process.argv);

const options = program.opts();

async function generateNewSummary(title, category) {
  const prompt = `Generate a clear, direct summary for an article about "${title}" in the ${category} category.

Requirements:
- Start with a clear, direct definition of the topic
- Focus on what the condition/topic IS, not what the article will cover
- Avoid phrases like "This article explores..." or "We discuss..."
- Keep it concise (2-3 sentences maximum)
- Use active voice and present tense
- Include key prevalence or impact information if relevant

Example good summary: "Obsessive-Compulsive Disorder (OCD) is a mental health condition characterized by persistent, unwanted thoughts (obsessions) and repetitive behaviors (compulsions). It affects approximately 1-2% of the population and can significantly impact daily functioning and quality of life."

Write only the summary, nothing else.`;

  const completion = await openai.chat.completions.create({
    model: options.model,
    messages: [
      {
        role: "system",
        content: "You are a mental health content expert. Generate clear, direct summaries that focus on what the condition/topic IS, not what the article will cover. Avoid phrases like 'This article explores...' or 'We discuss...'."
      },
      {
        role: "user",
        content: prompt
      }
    ]
  });

  return completion.choices[0].message.content.trim();
}

async function updateSummaries() {
  try {
    // Fetch the most recent articles from the database
    const { data: articles, error: fetchError } = await supabase
      .from('articles')
      .select('id, title, category, summary')
      .order('created_at', { ascending: false })
      .limit(parseInt(options.count));

    if (fetchError) {
      throw fetchError;
    }

    console.log(chalk.blue(`\nFound ${articles.length} articles to update.`));

    let updated = 0;
    let failed = 0;

    for (const article of articles) {
      console.log(chalk.yellow(`\nUpdating summary for: ${article.title}`));
      console.log(chalk.green('Old summary:'), article.summary);

      try {
        const newSummary = await generateNewSummary(article.title, article.category);
        console.log(chalk.green('New summary:'), newSummary);

        const { error: updateError } = await supabase
          .from('articles')
          .update({ summary: newSummary })
          .eq('id', article.id);

        if (updateError) {
          throw updateError;
        }

        updated++;
        console.log(chalk.green('✓ Successfully updated'));

        // Add a small delay to avoid rate limits
        await new Promise(resolve => setTimeout(resolve, 1000));
      } catch (error) {
        console.error(chalk.red('Error updating article:'), error);
        failed++;
      }
    }

    console.log(chalk.green('\n✅ Update complete:'));
    console.log(chalk.green(`✓ Successfully updated: ${updated} articles`));
    if (failed > 0) {
      console.log(chalk.red(`✗ Failed to update: ${failed} articles`));
    }

  } catch (error) {
    console.error(chalk.red('\nError:'), error);
    process.exit(1);
  }
}

// Run the script
updateSummaries(); 