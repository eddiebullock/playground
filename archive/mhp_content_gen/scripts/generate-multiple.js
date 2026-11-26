import { program } from 'commander';
import chalk from 'chalk';
import path from 'path';
import fs from 'fs/promises';
import { fileURLToPath } from 'url';

// Get __dirname equivalent in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

program
  .name('generate-multiple')
  .description('Generate multiple articles from a list of topics')
  .requiredOption('-t, --topics <string>', 'Comma-separated list of topics')
  .requiredOption('-c, --category <string>', 'Category for all articles')
  .option('-m, --model <string>', 'GPT model to use (ignored, always uses gpt-4o-mini)', 'gpt-4o-mini')
  .parse(process.argv);

const options = program.opts();
const topics = options.topics.split(',').map(t => t.trim());
const model = 'gpt-4o-mini';

async function generateMultipleArticles(topics, category, model) {
  console.log(chalk.blue(`\nGenerating ${topics.length} articles in the ${category} category...`));
  console.log(chalk.blue('Topics:', topics.join(', '), '\n'));

  // Clear the output file only once at the start
  const filePath = path.join(__dirname, '..', 'articles-data.json');
  await fs.writeFile(filePath, '[]');
  console.log(chalk.blue('Cleared existing articles from output file.'));

  let successCount = 0;
  let allArticles = [];

  for (const topic of topics) {
    try {
      console.log(chalk.yellow(`\nGenerating article about "${topic}"...`));
      
      // Run the generate-article script for each topic
      const { execSync } = await import('child_process');
      execSync(`node src/generate-article.js --topic "${topic}" --category "${category}" --model "${model}" --no-clear`, { stdio: 'inherit' });
      
      // Read the current articles after each generation
      const currentContent = await fs.readFile(filePath, 'utf-8');
      allArticles = JSON.parse(currentContent);
      
      successCount++;
      
      // Add a delay between articles to avoid rate limits
      if (topic !== topics[topics.length - 1]) {
        console.log(chalk.blue('\nWaiting 5 seconds before next article...'));
        await new Promise(resolve => setTimeout(resolve, 5000));
      }
    } catch (error) {
      console.error(chalk.red(`\nError generating article about "${topic}":`), error.message);
    }
  }

  // Write all articles back to the file to ensure they're all there
  await fs.writeFile(filePath, JSON.stringify(allArticles, null, 2));

  console.log(chalk.green('\nGeneration complete:'));
  console.log(chalk.green(`✅ Successfully generated: ${successCount} articles`));
  console.log(chalk.green(`📝 Total articles in file: ${allArticles.length}`));
}

// Run the script
generateMultipleArticles(topics, options.category, model).catch(console.error); 