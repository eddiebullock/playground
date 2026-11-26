import { program } from 'commander';
import chalk from 'chalk';
import path from 'path';
import fs from 'fs/promises';
import { fileURLToPath } from 'url';

// Get __dirname equivalent in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Predefined risk factor topics
const RISK_FACTOR_TOPICS = [
  'Childhood Trauma',
  'Genetic Predisposition',
  'Chronic Stress',
  'Social Isolation',
  'Substance Abuse',
  'Sleep Deprivation',
  'Poor Nutrition',
  'Physical Inactivity',
  'Environmental Toxins',
  'Socioeconomic Disadvantage',
  'Discrimination and Racism',
  'Family History of Mental Illness',
  'Early Life Adversity',
  'Urban Living',
  'Digital Overuse',
  'Workplace Stress',
  'Financial Insecurity',
  'Relationship Problems',
  'Academic Pressure',
  'Social Media Use',
  'Climate Change Anxiety',
  'Political Polarization',
  'Healthcare Access Barriers',
  'Housing Instability',
  'Food Insecurity'
];

program
  .name('generate-risk-factors')
  .description('Generate multiple risk factor articles')
  .option('-t, --topics <string>', 'Comma-separated list of specific topics (optional)')
  .option('-n, --number <number>', 'Number of articles to generate (default: 10)', '10')
  .option('-m, --model <string>', 'GPT model to use (ignored, always uses gpt-4o-mini)', 'gpt-4o-mini')
  .option('--all', 'Generate all predefined risk factor topics')
  .parse(process.argv);

const options = program.opts();
const model = 'gpt-4o-mini'; // Always use gpt-4o-mini regardless of user input

async function generateRiskFactorArticles(topics, model) {
  console.log(chalk.blue(`\nGenerating ${topics.length} risk factor articles...`));
  console.log(chalk.blue('Topics:', topics.join(', '), '\n'));

  // Clear the output file only once at the start
  const filePath = path.join(__dirname, '..', 'articles-data.json');
  await fs.writeFile(filePath, '[]');
  console.log(chalk.blue('Cleared existing articles from output file.'));

  let successCount = 0;
  let allArticles = [];

  for (const topic of topics) {
    try {
      console.log(chalk.yellow(`\nGenerating risk factor article about "${topic}"...`));
      
      // Run the generate-article script for each topic
      const { execSync } = await import('child_process');
      execSync(`node src/generate-article.js --topic "${topic}" --category "risk_factors" --model "${model}" --no-clear`, { stdio: 'inherit' });
      
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
  console.log(chalk.green(`✅ Successfully generated: ${successCount} risk factor articles`));
  console.log(chalk.green(`📝 Total articles in file: ${allArticles.length}`));
}

// Main execution
async function main() {
  let topics = [];

  if (options.all) {
    // Generate all predefined topics
    topics = RISK_FACTOR_TOPICS;
  } else if (options.topics) {
    // Use custom topics
    topics = options.topics.split(',').map(t => t.trim());
  } else {
    // Use a subset of predefined topics based on the number option
    const number = parseInt(options.number);
    topics = RISK_FACTOR_TOPICS.slice(0, Math.min(number, RISK_FACTOR_TOPICS.length));
  }

  if (topics.length === 0) {
    console.error(chalk.red('No topics specified. Use --topics or --number options.'));
    process.exit(1);
  }

  await generateRiskFactorArticles(topics, model);
}

// Run the script
main().catch(console.error); 