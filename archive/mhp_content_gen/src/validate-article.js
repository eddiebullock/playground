import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import chalk from 'chalk';
import { validateArticle } from './schema.js';

// Get __dirname equivalent in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function validateArticles(filePath = 'articles-data.json') {
  try {
    // Read and parse the articles file
    const fullPath = path.join(__dirname, '..', filePath);
    console.log(chalk.blue(`\nValidating articles in: ${filePath}`));
    
    const content = await fs.readFile(fullPath, 'utf-8');
    const articles = JSON.parse(content);

    if (!Array.isArray(articles)) {
      throw new Error('Articles data must be an array');
    }

    console.log(chalk.blue(`Found ${articles.length} articles to validate\n`));

    // Validate each article
    let validCount = 0;
    let invalidCount = 0;

    for (const [index, article] of articles.entries()) {
      console.log(chalk.blue(`\nValidating article ${index + 1}/${articles.length}:`));
      console.log(chalk.blue(`Title: ${article.title}`));
      console.log(chalk.blue(`Category: ${article.category}`));

      const validation = validateArticle(article);

      if (validation.isValid) {
        console.log(chalk.green('✓ Valid'));
        validCount++;
      } else {
        console.log(chalk.red('✗ Invalid'));
        console.log(chalk.red('Validation errors:'));
        validation.errors.forEach(error => {
          console.log(chalk.red(`  - ${error.path.join('.')}: ${error.message}`));
        });
        invalidCount++;
      }
    }

    // Print summary
    console.log(chalk.blue('\nValidation Summary:'));
    console.log(chalk.green(`✓ Valid articles: ${validCount}`));
    console.log(chalk.red(`✗ Invalid articles: ${invalidCount}`));
    console.log(chalk.blue(`Total articles: ${articles.length}`));

    if (invalidCount > 0) {
      process.exit(1);
    }

  } catch (error) {
    console.error(chalk.red('\nError:'), error.message);
    process.exit(1);
  }
}

// Run validation
validateArticles(); 