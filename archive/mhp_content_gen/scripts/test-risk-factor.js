import { execSync } from 'child_process';
import chalk from 'chalk';

console.log(chalk.blue('Testing risk factor article generation...\n'));

try {
  // Generate a single risk factor article
  execSync('node src/generate-article.js --topic "Chronic Stress" --category "risk_factors" --model "gpt-4o-mini"', { stdio: 'inherit' });
  
  console.log(chalk.green('\n✅ Risk factor article generation test completed successfully!'));
  console.log(chalk.blue('Check articles-data.json for the generated article.'));
} catch (error) {
  console.error(chalk.red('\n❌ Error during risk factor article generation:'), error.message);
  process.exit(1);
} 