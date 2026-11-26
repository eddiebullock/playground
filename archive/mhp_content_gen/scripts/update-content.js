import { OpenAI } from 'openai';
import dotenv from 'dotenv';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import chalk from 'chalk';
import { program } from 'commander';

// Get __dirname equivalent in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load environment variables
dotenv.config();

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

// Initialize CLI
program
  .name('update-content')
  .description('Update specific sections of articles with new content')
  .version('1.0.0')
  .requiredOption('-s, --section <string>', 'Section to update (e.g., summary, overview, practical_takeaways)')
  .option('-c, --count <number>', 'Number of most recent articles to update', '10')
  .option('-m, --model <string>', 'GPT model to use', 'gpt-4-turbo-preview')
  .parse(process.argv);

const options = program.opts();

// Define section-specific prompts
const sectionPrompts = {
  summary: `Generate a clear, direct summary for an article about "{topic}" in the {category} category.

Requirements:
- Start with a clear, direct definition of the topic
- Focus on what the condition/topic IS, not what the article will cover
- Avoid phrases like "This article explores..." or "We discuss..."
- Keep it concise (2-3 sentences maximum)
- Use active voice and present tense
- Include key prevalence or impact information if relevant

Example good summary: "Obsessive-Compulsive Disorder (OCD) is a mental health condition characterized by persistent, unwanted thoughts (obsessions) and repetitive behaviors (compulsions). It affects approximately 1-2% of the population and can significantly impact daily functioning and quality of life."

Write only the summary, nothing else.`,

  overview: `Generate a clear, engaging overview for an article about "{topic}" in the {category} category.

Requirements:
- Start with a clear definition and context
- Explain why this topic matters
- Include key statistics or prevalence data
- Keep it concise and engaging
- Use accessible language
- Include relevant citations

Write only the overview, nothing else.`,

  practical_takeaways: `Generate practical takeaways for an article about "{topic}" in the {category} category.

Requirements:
- Focus on actionable advice and strategies
- Include evidence-based recommendations
- Make it practical and implementable
- Use clear, direct language
- Include 3-5 key points
- Format as a cohesive paragraph

Write only the practical takeaways, nothing else.`,

  // Add more section prompts as needed
};

async function generateNewContent(topic, category, section) {
  const promptTemplate = sectionPrompts[section];
  if (!promptTemplate) {
    throw new Error(`No prompt template defined for section: ${section}`);
  }

  const prompt = promptTemplate
    .replace('{topic}', topic)
    .replace('{category}', category);

  const completion = await openai.chat.completions.create({
    model: options.model,
    messages: [
      {
        role: "system",
        content: "You are an expert content writer specializing in mental health, psychology, and neuroscience. Your task is to write clear, engaging content that is both scientifically accurate and accessible to a general audience."
      },
      {
        role: "user",
        content: prompt
      }
    ],
    temperature: 0.7
  });

  return completion.choices[0].message.content.trim();
}

async function updateContent() {
  try {
    // Read existing articles
    const filePath = path.join(__dirname, '..', 'articles-data.json');
    const articles = JSON.parse(await fs.readFile(filePath, 'utf-8'));
    
    console.log(chalk.blue(`\nFound ${articles.length} articles.`));
    
    // Update specified number of most recent articles
    const articlesToUpdate = articles.slice(-options.count);
    console.log(chalk.blue(`Updating ${options.section} for the last ${articlesToUpdate.length} articles...`));

    for (const article of articlesToUpdate) {
      console.log(chalk.yellow(`\nUpdating ${options.section} for: ${article.title}`));
      
      const newContent = await generateNewContent(article.title, article.category, options.section);
      
      // Handle both top-level and content_blocks fields
      if (options.section in article) {
        console.log(chalk.green('Old content:'), article[options.section]);
        console.log(chalk.green('New content:'), newContent);
        article[options.section] = newContent;
      } else if (article.content_blocks && options.section in article.content_blocks) {
        console.log(chalk.green('Old content:'), article.content_blocks[options.section]);
        console.log(chalk.green('New content:'), newContent);
        article.content_blocks[options.section] = newContent;
      } else {
        console.log(chalk.red(`Section ${options.section} not found in article structure`));
        continue;
      }
      
      // Add a small delay to avoid rate limits
      await new Promise(resolve => setTimeout(resolve, 1000));
    }

    // Save updated articles
    await fs.writeFile(filePath, JSON.stringify(articles, null, 2));
    console.log(chalk.green('\n✅ Successfully updated content!'));
    console.log(chalk.blue('Updated articles saved to articles-data.json'));

  } catch (error) {
    console.error(chalk.red('\nError updating content:'), error);
    process.exit(1);
  }
}

// Run the script
updateContent(); 