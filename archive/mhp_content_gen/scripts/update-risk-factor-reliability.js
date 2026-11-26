import { createClient } from '@supabase/supabase-js';
import OpenAI from 'openai';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import chalk from 'chalk';

// Get __dirname equivalent in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load environment variables
dotenv.config({ path: path.resolve(__dirname, '..', '.env') });

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

// Initialize Supabase client
const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

async function generateReliabilityScore(article) {
  const prompt = `
You are an expert research analyst. Based on the following risk factor article, calculate a reliability score (0-1) that reflects the strength and consistency of the evidence.

Article Title: ${article.title}
Evidence Summary: ${article.content_blocks?.evidence_summary || article.evidence_summary || 'No evidence summary available'}

Calculate the reliability score based on these factors:
1. Effect sizes mentioned (0.1-0.3 = small, 0.3-0.5 = medium, >0.5 = large)
2. Number of studies/replications mentioned
3. Quality of evidence (RCTs, meta-analyses, longitudinal studies, etc.)
4. Consistency of findings across studies
5. Sample sizes and methodology quality

Scoring guidelines:
- 0.9-1.0: Multiple large-scale studies, meta-analyses, strong effect sizes, consistent findings
- 0.7-0.9: Good quality studies, moderate to large effect sizes, mostly consistent findings
- 0.5-0.7: Moderate quality evidence, some replication, mixed findings
- 0.3-0.5: Limited studies, small effect sizes, inconsistent findings
- 0.1-0.3: Very limited evidence, weak associations, poor methodology

Respond with ONLY a number between 0 and 1 (e.g., 0.75), no other text.
`;

  try {
    const completion = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.1,
      max_tokens: 10
    });

    const score = parseFloat(completion.choices[0].message.content.trim());
    
    // Validate the score is between 0 and 1
    if (isNaN(score) || score < 0 || score > 1) {
      console.log(chalk.yellow(`Invalid score generated for "${article.title}": ${completion.choices[0].message.content}`));
      return 0.5; // Default to medium reliability
    }
    
    return score;
  } catch (error) {
    console.error(chalk.red(`Error generating reliability score for "${article.title}":`, error.message));
    return 0.5; // Default to medium reliability
  }
}

async function updateRiskFactorsReliability() {
  console.log(chalk.blue('Fetching risk factors articles from database...'));

  try {
    // Fetch all risk factors articles
    const { data: articles, error } = await supabase
      .from('articles')
      .select('*')
      .eq('category', 'risk_factors');

    if (error) {
      console.error(chalk.red('Error fetching articles:', error));
      return;
    }

    console.log(chalk.green(`Found ${articles.length} risk factors articles`));

    let updated = 0;
    let failed = 0;

    for (const article of articles) {
      try {
        console.log(chalk.yellow(`\nProcessing: ${article.title}`));
        
        // Generate reliability score
        const reliabilityScore = await generateReliabilityScore(article);
        
        console.log(chalk.blue(`Generated reliability score: ${reliabilityScore}`));

        // Update the article in the database
        const { error: updateError } = await supabase
          .from('articles')
          .update({ 
            content_blocks: {
              ...article.content_blocks,
              reliability_score: reliabilityScore
            }
          })
          .eq('id', article.id);

        if (updateError) {
          console.error(chalk.red(`Error updating article "${article.title}":`, JSON.stringify(updateError, null, 2)));
          failed++;
        } else {
          console.log(chalk.green(`✅ Updated "${article.title}" with reliability score: ${reliabilityScore}`));
          updated++;
        }

        // Add a small delay to avoid rate limits
        await new Promise(resolve => setTimeout(resolve, 1000));

      } catch (error) {
        console.error(chalk.red(`Error processing article "${article.title}":`, error.message));
        failed++;
      }
    }

    console.log(chalk.green('\n=== Update Complete ==='));
    console.log(chalk.green(`✅ Successfully updated: ${updated} articles`));
    console.log(chalk.red(`❌ Failed to update: ${failed} articles`));

  } catch (error) {
    console.error(chalk.red('Error in updateRiskFactorsReliability:', error));
  }
}

// Run the script
updateRiskFactorsReliability().catch(console.error); 