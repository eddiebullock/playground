import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';

dotenv.config({ path: '.env.local' });

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

async function testReliabilityScores() {
  console.log('Testing Reliability Scores\n');

  try {
    // Get all published articles
    const { data: articles, error } = await supabase
      .from('articles')
      .select('*')
      .eq('status', 'published')
      .limit(50);

    if (error) {
      console.error('Error fetching articles:', error);
      return;
    }

    console.log(`Found ${articles.length} articles\n`);

    // Calculate reliability scores
    const scoredArticles = articles.map(article => {
      const score = determineReliability(article);
      return {
        title: article.title,
        category: article.category,
        score: score,
        contentBlocks: Object.keys(article.content_blocks || {}).length,
        summaryLength: (article.summary || '').length,
        tags: (article.tags || []).length
      };
    });

    // Sort by score (highest to lowest)
    scoredArticles.sort((a, b) => b.score - a.score);

    // Show score distribution
    const scoreDistribution = {};
    scoredArticles.forEach(article => {
      const score = article.score;
      scoreDistribution[score] = (scoreDistribution[score] || 0) + 1;
    });

    console.log('Score Distribution:');
    Object.entries(scoreDistribution).forEach(([score, count]) => {
      console.log(`  ${score}/5: ${count} articles`);
    });

    console.log('\nTop 10 Articles by Reliability:');
    scoredArticles.slice(0, 10).forEach((article, index) => {
      console.log(`${index + 1}. "${article.title}" - ${article.score}/5 (${article.category})`);
    });

    console.log('\nBottom 10 Articles by Reliability:');
    scoredArticles.slice(-10).reverse().forEach((article, index) => {
      console.log(`${index + 1}. "${article.title}" - ${article.score}/5 (${article.category})`);
    });

  } catch (error) {
    console.error('Script error:', error);
  }
}

function determineReliability(article) {
  // Base reliability on content quality indicators
  let score = 2; // Start with a lower base score
  
  const contentBlocks = article.content_blocks || {};
  const summary = article.summary || '';
  const tags = article.tags || [];
  
  // Bonus for comprehensive content
  if (Object.keys(contentBlocks).length > 3) score += 0.5;
  if (Object.keys(contentBlocks).length > 5) score += 0.5;
  if (summary.length > 150) score += 0.5;
  if (summary.length > 300) score += 0.5;
  
  // Bonus for evidence-based content
  if (contentBlocks.evidence_summary) score += 1;
  if (contentBlocks.key_studies) score += 1;
  if (contentBlocks.research_evidence) score += 0.5;
  
  // Bonus for practical content
  if (contentBlocks.practical_takeaways) score += 0.5;
  if (contentBlocks.practical_applications) score += 0.5;
  if (contentBlocks.implementation) score += 0.5;
  
  // Bonus for well-tagged content
  if (tags.length > 2) score += 0.5;
  
  // Bonus for specific content types
  if (contentBlocks.clinical_guidelines) score += 1;
  if (contentBlocks.meta_analysis) score += 1;
  if (contentBlocks.systematic_review) score += 1;
  
  // Ensure score is between 1-5
  return Math.min(5, Math.max(1, Math.round(score * 10) / 10));
}

testReliabilityScores(); 