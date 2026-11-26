import { createClient } from '@supabase/supabase-js';
import OpenAI from 'openai';
import dotenv from 'dotenv';
import chalk from 'chalk';

// Load environment variables
dotenv.config();

if (!process.env.NEXT_PUBLIC_SUPABASE_URL) {
    throw new Error('NEXT_PUBLIC_SUPABASE_URL is not set in environment variables');
}

if (!process.env.SUPABASE_SERVICE_ROLE_KEY) {
    throw new Error('SUPABASE_SERVICE_ROLE_KEY is not set in environment variables');
}

if (!process.env.OPENAI_API_KEY) {
    throw new Error('OPENAI_API_KEY is not set in environment variables');
}

const supabase = createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL,
    process.env.SUPABASE_SERVICE_ROLE_KEY
);

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

// Reliability scoring criteria
const RELIABILITY_CRITERIA = {
    effectSize: {
        large: { range: '>0.5', score: 1.0, description: 'Large effects (>0.5)' },
        medium: { range: '0.3-0.5', score: 0.7, description: 'Medium effects (0.3-0.5)' },
        small: { range: '0.1-0.3', score: 0.4, description: 'Small effects (0.1-0.3)' },
        verySmall: { range: '<0.1', score: 0.2, description: 'Very small effects (<0.1)' }
    },
    studyQuality: {
        metaAnalysis: { score: 1.0, description: 'Meta-analyses/systematic reviews' },
        rct: { score: 0.8, description: 'Randomized controlled trials' },
        longitudinal: { score: 0.6, description: 'Longitudinal studies' },
        crossSectional: { score: 0.4, description: 'Cross-sectional studies' },
        caseStudy: { score: 0.2, description: 'Case studies' }
    },
    replication: {
        highlyConsistent: { score: 1.0, description: 'Highly consistent findings across studies' },
        mostlyConsistent: { score: 0.7, description: 'Mostly consistent findings' },
        mixed: { score: 0.4, description: 'Mixed results' },
        inconsistent: { score: 0.2, description: 'Inconsistent findings' }
    },
    sampleSize: {
        large: { range: '>1000', score: 1.0, description: 'Large samples (>1000 participants)' },
        medium: { range: '100-1000', score: 0.7, description: 'Medium samples (100-1000 participants)' },
        small: { range: '<100', score: 0.4, description: 'Small samples (<100 participants)' }
    }
};

async function analyzeEvidenceContent(articleTitle, evidenceSummary) {
    const prompt = `
You are an expert research analyst evaluating the reliability of evidence for mental health interventions and lifestyle factors.

Analyze the following evidence summary for "${articleTitle}" and provide a structured assessment:

EVIDENCE SUMMARY:
${evidenceSummary}

Please analyze this evidence and provide a JSON response with the following structure:

{
  "effectSize": {
    "assessment": "large|medium|small|verySmall",
    "reasoning": "Brief explanation of effect size assessment",
    "examples": "Specific effect sizes mentioned if any"
  },
  "studyQuality": {
    "assessment": "metaAnalysis|rct|longitudinal|crossSectional|caseStudy",
    "reasoning": "Brief explanation of study quality assessment",
    "examples": "Types of studies mentioned"
  },
  "replication": {
    "assessment": "highlyConsistent|mostlyConsistent|mixed|inconsistent",
    "reasoning": "Brief explanation of replication consistency",
    "examples": "Evidence of replication or consistency"
  },
  "sampleSize": {
    "assessment": "large|medium|small",
    "reasoning": "Brief explanation of sample size assessment",
    "examples": "Sample sizes mentioned if any"
  },
  "confidence": "high|medium|low",
  "notes": "Any additional observations about the evidence quality"
}

Guidelines:
- Be conservative in your assessments
- If information is unclear or missing, default to lower scores
- Focus on the actual evidence presented, not general knowledge
- Consider the overall strength and consistency of the evidence
- If multiple studies are mentioned, assess the overall pattern

Return ONLY the JSON object, no additional text.`;

    try {
        const response = await openai.chat.completions.create({
            model: 'gpt-4o-mini',
            messages: [
                {
                    role: 'system',
                    content: 'You are a research analyst expert at evaluating evidence quality. Provide only JSON responses.'
                },
                {
                    role: 'user',
                    content: prompt
                }
            ],
            temperature: 0.1,
            max_tokens: 1000
        });

        const content = response.choices[0].message.content;
        return JSON.parse(content);
    } catch (error) {
        console.error(chalk.red(`Error analyzing evidence for ${articleTitle}:`), error);
        return null;
    }
}

function calculateReliabilityScore(analysis) {
    if (!analysis) return 0.5; // Default fallback

    const effectSizeScore = RELIABILITY_CRITERIA.effectSize[analysis.effectSize?.assessment]?.score || 0.4;
    const studyQualityScore = RELIABILITY_CRITERIA.studyQuality[analysis.studyQuality?.assessment]?.score || 0.4;
    const replicationScore = RELIABILITY_CRITERIA.replication[analysis.replication?.assessment]?.score || 0.4;
    const sampleSizeScore = RELIABILITY_CRITERIA.sampleSize[analysis.sampleSize?.assessment]?.score || 0.4;

    // Weighted formula: Effect Size (40%) + Study Quality (30%) + Replication (15%) + Sample Size (15%)
    const reliabilityScore = (
        effectSizeScore * 0.4 +
        studyQualityScore * 0.3 +
        replicationScore * 0.15 +
        sampleSizeScore * 0.15
    );

    return Math.round(reliabilityScore * 100) / 100; // Round to 2 decimal places
}

async function updateReliabilityScores() {
    try {
        console.log(chalk.blue('Starting reliability score analysis...'));

        // Fetch all intervention and lifestyle articles
        const { data: articles, error: fetchError } = await supabase
            .from('articles')
            .select('id, title, category, content_blocks')
            .in('category', ['interventions', 'lifestyle_factors']);

        if (fetchError) {
            console.error(chalk.red('Error fetching articles:'), fetchError);
            return false;
        }

        console.log(chalk.green(`✅ Found ${articles.length} intervention/lifestyle articles to analyze`));

        let updated = 0;
        let failed = 0;
        const results = [];

        for (const article of articles) {
            console.log(chalk.blue(`\nAnalyzing: ${article.title}`));
            
            const contentBlocks = article.content_blocks || {};
            const evidenceSummary = contentBlocks.evidence_summary;

            if (!evidenceSummary) {
                console.log(chalk.yellow(`⚠ No evidence_summary found for ${article.title}`));
                failed++;
                continue;
            }

            try {
                // Analyze the evidence content
                const analysis = await analyzeEvidenceContent(article.title, evidenceSummary);
                
                if (!analysis) {
                    console.log(chalk.red(`✗ Failed to analyze ${article.title}`));
                    failed++;
                    continue;
                }

                // Calculate reliability score
                const reliabilityScore = calculateReliabilityScore(analysis);

                console.log(chalk.green(`✓ Analysis complete for ${article.title}`));
                console.log(chalk.cyan(`  Effect Size: ${analysis.effectSize?.assessment} (${analysis.effectSize?.reasoning})`));
                console.log(chalk.cyan(`  Study Quality: ${analysis.studyQuality?.assessment} (${analysis.studyQuality?.reasoning})`));
                console.log(chalk.cyan(`  Replication: ${analysis.replication?.assessment} (${analysis.replication?.reasoning})`));
                console.log(chalk.cyan(`  Sample Size: ${analysis.sampleSize?.assessment} (${analysis.sampleSize?.reasoning})`));
                console.log(chalk.yellow(`  Reliability Score: ${reliabilityScore}`));

                // Update the article
                const newContentBlocks = {
                    ...contentBlocks,
                    reliability_score: reliabilityScore
                };

                const { error: updateError } = await supabase
                    .from('articles')
                    .update({ content_blocks: newContentBlocks })
                    .eq('id', article.id);

                if (updateError) {
                    console.error(chalk.red(`Error updating ${article.title}:`), updateError);
                    failed++;
                } else {
                    console.log(chalk.green(`✓ Updated reliability score for ${article.title}`));
                    updated++;
                    
                    results.push({
                        title: article.title,
                        category: article.category,
                        reliability_score: reliabilityScore,
                        analysis: analysis
                    });
                }

                // Add a small delay to avoid rate limits
                await new Promise(resolve => setTimeout(resolve, 1000));

            } catch (error) {
                console.error(chalk.red(`Error processing ${article.title}:`), error);
                failed++;
            }
        }

        // Generate summary report
        console.log(chalk.green('\n✅ Reliability score analysis completed!'));
        console.log(chalk.blue(`\nSummary:`));
        console.log(chalk.blue(`- Articles analyzed: ${articles.length}`));
        console.log(chalk.blue(`- Successfully updated: ${updated}`));
        console.log(chalk.blue(`- Failed: ${failed}`));

        if (results.length > 0) {
            console.log(chalk.green('\nReliability Score Summary:'));
            
            // Group by category
            const byCategory = results.reduce((acc, result) => {
                if (!acc[result.category]) acc[result.category] = [];
                acc[result.category].push(result);
                return acc;
            }, {});

            Object.entries(byCategory).forEach(([category, categoryResults]) => {
                console.log(chalk.cyan(`\n${category.toUpperCase()}:`));
                const avgScore = categoryResults.reduce((sum, r) => sum + r.reliability_score, 0) / categoryResults.length;
                console.log(chalk.cyan(`  Average reliability score: ${avgScore.toFixed(2)}`));
                
                // Show top and bottom scores
                const sorted = categoryResults.sort((a, b) => b.reliability_score - a.reliability_score);
                console.log(chalk.cyan(`  Highest: ${sorted[0].title} (${sorted[0].reliability_score})`));
                console.log(chalk.cyan(`  Lowest: ${sorted[sorted.length - 1].title} (${sorted[sorted.length - 1].reliability_score})`));
            });

            // Save detailed results to file
            const fs = await import('fs');
            const resultsPath = './reliability-analysis-results.json';
            fs.writeFileSync(resultsPath, JSON.stringify(results, null, 2));
            console.log(chalk.green(`\nDetailed results saved to: ${resultsPath}`));
        }

        return true;
    } catch (error) {
        console.error(chalk.red('Reliability score analysis failed:'), error);
        return false;
    }
}

// Run the analysis
updateReliabilityScores().then(success => {
    if (!success) {
        console.error(chalk.red('\n❌ Reliability score analysis failed. Please check the errors above.'));
        process.exit(1);
    }
    console.log(chalk.green('\n🎉 Reliability score analysis completed successfully!'));
}); 