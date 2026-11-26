import { createClient } from '@supabase/supabase-js';
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

const supabase = createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL,
    process.env.SUPABASE_SERVICE_ROLE_KEY
);

async function migrateArticlesSimple() {
    try {
        console.log(chalk.blue('Starting simple article migration...'));

        // 1. Fetch all existing articles
        console.log(chalk.yellow('Fetching existing articles...'));
        const { data: articles, error: fetchError } = await supabase
            .from('articles')
            .select('id, title, category, content_blocks');

        if (fetchError) {
            console.error(chalk.red('Error fetching articles:'), fetchError);
            return false;
        }

        console.log(chalk.green(`✅ Found ${articles.length} articles to migrate`));

        let migrated = 0;
        let skipped = 0;

        // 2. Process each article
        for (const article of articles) {
            console.log(chalk.blue(`\nProcessing: ${article.title}`));
            
            const contentBlocks = article.content_blocks || {};
            let needsUpdate = false;
            const newContentBlocks = { ...contentBlocks };

            // Consolidate evidence fields into evidence_summary
            let evidenceSummary = '';
            if (contentBlocks.key_evidence) {
                evidenceSummary += contentBlocks.key_evidence + ' ';
                delete newContentBlocks.key_evidence;
                needsUpdate = true;
            }
            if (contentBlocks.effectiveness) {
                evidenceSummary += contentBlocks.effectiveness + ' ';
                delete newContentBlocks.effectiveness;
                needsUpdate = true;
            }
            if (contentBlocks.evidence_base) {
                evidenceSummary += contentBlocks.evidence_base + ' ';
                delete newContentBlocks.evidence_base;
                needsUpdate = true;
            }

            if (evidenceSummary) {
                newContentBlocks.evidence_summary = evidenceSummary.trim();
            }

            // Consolidate practical fields into practical_applications
            let practicalApplications = '';
            if (contentBlocks.practical_takeaways) {
                practicalApplications += contentBlocks.practical_takeaways + ' ';
                delete newContentBlocks.practical_takeaways;
                needsUpdate = true;
            }
            if (contentBlocks.practical_applications) {
                practicalApplications += contentBlocks.practical_applications + ' ';
            }

            if (practicalApplications) {
                newContentBlocks.practical_applications = practicalApplications.trim();
            }

            // Add reliability_score for intervention and lifestyle categories
            if (['interventions', 'lifestyle_factors'].includes(article.category)) {
                if (newContentBlocks.reliability_score === undefined) {
                    newContentBlocks.reliability_score = 0.5; // Default score
                    needsUpdate = true;
                }
            }

            // Update the article if changes were made
            if (needsUpdate) {
                try {
                    const { error: updateError } = await supabase
                        .from('articles')
                        .update({ content_blocks: newContentBlocks })
                        .eq('id', article.id);

                    if (updateError) {
                        console.error(chalk.red(`Error updating ${article.title}:`), updateError);
                    } else {
                        console.log(chalk.green(`✓ Migrated: ${article.title}`));
                        migrated++;
                    }
                } catch (error) {
                    console.error(chalk.red(`Error updating ${article.title}:`), error);
                }
            } else {
                console.log(chalk.yellow(`- Skipped: ${article.title} (no changes needed)`));
                skipped++;
            }
        }

        console.log(chalk.green('\n✅ Migration completed!'));
        console.log(chalk.blue(`\nMigration summary:`));
        console.log(chalk.blue(`- Articles migrated: ${migrated}`));
        console.log(chalk.blue(`- Articles skipped: ${skipped}`));
        console.log(chalk.blue(`- Total articles processed: ${articles.length}`));

        if (migrated > 0) {
            console.log(chalk.green('\nChanges made:'));
            console.log(chalk.green('- Consolidated evidence fields into evidence_summary'));
            console.log(chalk.green('- Consolidated practical fields into practical_applications'));
            console.log(chalk.green('- Added reliability_score for intervention and lifestyle categories'));
            console.log(chalk.green('- Removed old field names'));
        }

        return true;
    } catch (error) {
        console.error(chalk.red('Migration failed:'), error);
        return false;
    }
}

// Run the migration
migrateArticlesSimple().then(success => {
    if (!success) {
        console.error(chalk.red('\n❌ Article migration failed. Please check the errors above.'));
        process.exit(1);
    }
    console.log(chalk.green('\n🎉 Migration completed successfully!'));
}); 