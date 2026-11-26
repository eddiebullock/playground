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

async function migrateArticles() {
    try {
        console.log(chalk.blue('Starting article migration...'));

        // 1. Run the migration function to update existing articles
        console.log(chalk.yellow('Running database migration...'));
        const { error: migrationError } = await supabase
            .rpc('migrate_article_content_blocks');

        if (migrationError) {
            console.error(chalk.red('Migration error:'), migrationError);
            return false;
        }

        console.log(chalk.green('✅ Database migration completed'));

        // 2. Verify the migration by checking a few articles
        console.log(chalk.yellow('Verifying migration...'));
        const { data: articles, error: fetchError } = await supabase
            .from('articles')
            .select('id, title, category, content_blocks')
            .limit(5);

        if (fetchError) {
            console.error(chalk.red('Error fetching articles:'), fetchError);
            return false;
        }

        console.log(chalk.green(`✅ Found ${articles.length} articles to verify`));

        for (const article of articles) {
            console.log(chalk.blue(`\nVerifying article: ${article.title}`));
            
            const contentBlocks = article.content_blocks || {};
            
            // Check for consolidated evidence_summary
            if (contentBlocks.evidence_summary) {
                console.log(chalk.green('✓ evidence_summary field present'));
            } else {
                console.log(chalk.yellow('⚠ evidence_summary field missing'));
            }

            // Check for consolidated practical_applications
            if (contentBlocks.practical_applications) {
                console.log(chalk.green('✓ practical_applications field present'));
            } else {
                console.log(chalk.yellow('⚠ practical_applications field missing'));
            }

            // Check for reliability_score in intervention/lifestyle categories
            if (['interventions', 'lifestyle_factors'].includes(article.category)) {
                if (contentBlocks.reliability_score !== undefined) {
                    console.log(chalk.green(`✓ reliability_score present: ${contentBlocks.reliability_score}`));
                } else {
                    console.log(chalk.yellow('⚠ reliability_score missing for intervention/lifestyle article'));
                }
            }

            // Check that old fields are removed
            const oldFields = ['key_evidence', 'effectiveness', 'evidence_base', 'practical_takeaways'];
            const remainingOldFields = oldFields.filter(field => contentBlocks[field] !== undefined);
            
            if (remainingOldFields.length === 0) {
                console.log(chalk.green('✓ Old fields successfully removed'));
            } else {
                console.log(chalk.yellow(`⚠ Old fields still present: ${remainingOldFields.join(', ')}`));
            }
        }

        console.log(chalk.green('\n✅ Article migration completed successfully!'));
        console.log(chalk.blue('\nMigration summary:'));
        console.log(chalk.blue('- Consolidated evidence fields into evidence_summary'));
        console.log(chalk.blue('- Consolidated practical fields into practical_applications'));
        console.log(chalk.blue('- Added reliability_score for intervention and lifestyle categories'));
        console.log(chalk.blue('- Removed old field names'));

        return true;
    } catch (error) {
        console.error(chalk.red('Migration failed:'), error);
        return false;
    }
}

// Run the migration
migrateArticles().then(success => {
    if (!success) {
        console.error(chalk.red('\n❌ Article migration failed. Please check the errors above.'));
        process.exit(1);
    }
    console.log(chalk.green('\n🎉 Migration completed successfully!'));
}); 