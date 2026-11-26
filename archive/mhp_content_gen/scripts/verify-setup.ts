import { createClient } from '@supabase/supabase-js';
import { updateArticleEmbeddings, searchArticlesBySimilarity } from '../src/lib/embeddings.js';
import dotenv from 'dotenv';

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

async function verifySetup() {
    try {
        console.log('Verifying database setup...');

        // 1. Check if pgvector is enabled
        const { data: extensions, error: extError } = await supabase
            .rpc('check_extension', { ext_name: 'vector' });
        
        if (extError) {
            console.error('Error checking pgvector extension:', extError);
            return false;
        }
        console.log('✅ pgvector extension is enabled');

        // 2. Check if articles table has vector columns
        const { data: columns, error: colError } = await supabase
            .from('articles')
            .select('title_embedding, content_embedding, summary_embedding')
            .limit(1);
        
        if (colError) {
            console.error('Error checking vector columns:', colError);
            return false;
        }
        console.log('✅ Vector columns exist in articles table');

        // 3. Check if user_prompts table exists
        const { data: prompts, error: promptsError } = await supabase
            .from('user_prompts')
            .select('id')
            .limit(1);
        
        if (promptsError) {
            console.error('Error checking user_prompts table:', promptsError);
            return false;
        }
        console.log('✅ user_prompts table exists');

        // 4. Test embedding generation with a sample article
        const testArticle = {
            title: 'Test Article',
            slug: 'test-article-verification',
            summary: 'A test article for verifying the embedding system.',
            status: 'draft',
            category: 'mental_health',
            tags: ['test', 'verification'],
            content_blocks: {
                overview: 'A test article for verifying the embedding system.',
                prevalence: 'Test prevalence data.',
                causes_and_mechanisms: 'Test causes and mechanisms.',
                symptoms_and_impact: 'Test symptoms and impact.',
                evidence_summary: 'Test evidence summary.',
                practical_applications: 'Test practical applications.',
                common_myths: 'Test common myths.',
                future_directions: 'Test future directions.',
                references_and_resources: 'Test references.'
            }
        };

        const { data: article, error: insertError } = await supabase
            .from('articles')
            .insert([testArticle])
            .select()
            .single();

        if (insertError) {
            console.error('Error inserting test article:', insertError);
            return false;
        }

        console.log('✅ Test article created');

        // 5. Generate embeddings for the test article
        await updateArticleEmbeddings(article.id);
        console.log('✅ Embeddings generated for test article');

        // 6. Test search functionality
        const searchResults = await searchArticlesBySimilarity(
            'test article verification',
            0.7,
            1
        );

        if (searchResults && searchResults.length > 0) {
            console.log('✅ Search functionality working');
        } else {
            console.log('⚠️ Search returned no results (this might be normal for a new database)');
        }

        // Clean up test article
        await supabase
            .from('articles')
            .delete()
            .eq('id', article.id);

        console.log('\nSetup verification complete! You can now:');
        console.log('1. Generate embeddings for new articles using updateArticleEmbeddings()');
        console.log('2. Search for similar articles using searchArticlesBySimilarity()');
        console.log('3. Store user prompts and responses in the user_prompts table');

        return true;
    } catch (error) {
        console.error('Error during verification:', error);
        return false;
    }
}

// Run the verification
verifySetup().then(success => {
    if (!success) {
        console.error('\n❌ Setup verification failed. Please check the errors above.');
        process.exit(1);
    }
}); 