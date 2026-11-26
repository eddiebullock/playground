import { OpenAI } from 'openai';
import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

if (!process.env.OPENAI_API_KEY) {
    throw new Error('OPENAI_API_KEY is not set in environment variables');
}

if (!process.env.NEXT_PUBLIC_SUPABASE_URL) {
    throw new Error('NEXT_PUBLIC_SUPABASE_URL is not set in environment variables');
}

if (!process.env.SUPABASE_SERVICE_ROLE_KEY) {
    throw new Error('SUPABASE_SERVICE_ROLE_KEY is not set in environment variables');
}

// Initialize OpenAI client
const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

// Initialize Supabase client
const supabase = createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL,
    process.env.SUPABASE_SERVICE_ROLE_KEY
);

// Function to generate embeddings using OpenAI
async function generateEmbedding(text: string): Promise<number[]> {
    try {
        const response = await openai.embeddings.create({
            model: "text-embedding-ada-002",
            input: text,
        });
        return response.data[0].embedding;
    } catch (error) {
        console.error('Error generating embedding:', error);
        throw error;
    }
}

// Function to update embeddings for a single article
export async function updateArticleEmbeddings(articleId: string) {
    try {
        // Fetch the article
        const { data: article, error: fetchError } = await supabase
            .from('articles')
            .select('title, content, summary')
            .eq('id', articleId)
            .single();

        if (fetchError) throw fetchError;
        if (!article) throw new Error('Article not found');

        // Generate embeddings for each field
        const [titleEmbedding, contentEmbedding, summaryEmbedding] = await Promise.all([
            generateEmbedding(article.title),
            generateEmbedding(JSON.stringify(article.content)),
            article.summary ? generateEmbedding(article.summary) : null,
        ]);

        // Update the article with new embeddings
        const { error: updateError } = await supabase
            .from('articles')
            .update({
                title_embedding: titleEmbedding,
                content_embedding: contentEmbedding,
                summary_embedding: summaryEmbedding,
            })
            .eq('id', articleId);

        if (updateError) throw updateError;

        return { success: true };
    } catch (error) {
        console.error('Error updating article embeddings:', error);
        throw error;
    }
}

// Function to backfill embeddings for all articles
export async function backfillArticleEmbeddings() {
    try {
        // Get all articles without embeddings
        const { data: articles, error: fetchError } = await supabase
            .from('articles')
            .select('id')
            .or('title_embedding.is.null,content_embedding.is.null,summary_embedding.is.null');

        if (fetchError) throw fetchError;
        if (!articles) return { success: true, processed: 0 };

        // Process articles in batches to avoid rate limits
        const batchSize = 5;
        for (let i = 0; i < articles.length; i += batchSize) {
            const batch = articles.slice(i, i + batchSize);
            await Promise.all(
                batch.map(article => updateArticleEmbeddings(article.id))
            );
            // Add a small delay between batches to avoid rate limits
            if (i + batchSize < articles.length) {
                await new Promise(resolve => setTimeout(resolve, 1000));
            }
        }

        return { success: true, processed: articles.length };
    } catch (error) {
        console.error('Error backfilling article embeddings:', error);
        throw error;
    }
}

// Function to search articles by semantic similarity
export async function searchArticlesBySimilarity(
    query: string,
    similarityThreshold: number = 0.7,
    matchCount: number = 5
) {
    try {
        // Generate embedding for the query
        const queryEmbedding = await generateEmbedding(query);

        // Search for similar articles
        const { data: results, error } = await supabase
            .rpc('search_articles', {
                query_embedding: queryEmbedding,
                similarity_threshold: similarityThreshold,
                match_count: matchCount,
            });

        if (error) throw error;
        return results;
    } catch (error) {
        console.error('Error searching articles:', error);
        throw error;
    }
}

// Example usage:
/*
// Update embeddings for a single article
await updateArticleEmbeddings('article-uuid-here');

// Backfill embeddings for all articles
await backfillArticleEmbeddings();

// Search for similar articles
const results = await searchArticlesBySimilarity(
    'I have trouble sleeping because my mind races',
    0.7,
    5
);
*/ 