import { createClient } from '@supabase/supabase-js';
import { OpenAI } from 'openai';
import dotenv from 'dotenv';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const envPath = path.resolve(process.cwd(), '.env');
console.log('Loading environment from:', envPath);
dotenv.config({ path: envPath });

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
const openaiApiKey = process.env.OPENAI_API_KEY;

if (!supabaseUrl || !supabaseKey || !openaiApiKey) {
  console.error('Missing environment variables. Please ensure NEXT_PUBLIC_SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, and OPENAI_API_KEY are set in your .env file.');
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseKey);
const openai = new OpenAI({ apiKey: openaiApiKey });

async function generateEmbedding(text: string) {
  const response = await openai.embeddings.create({ model: 'text-embedding-3-small', input: text });
  return (response.data[0].embedding);
}

async function backfillEmbeddings() {
  console.log('Fetching articles (where title_embedding, content_embedding, or summary_embedding is null) from Supabase...');
  // First, get all articles that need embeddings
  const { data: articlesToUpdate, error: fetchError } = await supabase
    .from('articles')
    .select('*')  // Select all fields
    .or('title_embedding.is.null,content_embedding.is.null,summary_embedding.is.null');

  if (fetchError) {
    console.error('Error fetching articles:', fetchError);
    return;
  }

  console.log(`Found ${articlesToUpdate?.length || 0} articles needing embeddings.`);
  let updated = 0;
  let failed = 0;

  for (const article of (articlesToUpdate || [])) {
    console.log(`Processing article: ${article.title} (${article.slug})...`);
    
    try {
      // Generate embeddings while preserving all other article data
      const titleEmbedding = article.title_embedding || await generateEmbedding(article.title);
      const contentEmbedding = article.content_embedding || await generateEmbedding(JSON.stringify(article.content_blocks || {}));
      const summaryEmbedding = article.summary_embedding || await generateEmbedding(article.summary);

      // Update only the embedding fields while preserving all other data
      const { error: updateError } = await supabase
        .from('articles')
        .update({
          title_embedding: titleEmbedding,
          content_embedding: contentEmbedding,
          summary_embedding: summaryEmbedding
        })
        .eq('id', article.id);

      if (updateError) {
        console.error(`Error updating embeddings for ${article.slug}:`, updateError);
        failed++;
      } else {
        console.log(`Successfully updated embeddings for ${article.slug}`);
        updated++;
      }
    } catch (error) {
      console.error(`Error processing article ${article.slug}:`, error);
      failed++;
    }
  }

  console.log('\nBackfill completed:');
  console.log(`- Successfully updated: ${updated} articles`);
  console.log(`- Failed to update: ${failed} articles`);
}

backfillEmbeddings().catch(console.error); 