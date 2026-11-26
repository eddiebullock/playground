import { createClient } from '@supabase/supabase-js';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { v4 as uuidv4 } from 'uuid';
import dotenv from 'dotenv';

// Get __dirname equivalent in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load environment variables from .env
const envPath = path.resolve(process.cwd(), '.env');
console.log('Loading environment from:', envPath);
dotenv.config({ path: envPath });

// Debug environment variables
console.log('Environment variables:');
console.log('NEXT_PUBLIC_SUPABASE_URL:', process.env.NEXT_PUBLIC_SUPABASE_URL ? 'Set' : 'Not set');
console.log('SUPABASE_SERVICE_ROLE_KEY:', process.env.SUPABASE_SERVICE_ROLE_KEY ? 'Set' : 'Not set');

type ArticleCategory = 'mental_health' | 'neuroscience' | 'psychology' | 'brain_health' | 
                      'neurodiversity' | 'interventions' | 'lifestyle_factors' | 'lab_testing' | 'risk_factors';

interface BaseArticle {
  title: string;
  slug: string;
  summary: string;
  category: ArticleCategory;
  overview: string;
  future_directions: string;
  references_and_resources: string;
  status: 'published' | 'draft' | 'archived';
  tags: string[];
  faqs?: Array<{
    question: string;
    answer: string;
  }>;
  evidence_summary?: string;
  practical_applications?: string;
  content_blocks?: Record<string, string>;
}

interface MentalHealthArticle extends BaseArticle {
  category: 'mental_health';
  prevalence: string;
  causes_and_mechanisms: string;
  symptoms_and_impact: string;
  common_myths: string;
}

interface NeuroscienceArticle extends BaseArticle {
  category: 'neuroscience' | 'psychology' | 'brain_health';
  definition: string;
  mechanisms: string;
  relevance: string;
  key_studies: string;
  common_misconceptions: string;
}

interface PsychologyArticle extends BaseArticle {
  category: 'psychology';
  definition: string;
  core_principles: string;
  relevance: string;
  key_studies_and_theories: string;
  common_misconceptions: string;
}

interface NeurodiversityArticle extends BaseArticle {
  category: 'neurodiversity';
  neurodiversity_perspective: string;
  common_strengths_and_challenges: string;
  prevalence_and_demographics: string;
  mechanisms_and_understanding: string;
  common_misconceptions: string;
  lived_experience: string;
}

interface InterventionArticle extends BaseArticle {
  category: 'interventions' | 'lifestyle_factors';
  how_it_works: string;
  common_myths: string;
  risks_and_limitations: string;
  reliability_score?: number;
}

interface LabTestingArticle extends BaseArticle {
  category: 'lab_testing';
  how_it_works: string;
  applications: string;
  strengths_and_limitations: string;
  risks_and_limitations: string;
}

interface RiskFactorsArticle extends BaseArticle {
  category: 'risk_factors';
  overview: string;
  prevalence: string;
  mechanisms: string;
  evidence_summary: string;
  modifiable_factors: string;
  protective_factors: string;
  practical_takeaways: string;
  reliability_score?: number;
}

type Article = MentalHealthArticle | NeuroscienceArticle | PsychologyArticle | 
              NeurodiversityArticle | InterventionArticle | LabTestingArticle | RiskFactorsArticle;

// Updated content fields for the new schema
const ALL_CONTENT_FIELDS = [
  'overview',
  'future_directions',
  'references_and_resources',
  'evidence_summary',
  'practical_applications',
  // Mental health specific
  'prevalence',
  'causes_and_mechanisms',
  'symptoms_and_impact',
  'common_myths',
  // Neuroscience/Psychology specific
  'definition',
  'mechanisms',
  'relevance',
  'key_studies',
  'common_misconceptions',
  // Neurodiversity specific
  'neurodiversity_perspective',
  'common_strengths_and_challenges',
  'prevalence_and_demographics',
  'mechanisms_and_understanding',
  'lived_experience',
  // Intervention/Lifestyle specific
  'how_it_works',
  'risks_and_limitations',
  'reliability_score',
  // Lab testing specific
  'applications',
  'strengths_and_limitations',
  // Risk factors specific
  'modifiable_factors',
  'protective_factors',
  'practical_takeaways',
  'reliability_score'
];

// Validate article data
function validateArticle(article: any): boolean {
  const requiredFields = ['title', 'slug', 'summary', 'category'];
  const hasRequiredFields = requiredFields.every(field => article[field]);
  
  if (!hasRequiredFields) {
    console.error('Missing required fields:', requiredFields.filter(field => !article[field]));
    return false;
  }

  if (!article.content_blocks) {
    console.error('Missing content_blocks');
    return false;
  }

  return true;
}

// Transform article: always merge all content fields into content_blocks
function transformArticle(article: any) {
  const baseTransformation = {
    title: article.title,
    slug: article.slug,
    summary: article.summary,
    category: article.category,
    status: article.status || 'draft',
    tags: article.tags || [],
  };
  // Merge all content fields into content_blocks
  const content_blocks: Record<string, string> = {};
  for (const field of ALL_CONTENT_FIELDS) {
    if (article[field]) {
      content_blocks[field] = article[field];
    }
  }
  // If the article already has a content_blocks object, merge it in
  if (article.content_blocks && typeof article.content_blocks === 'object') {
    Object.assign(content_blocks, article.content_blocks);
  }
  return {
    ...baseTransformation,
    content_blocks
  };
}

async function bulkUploadArticles() {
  // Add debug logging
  console.log('Debug: Supabase URL:', process.env.NEXT_PUBLIC_SUPABASE_URL?.substring(0, 20) + '...');
  console.log('Debug: Service Role Key:', process.env.SUPABASE_SERVICE_ROLE_KEY?.substring(0, 10) + '...');

  const supabase = createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY!
  );

  // Test the connection
  const { data, error } = await supabase.from('articles').select('count').limit(1);
  if (error) {
    console.error('Debug: Supabase connection test failed:', error);
    return;
  }
  console.log('Debug: Supabase connection test successful');

  const articlesData = JSON.parse(fs.readFileSync('articles-data.json', 'utf-8'));
  let uploaded = 0, failed = 0;
  for (const article of articlesData) {
    const transformedArticle = transformArticle(article);
        if (!validateArticle(transformedArticle)) {
      console.error('Validation failed for article:', article.title);
      failed++;
          continue;
        }
    const { error: insertError } = await supabase
            .from('articles')
      .upsert(transformedArticle, { onConflict: 'slug' });
    if (insertError) {
      console.error('Error upserting article (slug: ' + transformedArticle.slug + '): ', insertError);
      failed++;
    } else {
      console.log('Article ' + transformedArticle.slug + ' upserted.');
      uploaded++;
    }
  }
  console.log('Bulk upload completed. Uploaded ' + uploaded + ' article(s) (' + failed + ' failed).');
  }

bulkUploadArticles().catch(console.error); 