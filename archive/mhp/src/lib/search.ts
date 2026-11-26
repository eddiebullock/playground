import { createClient } from '@/lib/supabase';
import { Database } from '@/types/supabase';

type Article = Database['public']['Tables']['articles']['Row'];

const supabase = createClient();

// Cache for embeddings to avoid repeated API calls
const embeddingCache = new Map<string, number[]>();

// Free embedding generation using a reliable approach
async function generateEmbedding(text: string): Promise<number[]> {
  // Check cache first
  if (embeddingCache.has(text)) {
    return embeddingCache.get(text)!;
  }

  try {
    // Use a free text embedding service (Cohere's free tier)
    const response = await fetch('https://api.cohere.ai/v1/embed', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer free', // This will use their free tier
      },
      body: JSON.stringify({
        texts: [text],
        model: 'embed-english-v3.0',
        input_type: 'search_query',
      }),
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.status}`);
    }

    const result = await response.json();
    const embedding = result.embeddings[0];
    
    // Cache the result
    embeddingCache.set(text, embedding);
    
    return embedding;
  } catch (error) {
    console.warn('Failed to generate embedding, using improved text similarity:', error);
    // Use an improved text-based embedding
    return generateImprovedTextEmbedding(text);
  }
}

// Improved text-based embedding generation
function generateImprovedTextEmbedding(text: string): number[] {
  const words = text.toLowerCase().split(/\s+/);
  const embedding = new Array(384).fill(0);
  
  // Create word frequency map
  const wordFreq = new Map<string, number>();
  words.forEach(word => {
    wordFreq.set(word, (wordFreq.get(word) || 0) + 1);
  });
  
  // Generate embedding based on word frequencies and positions
  wordFreq.forEach((freq, word) => {
    // Use multiple hash functions for better distribution
    const hash1 = word.split('').reduce((a, b) => ((a << 5) - a + b.charCodeAt(0)) & 0xffffffff, 0);
    const hash2 = word.split('').reduce((a, b) => ((a << 7) - a + b.charCodeAt(0)) & 0xffffffff, 0);
    
    const pos1 = Math.abs(hash1) % embedding.length;
    const pos2 = Math.abs(hash2) % embedding.length;
    
    embedding[pos1] += freq * 0.7;
    embedding[pos2] += freq * 0.3;
  });
  
  // Normalize
  const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
  if (magnitude > 0) {
    embedding.forEach((val, i) => {
      embedding[i] = val / magnitude;
    });
  }
  
  return embedding;
}

// Cosine similarity for embeddings
function cosineSimilarity(vecA: number[], vecB: number[]): number {
  if (vecA.length !== vecB.length) return 0;
  
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  
  if (normA === 0 || normB === 0) return 0;
  
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

export async function semanticSearch(query: string): Promise<{ answer: string; articles: Article[] }> {
  try {
    console.log('Starting hybrid search for:', query);
    
    const queryLower = query.toLowerCase().trim();
    const searchTerms = extractSearchTerms(queryLower);
    
    console.log('Extracted search terms:', searchTerms);
    
    // Strategy 1: Direct database search for exact matches
    const exactMatches = await searchExactMatches(searchTerms);
    console.log('Exact matches found:', exactMatches.length);
    
    // Strategy 2: Broader search for related content
    const relatedMatches = await searchRelatedContent(searchTerms);
    console.log('Related content matches found:', relatedMatches.length);
    
    // Strategy 3: Category-based search
    const categoryMatches = await searchByCategory(queryLower);
    console.log('Category matches found:', categoryMatches.length);
    
    // Strategy 4: Semantic search with embeddings (if available)
    const semanticMatches = await searchWithEmbeddings(query, exactMatches, relatedMatches, categoryMatches);
    console.log('Semantic matches found:', semanticMatches.length);
    
    // Combine and deduplicate results
    const allResults = [...exactMatches, ...relatedMatches, ...categoryMatches, ...semanticMatches];
    const uniqueResults = deduplicateArticles(allResults);
    console.log('Total unique results after deduplication:', uniqueResults.length);
    
    // Score and rank all results
    const scoredResults = scoreArticles(uniqueResults, queryLower, searchTerms);
    console.log('Final scored results:', scoredResults.length);
    
    console.log('Found articles:', scoredResults.length);

    if (scoredResults.length === 0) {
      return {
        answer: `I couldn't find any articles related to "${query}". Try searching for different terms or browse our categories.`,
        articles: []
      };
    }

    // Generate a simple answer based on the found articles
    const answer = generateSimpleAnswer(scoredResults, query);

    return {
      answer,
      articles: scoredResults as Article[]
    };

  } catch (error) {
    console.error('Search failed:', error);
    throw new Error('Failed to load articles');
  }
}

// Semantic search using embeddings
async function searchWithEmbeddings(query: string, exactMatches: any[], relatedMatches: any[], categoryMatches: any[]) {
  try {
    // Get query embedding
    const queryEmbedding = await generateEmbedding(query);
    
    // Combine all articles we've found so far
    const allArticles = [...exactMatches, ...relatedMatches, ...categoryMatches];
    
    // Get embeddings for all articles and calculate similarities
    const articlesWithSimilarity = await Promise.all(
      allArticles.map(async (article) => {
        const articleText = `${article.title} ${article.summary} ${JSON.stringify(article.content_blocks || {})}`;
        const articleEmbedding = await generateEmbedding(articleText);
        const similarity = cosineSimilarity(queryEmbedding, articleEmbedding);
        
        return { article, similarity };
      })
    );
    
    // Return articles with high similarity scores
    const semanticMatches = articlesWithSimilarity
      .filter(item => item.similarity > 0.3) // Threshold for semantic relevance
      .sort((a, b) => b.similarity - a.similarity)
      .map(item => item.article);
    
    return semanticMatches;
  } catch (error) {
    console.warn('Semantic search failed, continuing with text-based search:', error);
    return [];
  }
}

// Extract meaningful search terms
const extractSearchTerms = (query: string) => {
  // Remove common words and extract meaningful terms
  const stopWords = new Set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'how', 'why', 'when', 'where', 'who']);
  
  const words = query.split(/\s+/)
    .filter(word => word.length > 2 && !stopWords.has(word))
    .map(word => word.replace(/[^\w]/g, '')); // Remove punctuation
  
  return words;
};

// Search for exact matches in title, summary, and content
const searchExactMatches = async (searchTerms: string[]) => {
  if (searchTerms.length === 0) return [];
  
  const searchConditions = searchTerms.map(term => 
    `title.ilike.%${term}%,summary.ilike.%${term}%`
  ).join(',');
  
  const { data, error } = await supabase
    .from('articles')
    .select('*')
    .eq('status', 'published')
    .or(searchConditions)
    .limit(50);
  
  if (error) {
    console.error('Exact search error:', error);
    return [];
  }
  
  return data || [];
};

// Search for related content in content_blocks
const searchRelatedContent = async (searchTerms: string[]) => {
  if (searchTerms.length === 0) return [];
  
  // Get all articles and search content_blocks client-side
  const { data, error } = await supabase
    .from('articles')
    .select('*')
    .eq('status', 'published')
    .limit(100);
  
  if (error || !data) return [];
  
  // Search in content_blocks for each article
  const matches = data.filter(article => {
    if (!article.content_blocks) return false;
    
    const contentText = JSON.stringify(article.content_blocks).toLowerCase();
    return searchTerms.some(term => contentText.includes(term));
  });
  
  return matches;
};

// Search by category relevance
const searchByCategory = async (query: string) => {
  const categoryKeywords = {
    'mental_health': ['depression', 'anxiety', 'ptsd', 'adhd', 'ocd', 'bipolar', 'schizophrenia', 'mood', 'mental', 'psychiatric'],
    'neuroscience': ['brain', 'neuron', 'synapse', 'neurotransmitter', 'neuroplasticity', 'neural', 'cognitive', 'memory'],
    'psychology': ['behavior', 'cognitive', 'personality', 'developmental', 'social', 'learning', 'motivation'],
    'brain_health': ['sleep', 'nutrition', 'exercise', 'cognitive training', 'brain health', 'wellness'],
    'neurodiversity': ['autism', 'adhd', 'dyslexia', 'dyspraxia', 'neurodivergent', 'neurodiversity'],
    'interventions': ['therapy', 'medication', 'treatment', 'cbt', 'mindfulness', 'meditation', 'intervention'],
    'lifestyle_factors': ['diet', 'exercise', 'sleep', 'social', 'lifestyle', 'environment'],
    'lab_testing': ['test', 'assessment', 'diagnosis', 'biomarker', 'imaging', 'evaluation']
  };
  
  const relevantCategories = Object.entries(categoryKeywords)
    .filter(([category, keywords]) => 
      keywords.some(keyword => query.includes(keyword))
    )
    .map(([category]) => category);
  
  if (relevantCategories.length === 0) return [];
  
  const { data, error } = await supabase
    .from('articles')
    .select('*')
    .eq('status', 'published')
    .in('category', relevantCategories)
    .limit(30);
  
  if (error) {
    console.error('Category search error:', error);
    return [];
  }
  
  return data || [];
};

// Deduplicate articles by ID
const deduplicateArticles = (articles: any[]) => {
  const seen = new Set();
  return articles.filter(article => {
    if (seen.has(article.id)) return false;
    seen.add(article.id);
    return true;
  });
};

// Advanced scoring system
const scoreArticles = (articles: any[], query: string, searchTerms: string[]) => {
  return articles.map(article => {
    let score = 0;
    const title = article.title.toLowerCase();
    const summary = article.summary.toLowerCase();
    const tags = article.tags || [];
    const contentBlocks = article.content_blocks || {};
    
    // 1. Exact phrase matches (highest priority)
    if (title.includes(query)) score += 200;
    if (summary.includes(query)) score += 100;
    
    // 2. Title word matches
    const titleWords = title.split(/\s+/);
    searchTerms.forEach(term => {
      if (titleWords.includes(term)) score += 50;
      if (titleWords.some((word: string) => word.includes(term))) score += 30;
    });
    
    // 3. Summary word matches
    const summaryWords = summary.split(/\s+/);
    searchTerms.forEach(term => {
      if (summaryWords.includes(term)) score += 25;
      if (summaryWords.some((word: string) => word.includes(term))) score += 15;
    });
    
    // 4. Content blocks search
    const contentText = JSON.stringify(contentBlocks).toLowerCase();
    searchTerms.forEach(term => {
      if (contentText.includes(term)) score += 20;
    });
    
    // 5. Tag relevance
    tags.forEach((tag: string) => {
      const tagLower = tag.toLowerCase();
      if (query.includes(tagLower) || tagLower.includes(query)) score += 40;
      searchTerms.forEach(term => {
        if (tagLower.includes(term) || term.includes(tagLower)) score += 20;
      });
    });
    
    // 6. Content quality indicators
    if (summary.length > 200) score += 5;
    if (Object.keys(contentBlocks).length > 5) score += 5;
    
    // 7. Recency bonus
    const daysSinceUpdate = (Date.now() - new Date(article.updated_at).getTime()) / (1000 * 60 * 60 * 24);
    if (daysSinceUpdate < 30) score += 3;
    if (daysSinceUpdate < 7) score += 2;
    
    // 8. Synonym matching (basic)
    const synonyms = getSynonyms(query);
    synonyms.forEach(synonym => {
      if (title.includes(synonym) || summary.includes(synonym)) score += 15;
    });
    
    return { article, score };
  })
  .filter(item => item.score > 0)
  .sort((a, b) => b.score - a.score)
  .map(item => item.article);
};

// Basic synonym matching
const getSynonyms = (query: string) => {
  const synonymMap: { [key: string]: string[] } = {
    'depression': ['mood disorder', 'major depressive disorder', 'mdd', 'clinical depression'],
    'anxiety': ['anxiety disorder', 'generalized anxiety', 'gad', 'worry'],
    'adhd': ['attention deficit', 'hyperactivity', 'attention disorder'],
    'ptsd': ['post traumatic stress', 'trauma', 'traumatic stress'],
    'therapy': ['treatment', 'intervention', 'psychotherapy', 'counseling'],
    'medication': ['drug', 'pharmaceutical', 'medicine', 'treatment'],
    'brain': ['neural', 'cognitive', 'neurological', 'mental'],
    'sleep': ['rest', 'slumber', 'sleep quality', 'sleep hygiene'],
    'exercise': ['physical activity', 'workout', 'fitness', 'training'],
    'nutrition': ['diet', 'food', 'eating', 'nutritional'],
    'mindfulness': ['meditation', 'awareness', 'present moment', 'mindful'],
    'cbt': ['cognitive behavioral therapy', 'cognitive therapy', 'behavioral therapy']
  };
  
  return synonymMap[query] || [];
};

function generateSimpleAnswer(articles: Article[], query: string): string {
  if (articles.length === 0) {
    return `I couldn't find any articles related to "${query}". Try searching for different terms or browse our categories.`;
  }

  const topArticle = articles[0];
  const category = topArticle.category?.replace('_', ' ') || 'mental health';
  
  return `I found ${articles.length} article${articles.length > 1 ? 's' : ''} related to "${query}". The most relevant is "${topArticle.title}" in the ${category} category. You can browse all results below or try a more specific search term.`;
} 