'use client';

import { useState, useEffect } from 'react';
import { createClient } from '@/lib/supabase';
import { Database } from '@/types/supabase';
import Link from 'next/link';

type Article = Database['public']['Tables']['articles']['Row'];

const supabase = createClient();

export default function FeedPage() {
  const [articles, setArticles] = useState<Article[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedCategory, setSelectedCategory] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [placeholder, setPlaceholder] = useState('');
  const [searchAnswer, setSearchAnswer] = useState<string | null>(null);
  const [isSemanticSearch, setIsSemanticSearch] = useState(false);

  const searchTopics = [
    'neuroplasticity',
    'neurotransmitters',
    'brain circuits',
    'mental health',
    'neuroscience',
    'depression',
    'anxiety',
    'brain health',
    'psychology',
    'cognition'
  ];

  useEffect(() => {
    fetchArticles();
  }, []);

  useEffect(() => {
    if (!searchQuery) {
      setSearchAnswer(null);
      setIsSemanticSearch(false);
      fetchArticles();
      return;
    }

    // Check if the query is conversational/natural language
    const isNaturalLanguage = /^(how|what|why|when|where|do|does|can|could|would|should|tell|explain|help|i|my|me|we|our)/i.test(searchQuery) ||
                             searchQuery.includes('?') ||
                             searchQuery.length > 30;

    if (isNaturalLanguage) {
      setIsSemanticSearch(true);
      handleSearch();
    } else {
      setIsSemanticSearch(false);
      fetchArticles();
    }
  }, [selectedCategory, searchQuery]);

  useEffect(() => {
    let currentIndex = 0;
    let currentText = '';
    let isDeleting = false;
    let typingSpeed = 100;

    const animatePlaceholder = () => {
      const currentTopic = searchTopics[currentIndex];
      
      if (isDeleting) {
        currentText = currentTopic.substring(0, currentText.length - 1);
      } else {
        currentText = currentTopic.substring(0, currentText.length + 1);
      }

      setPlaceholder(currentText);

      if (!isDeleting && currentText === currentTopic) {
        typingSpeed = 2000; // Pause at the end
        isDeleting = true;
      } else if (isDeleting && currentText === '') {
        isDeleting = false;
        currentIndex = (currentIndex + 1) % searchTopics.length;
        typingSpeed = 100;
      } else {
        typingSpeed = isDeleting ? 50 : 100;
      }

      setTimeout(animatePlaceholder, typingSpeed);
    };

    animatePlaceholder();
  }, []);

  const handleSearch = () => {
    if (searchQuery.trim()) {
      fetchArticles();
    }
  };

  const fetchArticles = async () => {
    try {
      setLoading(true);
      
      if (!searchQuery) {
        // If no search query, just get recent articles
        const { data, error } = await supabase
          .from('articles')
          .select('*')
          .eq('status', 'published')
          .order('updated_at', { ascending: false })
          .limit(20);

        if (error) throw error;
        setArticles(data || []);
        return;
      }

      // Enhanced search with multiple strategies
      const searchResults = await performComprehensiveSearch(searchQuery);
      setArticles(searchResults);
      
    } catch (error) {
      console.error('Error fetching articles:', error);
      setError('Failed to load articles');
    } finally {
      setLoading(false);
    }
  };

  // Comprehensive search function
  const performComprehensiveSearch = async (query: string) => {
    const queryLower = query.toLowerCase().trim();
    const searchTerms = extractSearchTerms(queryLower);
    
    console.log('Starting hybrid search for:', query);
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
    
    // Return top results
    return scoredResults.slice(0, 25);
  };

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
    console.log('Scoring', articles.length, 'articles for query:', query);
    
    const scored = articles.map(article => {
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
      
      // 6. Category relevance
      if (selectedCategory && article.category === selectedCategory) {
        score += 10;
      }
      
      // 7. Content quality indicators
      if (summary.length > 200) score += 5;
      if (Object.keys(contentBlocks).length > 5) score += 5;
      
      // 8. Recency bonus
      const daysSinceUpdate = (Date.now() - new Date(article.updated_at).getTime()) / (1000 * 60 * 60 * 24);
      if (daysSinceUpdate < 30) score += 3;
      if (daysSinceUpdate < 7) score += 2;
      
      // 9. Synonym matching (basic)
      const synonyms = getSynonyms(query);
      synonyms.forEach(synonym => {
        if (title.includes(synonym) || summary.includes(synonym)) score += 15;
      });
      
      if (score > 0) {
        console.log(`"${article.title}" - Score: ${score} (Category: ${article.category}, Tags: ${tags.join(', ')})`);
      }
      
      return { article, score };
    })
    .filter(item => item.score > 0)
    .sort((a, b) => b.score - a.score)
    .map(item => item.article);
    
    console.log('Articles with scores > 0:', scored.length);
    return scored;
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

  const getCategoryColor = (category: Article['category']) => {
    const colors = {
      mental_health: 'bg-blue-100 text-blue-800',
      neuroscience: 'bg-purple-100 text-purple-800',
      psychology: 'bg-green-100 text-green-800',
      brain_health: 'bg-yellow-100 text-yellow-800',
      neurodiversity: 'bg-indigo-100 text-indigo-800',
      interventions: 'bg-red-100 text-red-800',
      lifestyle_factors: 'bg-teal-100 text-teal-800',
      lab_testing: 'bg-orange-100 text-orange-800',
    };
    return colors[category] || 'bg-gray-100 text-gray-800';
  };

  const formatDate = (date: string) => {
    return new Date(date).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    });
  };

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

  if (error) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-800">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Search Section */}
      <div className="bg-white shadow rounded-lg p-6 mb-8">
        <h1 className="text-3xl font-bold text-center text-gray-900 mb-8">
          Mental Health & Neuroscience Research
        </h1>
        <div className="max-w-2xl mx-auto">
          <form
            onSubmit={e => {
              e.preventDefault();
              handleSearch();
            }}
            className="relative"
          >
            <input
              type="text"
              id="search"
              className="mt-1 block w-full rounded-md border-2 border-gray-300 shadow-lg focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500 text-xl py-4 px-6 text-gray-900 placeholder-gray-500 pr-14"
              placeholder={`Search for ${placeholder}...`}
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
            />
            <button
              type="submit"
              className="absolute right-2 top-1/2 transform -translate-y-1/2 px-3 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
              aria-label="Search"
            >
              {/* Go arrow icon (right arrow SVG) */}
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-6 h-6">
                <path strokeLinecap="round" strokeLinejoin="round" d="M17.25 6.75L21 12m0 0l-3.75 5.25M21 12H3" />
              </svg>
            </button>
          </form>
          <p className="mt-2 text-sm text-gray-500">
            {isSemanticSearch 
              ? "Ask any question about mental health, neuroscience, or brain health"
              : "Search for specific topics, conditions, or treatments"}
          </p>
        </div>
        
        {/* Category Filter */}
        <div className="mt-6 max-w-2xl mx-auto">
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
          >
            <option value="">All Categories</option>
            <option value="mental_health">Mental Health</option>
            <option value="neuroscience">Neuroscience</option>
            <option value="psychology">Psychology</option>
            <option value="brain_health">Brain Health</option>
            <option value="neurodiversity">Neurodiversity</option>
            <option value="interventions">Interventions</option>
            <option value="lifestyle_factors">Lifestyle Factors</option>
            <option value="lab_testing">Lab Testing</option>
          </select>
        </div>

        {/* Search Answer */}
        {searchAnswer && (
          <div className="mt-6 max-w-2xl mx-auto">
            <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-4">
              <h3 className="text-lg font-semibold text-indigo-900 mb-2">Answer:</h3>
              <p className="text-indigo-800">{searchAnswer}</p>
            </div>
          </div>
        )}

        {/* Search Error */}
        {searchError && (
          <div className="mt-6 max-w-2xl mx-auto">
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <p className="text-red-800">{searchError}</p>
            </div>
          </div>
        )}
      </div>

      {/* Feed Section */}
      {!searchQuery && (
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-gray-900">
            Latest Updates
          </h2>
          <p className="mt-2 text-lg text-gray-600">
            Stay up to date with the latest evidence-based mental health content
          </p>
        </div>
      )}

      {/* Loading State */}
      {(loading || searchLoading) ? (
        <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
        </div>
      ) : (
        <div className="space-y-6">
          {articles.map((article) => (
            <article
              key={article.id}
              className="bg-white shadow rounded-lg overflow-hidden hover:shadow-md transition-shadow"
            >
              <div className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <span
                    className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getCategoryColor(
                      article.category
                    )}`}
                  >
                    {article.category.split('_').map(word => 
                      word.charAt(0).toUpperCase() + word.slice(1)
                    ).join(' ')}
                  </span>
                  <time className="text-sm text-gray-500">
                    Updated {formatDate(article.updated_at)}
                  </time>
                </div>
                <Link
                  href={`/articles/${article.slug}`}
                  className="block group"
                >
                  <h2 className="text-xl font-semibold text-gray-900 group-hover:text-indigo-600 transition-colors">
                    {article.title}
                  </h2>
                  <p className="mt-2 text-gray-600 line-clamp-2">
                    {article.summary}
                  </p>
                </Link>
                {article.tags && article.tags.length > 0 && (
                  <div className="mt-4 flex flex-wrap gap-2">
                    {article.tags.map((tag) => (
                      <span
                        key={tag}
                        className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-800"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </article>
          ))}
        </div>
      )}
    </div>
  );
} 