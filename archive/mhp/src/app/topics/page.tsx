'use client';

import { useState, useEffect } from 'react';
import { createClient } from '@/lib/supabase';
import { Database } from '@/types/supabase';
import Link from 'next/link';
import { ChevronDownIcon, ChevronRightIcon, AcademicCapIcon, UserGroupIcon, HeartIcon, BeakerIcon, LightBulbIcon, PuzzlePieceIcon, Cog6ToothIcon, ExclamationTriangleIcon, GlobeAltIcon } from '@heroicons/react/24/outline';

type Article = Database['public']['Tables']['articles']['Row'];

const supabase = createClient();

// Define the category structure with subcategories
const categoryStructure = {
  neurodiversity: {
    name: 'Neurodiversity',
    color: 'bg-indigo-100 text-indigo-800',
    icon: UserGroupIcon,
    subcategories: {
      adhd: 'ADHD',
      autism: 'Autism',
      dyslexia: 'Dyslexia',
      dyspraxia: 'Dyspraxia',
      tourette: 'Tourette Syndrome',
    }
  },
  mental_health: {
    name: 'Mental Health',
    color: 'bg-blue-100 text-blue-800',
    icon: HeartIcon,
    subcategories: {
      depression: 'Depression',
      anxiety: 'Anxiety',
      bipolar: 'Bipolar Disorder',
      ocd: 'OCD',
      ptsd: 'PTSD',
      schizophrenia: 'Schizophrenia',
    }
  },
  lifestyle_factors: {
    name: 'Lifestyle Factors',
    color: 'bg-teal-100 text-teal-800',
    icon: GlobeAltIcon,
    subcategories: {
      sleep: 'Sleep',
      exercise: 'Exercise',
      nutrition: 'Nutrition',
      stress: 'Stress Management',
      social: 'Social Connection',
      environment: 'Environmental Factors',
    }
  },
  neuroscience: {
    name: 'Neuroscience',
    color: 'bg-purple-100 text-purple-800',
    icon: AcademicCapIcon,
    subcategories: {
      neuroplasticity: 'Neuroplasticity',
      neurotransmitters: 'Neurotransmitters',
      brain_circuits: 'Brain Circuits',
      memory: 'Memory',
      attention: 'Attention',
      emotion: 'Emotion Processing',
    }
  },
  interventions: {
    name: 'Interventions',
    color: 'bg-red-100 text-red-800',
    icon: PuzzlePieceIcon,
    subcategories: {
      therapy: 'Therapy',
      medication: 'Medication',
      lifestyle: 'Lifestyle Interventions',
      alternative: 'Alternative Treatments',
      technology: 'Technology-Based',
    }
  },
  brain_health: {
    name: 'Brain Health',
    color: 'bg-yellow-100 text-yellow-800',
    icon: LightBulbIcon,
    subcategories: {
      cognitive: 'Cognitive Health',
      aging: 'Brain Aging',
      injury: 'Brain Injury',
      optimization: 'Brain Optimization',
    }
  },
  psychology: {
    name: 'Psychology',
    color: 'bg-green-100 text-green-800',
    icon: Cog6ToothIcon,
    subcategories: {
      cognitive: 'Cognitive Psychology',
      behavioral: 'Behavioral Psychology',
      developmental: 'Developmental Psychology',
      social: 'Social Psychology',
    }
  },
  lab_testing: {
    name: 'Lab Testing',
    color: 'bg-orange-100 text-orange-800',
    icon: BeakerIcon,
    subcategories: {
      biomarkers: 'Biomarkers',
      genetics: 'Genetic Testing',
      blood: 'Blood Tests',
      imaging: 'Brain Imaging',
    }
  },
  risk_factors: {
    name: 'Risk Factors',
    color: 'bg-pink-100 text-pink-800',
    icon: ExclamationTriangleIcon,
    subcategories: {
      genetic: 'Genetic',
      environmental: 'Environmental',
      psychological: 'Psychological',
      lifestyle: 'Lifestyle',
      medical: 'Medical',
      social: 'Social',
    }
  },
};

export default function TopicsPage() {
  const [articles, setArticles] = useState<Article[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set());

  useEffect(() => {
    const fetchArticles = async () => {
      try {
        const { data, error } = await supabase
          .from('articles')
          .select('*')
          .eq('status', 'published')
          .order('title', { ascending: true });

        if (error) throw error;
        setArticles(data || []);
      } catch (err) {
        console.error('Error fetching articles:', err);
        setError('Failed to load articles');
      } finally {
        setLoading(false);
      }
    };

    fetchArticles();
  }, []);

  const toggleCategory = (category: string) => {
    const newExpanded = new Set(expandedCategories);
    if (newExpanded.has(category)) {
      newExpanded.delete(category);
    } else {
      newExpanded.add(category);
    }
    setExpandedCategories(newExpanded);
  };

  const getArticlesByCategory = (category: string) => {
    return articles.filter(article => article.category === category);
  };

  const getArticlesBySubcategory = (category: string, subcategory: string) => {
    return articles.filter(article => 
      article.category === category && 
      article.tags?.some(tag => 
        tag.toLowerCase().includes(subcategory.toLowerCase()) ||
        article.title.toLowerCase().includes(subcategory.toLowerCase())
      )
    );
  };

  const getCategoryArticles = (category: string) => {
    const categoryArticles = getArticlesByCategory(category);
    const subcategoryArticlesMap: Record<string, Article[]> = {};
    
    // Get articles for each subcategory
    Object.keys(categoryStructure[category as keyof typeof categoryStructure]?.subcategories || {}).forEach(subcategory => {
      subcategoryArticlesMap[subcategory] = getArticlesBySubcategory(category, subcategory);
    });

    // Get articles that don't fit into any subcategory
    const categorizedArticles = Object.values(subcategoryArticlesMap).flat();
    const uncategorizedArticles = categoryArticles.filter(article => 
      !categorizedArticles.some(catArticle => catArticle.id === article.id)
    );

    return { subcategoryArticles: subcategoryArticlesMap, uncategorizedArticles };
  };

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
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">
          Topics
        </h1>
        <p className="mt-2 text-lg text-gray-600">
          Explore mental health and neuroscience topics organized by category
        </p>
      </div>

      {loading ? (
        <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
        </div>
      ) : (
        <div className="space-y-6">
          {Object.entries(categoryStructure).map(([categoryKey, categoryInfo]) => {
            const { subcategoryArticles, uncategorizedArticles } = getCategoryArticles(categoryKey);
            const totalArticles = getArticlesByCategory(categoryKey).length;
            const isExpanded = expandedCategories.has(categoryKey);
            const Icon = categoryInfo.icon;

            return (
              <div key={categoryKey} className="bg-white shadow rounded-lg overflow-hidden">
                <button
                  onClick={() => toggleCategory(categoryKey)}
                  className="w-full px-6 py-4 flex items-center justify-between hover:bg-gray-50 transition-colors"
                >
                  <div className="flex items-center space-x-3">
                    <span className="inline-flex items-center justify-center rounded-full bg-gray-100 p-2 mr-2">
                      <Icon className="h-6 w-6 text-gray-500" aria-hidden="true" />
                    </span>
                    <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${categoryInfo.color}`}>
                      {categoryInfo.name}
                    </span>
                    <span className="ml-3 text-sm text-gray-500">
                      {totalArticles} article{totalArticles !== 1 ? 's' : ''}
                    </span>
                  </div>
                  {isExpanded ? (
                    <ChevronDownIcon className="h-5 w-5 text-gray-400" />
                  ) : (
                    <ChevronRightIcon className="h-5 w-5 text-gray-400" />
                  )}
                </button>

                {isExpanded && (
                  <div className="border-t border-gray-200">
                    <div className="px-6 py-4">
                      {/* Subcategories */}
                      {Object.entries(categoryInfo.subcategories).map(([subcategoryKey, subcategoryName]) => {
                        const articlesInSubcategory = subcategoryArticles[subcategoryKey] || [];
                        
                        if (articlesInSubcategory.length === 0) return null;

                        return (
                          <div key={subcategoryKey} className="mb-6">
                            <h3 className="text-lg font-semibold text-gray-900 mb-3">
                              {subcategoryName}
                            </h3>
                            <div className="space-y-2">
                              {articlesInSubcategory.map((article: Article) => (
                                <Link
                                  key={article.id}
                                  href={`/articles/${article.slug}`}
                                  className="block p-3 rounded-lg hover:bg-gray-50 transition-colors"
                                >
                                  <h4 className="font-medium text-gray-900 hover:text-indigo-600">
                                    {article.title}
                                  </h4>
                                  <p className="text-sm text-gray-600 mt-1 line-clamp-2">
                                    {article.summary}
                                  </p>
                                </Link>
                              ))}
                            </div>
                          </div>
                        );
                      })}

                      {/* Uncategorized articles */}
                      {uncategorizedArticles.length > 0 && (
                        <div className="mb-6">
                          <h3 className="text-lg font-semibold text-gray-900 mb-3">
                            Other {categoryInfo.name} Topics
                          </h3>
                          <div className="space-y-2">
                            {uncategorizedArticles.map((article) => (
                              <Link
                                key={article.id}
                                href={`/articles/${article.slug}`}
                                className="block p-3 rounded-lg hover:bg-gray-50 transition-colors"
                              >
                                <h4 className="font-medium text-gray-900 hover:text-indigo-600">
                                  {article.title}
                                </h4>
                                <p className="text-sm text-gray-600 mt-1 line-clamp-2">
                                  {article.summary}
                                </p>
                              </Link>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
} 