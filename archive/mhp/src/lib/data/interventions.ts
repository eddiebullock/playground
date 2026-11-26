import { createClient } from '@/lib/supabase';
import { ArticleCategory } from '@/types/supabase';
import { Database } from '@/types/supabase';

type Article = Database['public']['Tables']['articles']['Row'];

export type Intervention = {
  id: string;
  slug: string;
  title: string;
  category: ArticleCategory;
  evidenceStrength: 'Strong' | 'Moderate' | 'Limited' | 'Insufficient';
  studyCount: number;
  reliabilityRating: number | null;
  conditions: string[];
  summary: string;
  content: Article['content_blocks'];
  tags: string[];
  status: Article['status'];
};

const supabase = createClient();

// Map frontend categories to database categories and search strategies
const categoryMap = {
  lifestyle: {
    categories: ['lifestyle_factors'] as ArticleCategory[],
    keywords: ['lifestyle', 'diet', 'exercise', 'sleep', 'nutrition', 'wellness', 'fitness', 'wellbeing', 'lifestyle factor', 'modifiable', 'environment', 'social', 'physical activity', 'mindfulness', 'meditation', 'stress management', 'self-care', 'social connection', 'social support'],
    type: 'lifestyle'
  },
  clinical: {
    categories: ['interventions'] as ArticleCategory[],
    keywords: ['therapy', 'medication', 'treatment', 'intervention', 'clinical', 'therapeutic', 'cbt', 'cognitive behavioral', 'psychotherapy', 'pharmacological', 'medication', 'drug', 'therapeutic approach'],
    type: 'clinical'
  },
  risk_factor: {
    categories: ['risk_factors'] as ArticleCategory[],
    keywords: ['risk', 'factor', 'cause', 'trigger', 'vulnerability', 'predisposition', 'etiology', 'contributing factor', 'risk factor', 'causal factor', 'predisposing factor'],
    type: 'risk_factor'
  }
};

// Enhanced condition mapping with synonyms and variations
const conditionMap = {
  'depression': ['depression', 'depressive', 'mood disorder', 'major depressive disorder', 'mdd', 'dysthymia'],
  'anxiety': ['anxiety', 'anxious', 'anxiety disorder', 'generalized anxiety', 'gad', 'panic', 'worry'],
  'ptsd': ['ptsd', 'post traumatic stress', 'post-traumatic stress', 'trauma', 'traumatic stress', 'posttraumatic'],
  'ocd': ['ocd', 'obsessive compulsive', 'obsessive-compulsive', 'obsession', 'compulsion'],
  'adhd': ['adhd', 'attention deficit', 'attention-deficit', 'hyperactivity', 'attention disorder'],
  'bipolar': ['bipolar', 'manic', 'mania', 'bipolar disorder', 'manic depression'],
  'eating_disorder': ['eating disorder', 'anorexia', 'bulimia', 'binge eating', 'arfid', 'pica'],
  'sleep_disorder': ['sleep disorder', 'insomnia', 'sleep apnea', 'narcolepsy', 'sleep problem', 'sleep issue'],
  'substance_use': ['substance use', 'addiction', 'alcohol', 'drug', 'substance abuse', 'substance dependence'],
  'personality_disorder': ['personality disorder', 'borderline', 'narcissistic', 'antisocial', 'avoidant', 'dependent'],
  'schizophrenia': ['schizophrenia', 'psychosis', 'psychotic', 'schizoaffective', 'delusional'],
  'autism': ['autism', 'autistic', 'asd', 'autism spectrum', 'neurodiversity'],
  'stress': ['stress', 'stressed', 'stressful', 'stress management', 'stress reduction'],
  'trauma': ['trauma', 'traumatic', 'trauma therapy', 'trauma treatment', 'trauma-informed'],
  'mood_disorder': ['mood disorder', 'mood', 'emotional regulation', 'mood regulation', 'mood stabilization']
};

export async function getInterventions(category: keyof typeof categoryMap) {
  const config = categoryMap[category];
  
  console.log(`Fetching ${category} interventions...`);
  
  try {
    // Get articles from relevant categories
    const { data, error } = await supabase
      .from('articles')
      .select('*')
      .eq('status', 'published')
      .in('category', config.categories)
      .order('created_at', { ascending: false })
      .limit(200); // Get more articles to work with

    if (error) {
      console.error('Error fetching interventions:', error);
      throw error;
    }

    console.log(`Found ${data?.length || 0} articles in categories: ${config.categories.join(', ')}`);

    // Filter and score articles based on category relevance
    const relevantArticles = data?.filter(article => {
      const title = article.title.toLowerCase();
      const summary = article.summary.toLowerCase();
      const tags = (article.tags || []).map((t: string) => t.toLowerCase());
      const contentText = JSON.stringify(article.content_blocks || {}).toLowerCase();
      
      // Check if article matches category keywords
      return config.keywords.some(keyword => 
        title.includes(keyword) || 
        summary.includes(keyword) || 
        tags.some((tag: string) => tag.includes(keyword)) ||
        contentText.includes(keyword)
      );
    }) || [];

    console.log(`Filtered to ${relevantArticles.length} relevant ${category} articles`);

    // Transform the data to match our Intervention type
    return relevantArticles.map((article): Intervention => {
      // Check if there's a stored reliability score in content_blocks
      const storedReliability = article.content_blocks?.reliability_score;
      const reliabilityRating = storedReliability ? parseFloat(storedReliability) : null;
      
      return {
        id: article.id,
        slug: article.slug,
        title: article.title,
        category: article.category,
        evidenceStrength: determineEvidenceStrength(article),
        studyCount: countStudies(article),
        reliabilityRating: reliabilityRating,
        conditions: article.tags || [],
        summary: article.summary || '',
        content: article.content_blocks,
        tags: article.tags || [],
        status: article.status,
      };
    });

  } catch (error) {
    console.error('Error in getInterventions:', error);
    throw error;
  }
}

export async function getInterventionsByCondition(
  category: keyof typeof categoryMap,
  condition: string
) {
  const config = categoryMap[category];
  
  console.log(`Filtering ${category} for condition: ${condition}`);
  
  try {
    // Get articles from relevant categories
    const { data, error } = await supabase
      .from('articles')
      .select('*')
      .eq('status', 'published')
      .in('category', config.categories)
      .order('created_at', { ascending: false })
      .limit(200);

    if (error) {
      console.error('Error fetching interventions by condition:', error);
      throw error;
    }

    console.log(`Found ${data?.length || 0} articles in categories: ${config.categories.join(', ')}`);

    // Get condition keywords
    const conditionKeywords = conditionMap[condition as keyof typeof conditionMap] || [condition];
    console.log(`Searching for condition keywords: ${conditionKeywords.join(', ')}`);

    // Filter by condition with comprehensive matching
    const filteredData = data?.filter((article) => {
      const title = article.title.toLowerCase();
      const summary = article.summary.toLowerCase();
      const tags = (article.tags || []).map((t: string) => t.toLowerCase());
      const contentText = JSON.stringify(article.content_blocks || {}).toLowerCase();
      
      // Check if article matches category keywords
      const categoryMatch = config.keywords.some(keyword => 
        title.includes(keyword) || 
        summary.includes(keyword) || 
        tags.some((tag: string) => tag.includes(keyword)) ||
        contentText.includes(keyword)
      );
      
      // Check if article matches condition keywords
      const conditionMatch = conditionKeywords.some(keyword => 
        title.includes(keyword) || 
        summary.includes(keyword) || 
        tags.some((tag: string) => tag.includes(keyword)) ||
        contentText.includes(keyword)
      );
      
      const isMatch = categoryMatch && conditionMatch;
      
      if (isMatch) {
        console.log(`Match found: ${article.title} (tags: ${article.tags?.join(', ')})`);
      }
      
      return isMatch;
    }) || [];

    console.log(`Filtered to ${filteredData.length} articles for condition: ${condition}`);

    return filteredData.map((article): Intervention => {
      // Check if there's a stored reliability score in content_blocks
      const storedReliability = article.content_blocks?.reliability_score;
      const reliabilityRating = storedReliability ? parseFloat(storedReliability) : null;
      
      return {
        id: article.id,
        slug: article.slug,
        title: article.title,
        category: article.category,
        evidenceStrength: determineEvidenceStrength(article),
        studyCount: countStudies(article),
        reliabilityRating: reliabilityRating,
        conditions: article.tags || [],
        summary: article.summary || '',
        content: article.content_blocks,
        tags: article.tags || [],
        status: article.status,
      };
    });

  } catch (error) {
    console.error('Error in getInterventionsByCondition:', error);
    throw error;
  }
}

// Helper functions for determining evidence strength and reliability
function determineEvidenceStrength(article: Article): 'Strong' | 'Moderate' | 'Limited' | 'Insufficient' {
  const evidenceText = article.content_blocks?.evidence_summary?.toLowerCase() || '';
  const keyStudiesText = article.content_blocks?.key_studies?.toLowerCase() || '';
  
  if (evidenceText.includes('strong') || evidenceText.includes('robust') || evidenceText.includes('extensive')) {
    return 'Strong';
  } else if (evidenceText.includes('moderate') || evidenceText.includes('good') || evidenceText.includes('substantial')) {
    return 'Moderate';
  } else if (evidenceText.includes('limited') || evidenceText.includes('some') || evidenceText.includes('emerging')) {
    return 'Limited';
  } else if (keyStudiesText.split('\n').length > 5) {
    return 'Moderate'; // If there are many studies mentioned, assume moderate evidence
  } else {
    return 'Insufficient';
  }
}

function countStudies(article: Article): number {
  const keyStudiesText = article.content_blocks?.key_studies || '';
  if (!keyStudiesText) return 0;
  
  // Count lines that look like study references
  const lines = keyStudiesText.split('\n').filter(line => 
    line.trim().length > 0 && 
    (line.includes('(') || line.includes('et al') || line.includes('study') || line.includes('research'))
  );
  
  return Math.max(lines.length, 1); // At least 1 study
}

function determineReliability(article: Article): number {
  // Base reliability on content quality indicators
  let score = 0.5; // Start with a lower base score
  
  const contentBlocks = article.content_blocks || {};
  const summary = article.summary || '';
  const tags = article.tags || [];
  
  // Bonus for comprehensive content
  const contentBlockCount = Object.keys(contentBlocks).length;
  if (contentBlockCount > 2) score += 0.1;
  if (contentBlockCount > 4) score += 0.1;
  if (contentBlockCount > 6) score += 0.1;
  
  // Bonus for detailed summary
  if (summary.length > 100) score += 0.05;
  if (summary.length > 250) score += 0.05;
  if (summary.length > 500) score += 0.05;
  
  // Bonus for evidence-based content (most important)
  if (contentBlocks.evidence_summary) score += 0.15;
  if (contentBlocks.key_studies) score += 0.15;
  if (contentBlocks.key_studies_and_theories) score += 0.1;
  
  // Bonus for practical content
  if (contentBlocks.practical_takeaways) score += 0.05;
  if (contentBlocks.practical_applications) score += 0.05;
  
  // Bonus for well-tagged content
  if (tags.length > 1) score += 0.02;
  if (tags.length > 3) score += 0.02;
  if (tags.length > 5) score += 0.02;
  
  // Bonus for comprehensive content sections
  if (contentBlocks.overview) score += 0.02;
  if (contentBlocks.definition) score += 0.02;
  if (contentBlocks.symptoms_and_impact) score += 0.03;
  if (contentBlocks.causes_and_mechanisms) score += 0.03;
  if (contentBlocks.mechanisms) score += 0.03;
  if (contentBlocks.relevance) score += 0.02;
  if (contentBlocks.effectiveness) score += 0.05;
  if (contentBlocks.evidence_base) score += 0.05;
  
  // Penalty for very basic content
  if (contentBlockCount <= 1) score -= 0.1;
  if (summary.length < 50) score -= 0.05;
  
  // Additional penalties for low-quality content
  if (contentBlockCount === 0) score -= 0.15;
  if (!summary || summary.length < 20) score -= 0.1;
  
  // Ensure score is between 0.1 and 1.0
  return Math.min(1.0, Math.max(0.1, Math.round(score * 100) / 100));
} 