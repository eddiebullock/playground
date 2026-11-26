import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';

// Load environment variables from .env.local
dotenv.config({ path: '.env.local' });

// Initialize Supabase client
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!supabaseUrl || !supabaseKey) {
  console.error('Missing required environment variables');
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseKey);

// Test conditions from the user's list
const testConditions = [
  'PTSD', 'OCD', 'ADHD', 'Bipolar Disorder', 'Eating Disorders', 
  'Sleep Disorders', 'Substance Use', 'Personality Disorders', 
  'Schizophrenia', 'Autism Spectrum', 'Stress', 'Trauma', 'Mood Disorders'
];

// Category configurations (same as in interventions.ts)
const categoryMap = {
  lifestyle: {
    categories: ['lifestyle_factors', 'brain_health'],
    keywords: ['lifestyle', 'diet', 'exercise', 'sleep', 'nutrition', 'wellness', 'health', 'fitness', 'wellbeing'],
    type: 'lifestyle'
  },
  clinical: {
    categories: ['interventions', 'mental_health'],
    keywords: ['therapy', 'medication', 'treatment', 'intervention', 'clinical', 'therapeutic', 'cbt', 'meditation', 'mindfulness'],
    type: 'clinical'
  },
  risk_factor: {
    categories: ['mental_health', 'neuroscience', 'psychology'],
    keywords: ['risk', 'factor', 'cause', 'trigger', 'vulnerability', 'predisposition', 'etiology'],
    type: 'risk_factor'
  }
};

// Condition mapping (same as in interventions.ts)
const conditionMap = {
  'PTSD': ['ptsd', 'post traumatic stress', 'post-traumatic stress', 'trauma', 'traumatic stress', 'posttraumatic'],
  'OCD': ['ocd', 'obsessive compulsive', 'obsessive-compulsive', 'obsession', 'compulsion'],
  'ADHD': ['adhd', 'attention deficit', 'attention-deficit', 'hyperactivity', 'attention disorder'],
  'Bipolar Disorder': ['bipolar', 'manic', 'mania', 'bipolar disorder', 'manic depression'],
  'Eating Disorders': ['eating disorder', 'anorexia', 'bulimia', 'binge eating', 'arfid', 'pica'],
  'Sleep Disorders': ['sleep disorder', 'insomnia', 'sleep apnea', 'narcolepsy', 'sleep problem', 'sleep issue'],
  'Substance Use': ['substance use', 'addiction', 'alcohol', 'drug', 'substance abuse', 'substance dependence'],
  'Personality Disorders': ['personality disorder', 'borderline', 'narcissistic', 'antisocial', 'avoidant', 'dependent'],
  'Schizophrenia': ['schizophrenia', 'psychosis', 'psychotic', 'schizoaffective', 'delusional'],
  'Autism Spectrum': ['autism', 'autistic', 'asd', 'autism spectrum', 'neurodiversity'],
  'Stress': ['stress', 'stressed', 'stressful', 'stress management', 'stress reduction'],
  'Trauma': ['trauma', 'traumatic', 'trauma therapy', 'trauma treatment', 'trauma-informed'],
  'Mood Disorders': ['mood disorder', 'mood', 'emotional regulation', 'mood regulation', 'mood stabilization']
};

async function testEvidencePageFiltering() {
  console.log('Testing Evidence Page Filtering\n');

  for (const condition of testConditions) {
    console.log(`\nTesting condition: ${condition}`);
    
    const conditionKeywords = conditionMap[condition] || [condition.toLowerCase()];
    console.log(`Keywords: ${conditionKeywords.join(', ')}`);
    
    for (const [categoryKey, config] of Object.entries(categoryMap)) {
      console.log(`\n  Category: ${categoryKey}`);
      
      try {
        // Get articles from relevant categories
        const { data, error } = await supabase
          .from('articles')
          .select('*')
          .eq('status', 'published')
          .in('category', config.categories)
          .order('created_at', { ascending: false })
          .limit(100);

        if (error) {
          console.error(`    Error: ${error.message}`);
          continue;
        }

        // Filter by condition with comprehensive matching
        const filteredData = data?.filter((article) => {
          const title = article.title.toLowerCase();
          const summary = article.summary.toLowerCase();
          const tags = (article.tags || []).map(t => t.toLowerCase());
          const contentText = JSON.stringify(article.content_blocks || {}).toLowerCase();
          
          // Check if article matches category keywords
          const categoryMatch = config.keywords.some(keyword => 
            title.includes(keyword) || 
            summary.includes(keyword) || 
            tags.some(tag => tag.includes(keyword)) ||
            contentText.includes(keyword)
          );
          
          // Check if article matches condition keywords
          const conditionMatch = conditionKeywords.some(keyword => 
            title.includes(keyword) || 
            summary.includes(keyword) || 
            tags.some(tag => tag.includes(keyword)) ||
            contentText.includes(keyword)
          );
          
          return categoryMatch && conditionMatch;
        }) || [];

        console.log(`    Found ${filteredData.length} articles`);
        
        if (filteredData.length > 0) {
          console.log(`    Examples:`);
          filteredData.slice(0, 3).forEach(article => {
            console.log(`      - ${article.title} (${article.category})`);
          });
        } else {
          console.log(`    No articles found for ${condition} in ${categoryKey} category`);
        }
        
      } catch (error) {
        console.error(`    Error testing ${categoryKey}:`, error);
      }
    }
  }
}

testEvidencePageFiltering(); 