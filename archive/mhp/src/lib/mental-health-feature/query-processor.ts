import nlp from 'compromise';
import natural from 'natural';
import { detectCrisis } from './crisis-detector';

const medicalMappings: Record<string, string> = {
  'racing thoughts': 'insomnia',
  'can\'t concentrate': 'attention deficit',
  'feel really low': 'depression',
  'point in life': 'existential crisis',
  'happy': 'wellbeing',
  // Add more mappings as needed
};

const evidenceKeywords = [
  'systematic review',
  'meta-analysis',
  'randomized controlled trial',
  'RCT',
  'treatment',
  'intervention',
  'mental health',
  'student',
  'prevalence',
  'prevention',
  'therapy',
  'outcome',
];

const categories = [
  { type: 'sleep', keywords: ['sleep', 'insomnia', 'racing thoughts'] },
  { type: 'anxiety', keywords: ['anxiety', 'worry', 'panic'] },
  { type: 'depression', keywords: ['depression', 'low', 'hopeless'] },
  { type: 'concentration', keywords: ['concentrate', 'attention', 'focus'] },
  { type: 'existential', keywords: ['point in life', 'meaning', 'purpose'] },
  { type: 'wellbeing', keywords: ['happy', 'wellbeing', 'lifestyle'] },
];

export function processUserQuery(question: string) {
  // Lowercase and basic cleanup
  const cleaned = question.toLowerCase();
  // Crisis detection
  const crisis = detectCrisis(question);
  // Keyword extraction
  const doc = nlp(cleaned);
  let keywords = doc.nouns().out('array');
  if (keywords.length === 0) {
    keywords = cleaned.split(/\s+/).filter(w => w.length > 3);
  }
  // Map colloquial to medical
  const mapped = keywords.map((k: string) => medicalMappings[k] || k);
  // Category
  let category = 'general';
  for (const cat of categories) {
    if (cat.keywords.some(k => cleaned.includes(k))) {
      category = cat.type;
      break;
    }
  }
  // Build a focused search query (only user keywords, not evidenceKeywords)
  const mainTerms = Array.from(new Set([...keywords, ...mapped])).filter(Boolean);
  const searchQuery = mainTerms.join(' ');
  return {
    keywords: mainTerms,
    category,
    crisisDetected: crisis.crisisDetected,
    crisisTerms: crisis.crisisTerms,
    originalQuestion: question,
    searchQuery,
  };
} 