type Paper = {
  title: string;
  authors: string;
  journal: string;
  year: number | string;
  url: string;
  abstract: string;
  source: string;
  [key: string]: any;
};

const studyTypeHierarchy = [
  'systematic review',
  'meta-analysis',
  'randomized controlled trial',
  'RCT',
  'case study',
];

const relevanceKeywords = [
  'anxiety', 'panic', 'exam', 'student', 'school', 'stress', 'mental health', 'intervention', 'treatment', 'therapy', 'prevention', 'outcome',
];

function scorePaper(paper: Paper, query: { keywords: string[]; category: string }) {
  let score = 0;
  const text = `${paper.title} ${paper.abstract}`.toLowerCase();
  // Strongly boost if the main user concern is in the title
  if (query.keywords && query.keywords.length > 0 && paper.title.toLowerCase().includes(query.keywords[0])) {
    score += 10;
  }
  // Relevance: keyword match in title/abstract (user keywords)
  for (const kw of query.keywords) {
    if (text.includes(kw.toLowerCase())) score += 6;
  }
  // Extra: prioritize papers with relevance keywords (lower weight)
  for (const kw of relevanceKeywords) {
    if (text.includes(kw)) score += 1;
  }
  // Recency: last 5-10 years
  const year = parseInt(paper.year as string);
  if (!isNaN(year)) {
    const now = new Date().getFullYear();
    if (year >= now - 2) score += 3;
    else if (year >= now - 5) score += 2;
    else if (year >= now - 10) score += 1;
  }
  // Study type
  for (let i = 0; i < studyTypeHierarchy.length; i++) {
    if (text.includes(studyTypeHierarchy[i])) {
      score += (studyTypeHierarchy.length - i) * 2;
      break;
    }
  }
  // Deprioritize arXiv unless highly relevant
  if (paper.source === 'arxiv' && score < 15) score -= 2;
  // TODO: Journal quality/impact factor (future)
  // TODO: Citation count (future)
  return score;
}

export function rankAndSelectPapers(allResults: Paper[], query: { keywords: string[]; category: string }) {
  // Filter out papers with empty or very short abstracts
  const filtered = allResults.filter(p => p.abstract && p.abstract.length > 40);
  const scored = filtered.map(p => ({ ...p, relevanceScore: scorePaper(p, query) }));
  // Only keep papers with a minimum relevance score
  let relevant = scored.filter(p => p.relevanceScore >= 5 && query.keywords.some(kw => {
    const t = `${p.title} ${p.abstract}`.toLowerCase();
    return t.includes(kw.toLowerCase());
  }));
  relevant.sort((a, b) => b.relevanceScore - a.relevanceScore);
  // Fallback: if no relevant papers, return top 2 highest scoring
  if (relevant.length === 0) {
    return scored.sort((a, b) => b.relevanceScore - a.relevanceScore).slice(0, 2);
  }
  return relevant.slice(0, 8);
} 