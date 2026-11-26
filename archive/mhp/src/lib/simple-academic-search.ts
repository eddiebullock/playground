import axios from 'axios';
import OpenAI from 'openai';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY || '' });
const SEMANTIC_SCHOLAR_API_KEY = process.env.SEMANTIC_SCHOLAR_API_KEY || '';

// Rate limiting for APIs
let semanticScholarLastCall = 0;
let pubmedLastCall = 0;
const SEMANTIC_SCHOLAR_DELAY = 1000; // 1 second between calls
const PUBMED_DELAY = 350; // ~3 requests per second for PubMed

// Use GPT to generate simple, effective search terms
async function generateSearchTerms(userQuery: string): Promise<string[]> {
  console.log(`[AcademicSearch] Generating search terms for: "${userQuery}"`);
  
  const prompt = `Convert this user query into 3 simple, effective search terms for academic databases like PubMed and Semantic Scholar.

User Query: "${userQuery}"

Generate search terms that are:
1. Simple and direct (avoid complex Boolean operators)
2. Use key academic/medical terminology
3. Include synonyms and related terms
4. Focus on the core concepts

Example for "depression prevention":
- depression prevention
- depression treatment prevention
- major depressive disorder prevention

Return as JSON array: ["term1", "term2", "term3"]`;

  try {
    const completion = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [{ role: 'user', content: prompt }],
      max_tokens: 200,
      temperature: 0.1,
      response_format: { type: 'json_object' }
    });

    const content = completion.choices[0].message.content;
    console.log(`[AcademicSearch] GPT search terms response: ${content}`);
    
    const parsed = JSON.parse(content);
    let terms: string[] = [];
    
    // Handle different possible response formats
    if (Array.isArray(parsed)) {
      terms = parsed.filter((t): t is string => typeof t === 'string' && !!t);
    } else if (Array.isArray(parsed.search_terms)) {
      terms = parsed.search_terms.filter((t: any): t is string => typeof t === 'string' && !!t);
    } else if (Array.isArray(parsed.terms)) {
      terms = parsed.terms.filter((t: any): t is string => typeof t === 'string' && !!t);
    } else {
      // Extract any string values from the object
      terms = Object.values(parsed).filter((v): v is string => typeof v === 'string' && !!v);
    }
    
    // Fallback if no terms found
    if (!terms.length) {
      const fallback = userQuery.toLowerCase()
        .replace(/\b(i am|i'm|looking for|research|about|regarding|what are|the best)\b/g, '')
        .trim();
      terms = [fallback];
    }
    
    // Limit to 3 terms to avoid rate limiting
    terms = terms.slice(0, 3);
    
    console.log(`[AcademicSearch] Generated search terms:`, terms);
    return terms;
  } catch (error) {
    console.error('[AcademicSearch] Failed to generate search terms:', error);
    // Fallback to simple keyword extraction
    const fallback = userQuery.toLowerCase()
      .replace(/\b(i am|i'm|looking for|research|about|regarding|what are|the best)\b/g, '')
      .trim();
    return [fallback];
  }
}

// Search Semantic Scholar with rate limiting
async function searchSemanticScholar(query: string) {
  console.log(`[AcademicSearch] Searching Semantic Scholar: "${query}"`);
  
  // Rate limiting
  const now = Date.now();
  const timeSinceLastCall = now - semanticScholarLastCall;
  if (timeSinceLastCall < SEMANTIC_SCHOLAR_DELAY) {
    const delay = SEMANTIC_SCHOLAR_DELAY - timeSinceLastCall;
    console.log(`[AcademicSearch] Rate limiting Semantic Scholar, waiting ${delay}ms`);
    await new Promise(resolve => setTimeout(resolve, delay));
  }
  semanticScholarLastCall = now;
  
  try {
    const url = `https://api.semanticscholar.org/graph/v1/paper/search?query=${encodeURIComponent(query)}&limit=20&fields=title,authors,year,venue,abstract,url,citationCount,influentialCitationCount`;
    const headers = SEMANTIC_SCHOLAR_API_KEY ? { 'x-api-key': SEMANTIC_SCHOLAR_API_KEY } : {};
    const { data } = await axios.get(url, { headers });
    
    const results = (data.data || []).map((item: any) => ({
      title: item.title,
      authors: item.authors?.map((a: any) => a.name).join(', ') || '',
      year: item.year,
      journal: item.venue,
      url: item.url,
      abstract: item.abstract,
      impact: item.citationCount + (item.influentialCitationCount || 0),
      source: 'Semantic Scholar',
      searchTerm: query
    }));
    
    console.log(`[AcademicSearch] Semantic Scholar "${query}": ${results.length} papers`);
    return results;
  } catch (error: any) {
    console.error('[AcademicSearch] Semantic Scholar error:', error);
    if (error.response?.status === 429) {
      console.log('[AcademicSearch] Semantic Scholar rate limited, skipping');
    }
    return [];
  }
}

// Search PubMed with rate limiting
async function searchPubMed(query: string) {
  console.log(`[AcademicSearch] Searching PubMed: "${query}"`);
  
  // Rate limiting for PubMed (3 requests per second)
  const now = Date.now();
  const timeSinceLastCall = now - pubmedLastCall;
  if (timeSinceLastCall < PUBMED_DELAY) {
    const delay = PUBMED_DELAY - timeSinceLastCall;
    console.log(`[AcademicSearch] Rate limiting PubMed, waiting ${delay}ms`);
    await new Promise(resolve => setTimeout(resolve, delay));
  }
  pubmedLastCall = now;
  
  try {
    const url = `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json&retmax=20&term=${encodeURIComponent(query)}`;
    const { data } = await axios.get(url);
    const ids = (data.esearchresult.idlist || []).slice(0, 20);
    
    if (!ids.length) return [];
    
    const summaryUrl = `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&retmode=json&id=${ids.join(',')}`;
    const { data: summaryData } = await axios.get(summaryUrl);
    
    const results = Object.values(summaryData.result || {})
      .filter((item: any) => item.uid)
      .map((item: any) => ({
        title: item.title,
        authors: (item.authors || []).map((a: any) => a.name).join(', '),
        year: item.pubdate ? item.pubdate.split(' ')[0] : '',
        journal: item.fulljournalname,
        url: `https://pubmed.ncbi.nlm.nih.gov/${item.uid}/`,
        abstract: '',
        impact: item.citationcount || 0,
        source: 'PubMed',
        searchTerm: query
      }));
    
    console.log(`[AcademicSearch] PubMed "${query}": ${results.length} papers`);
    return results;
  } catch (error: any) {
    console.error('[AcademicSearch] PubMed error:', error);
    if (error.response?.status === 429) {
      console.log('[AcademicSearch] PubMed rate limited, skipping');
    }
    return [];
  }
}

// Deduplicate papers based on title similarity
function deduplicatePapers(papers: any[]): any[] {
  const seen = new Set();
  const deduplicated: any[] = [];
  
  for (const paper of papers) {
    const normalizedTitle = paper.title?.toLowerCase().replace(/[^\w\s]/g, '').trim();
    if (!normalizedTitle || seen.has(normalizedTitle)) continue;
    
    seen.add(normalizedTitle);
    deduplicated.push(paper);
  }
  
  console.log(`[AcademicSearch] Deduplicated from ${papers.length} to ${deduplicated.length} papers`);
  return deduplicated;
}

// Use GPT to select the most relevant papers
async function selectTopPapers(papers: any[], userQuery: string, targetCount: number): Promise<any[]> {
  if (papers.length <= targetCount) return papers;
  
  console.log(`[AcademicSearch] Selecting top ${targetCount} papers from ${papers.length} candidates`);
  
  const prompt = `You are an expert academic researcher. Select the ${targetCount} most relevant and high-quality papers from this list for the user's query.

User Query: "${userQuery}"

Papers to evaluate (${papers.length} total):
${papers.map((p, i) => `${i + 1}. "${p.title}" (${p.year}, ${p.source}, citations: ${p.impact})`).join('\n')}

Select the ${targetCount} most relevant papers by returning their numbers as a JSON array: [1, 5, 12, ...]

Consider:
- Relevance to the user's query
- Quality (systematic reviews, meta-analyses, RCTs preferred)
- Impact (citation count)
- Recency (prefer recent papers)
- Source reliability

Return only the array of numbers.`;

  try {
    const completion = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [{ role: 'user', content: prompt }],
      max_tokens: 200,
      temperature: 0.1,
      response_format: { type: 'json_object' }
    });

    const content = completion.choices[0].message.content;
    console.log(`[AcademicSearch] GPT selection response: ${content}`);
    
    const parsed = JSON.parse(content);
    let selectedIndices: number[] = [];
    
    // Handle different possible response formats
    if (Array.isArray(parsed)) {
      selectedIndices = parsed.filter((i): i is number => typeof i === 'number' && i >= 1 && i <= papers.length);
    } else if (Array.isArray(parsed.selected)) {
      selectedIndices = parsed.selected.filter((i: any): i is number => typeof i === 'number' && i >= 1 && i <= papers.length);
    } else if (Array.isArray(parsed.indices)) {
      selectedIndices = parsed.indices.filter((i: any): i is number => typeof i === 'number' && i >= 1 && i <= papers.length);
    } else if (Array.isArray(parsed.selected_papers)) {
      selectedIndices = parsed.selected_papers.filter((i: any): i is number => typeof i === 'number' && i >= 1 && i <= papers.length);
    } else if (Array.isArray(parsed.papers)) {
      selectedIndices = parsed.papers.filter((i: any): i is number => typeof i === 'number' && i >= 1 && i <= papers.length);
    } else {
      // Try to find any array in the object
      for (const [key, value] of Object.entries(parsed)) {
        if (Array.isArray(value)) {
          selectedIndices = value.filter((i: any): i is number => typeof i === 'number' && i >= 1 && i <= papers.length);
          if (selectedIndices.length > 0) break;
        }
      }
    }
    
    // Convert to 0-based indices and get selected papers
    const selectedPapers = selectedIndices
      .map(i => i - 1)
      .filter(i => i >= 0 && i < papers.length)
      .map(i => papers[i]);
    
    console.log(`[AcademicSearch] Selected ${selectedPapers.length} papers`);
    return selectedPapers.slice(0, targetCount);
  } catch (error) {
    console.error('[AcademicSearch] Failed to select top papers:', error);
    // Fallback: return first 20 papers
    return papers.slice(0, targetCount);
  }
}

export async function getRelevantPapers(query: string, showMore: boolean = false) {
  console.log(`[AcademicSearch] Starting search for: "${query}"${showMore ? ' (showing more)' : ''}`);
  
  try {
    // Generate search terms
    const searchTerms = await generateSearchTerms(query);
    
    // Search both APIs in parallel
    const searchPromises = searchTerms.flatMap(term => [
      searchSemanticScholar(term),
      searchPubMed(term)
    ]);
    
    const allResults = await Promise.all(searchPromises);
    const allPapers = allResults.flat();
    
    console.log(`[AcademicSearch] Found ${allPapers.length} total papers`);
    
    // Log source distribution
    const sourceCounts = allPapers.reduce((acc, paper) => {
      acc[paper.source] = (acc[paper.source] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);
    console.log(`[AcademicSearch] Source distribution:`, sourceCounts);
    
    // Deduplicate
    const uniquePapers = deduplicatePapers(allPapers);
    
    // Select top papers - show more if requested
    const targetCount = showMore ? 40 : 20;
    const topPapers = await selectTopPapers(uniquePapers, query, targetCount);
    
    console.log(`[AcademicSearch] Returning ${topPapers.length} papers`);
    
    return {
      papers: topPapers,
      searchTerms: searchTerms,
      totalFound: allPapers.length,
      uniqueFound: uniquePapers.length,
      sourceDistribution: sourceCounts,
      hasMore: uniquePapers.length > targetCount
    };
    
  } catch (error) {
    console.error('[AcademicSearch] Error in getRelevantPapers:', error);
    return {
      papers: [],
      searchTerms: [],
      totalFound: 0,
      uniqueFound: 0,
      sourceDistribution: {},
      hasMore: false,
      error: 'Failed to retrieve papers'
    };
  }
} 