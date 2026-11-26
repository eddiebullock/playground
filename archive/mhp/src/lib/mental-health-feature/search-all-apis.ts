import axios from 'axios';
import Bottleneck from 'bottleneck';
import xml2js from 'xml2js';

const PUBMED_API_KEY = process.env.PUBMED_API_KEY;
const CROSSREF_MAILTO = process.env.CROSSREF_MAILTO;
const SEMANTIC_SCHOLAR_API_KEY = process.env.SEMANTIC_SCHOLAR_API_KEY;

const limiter = new Bottleneck({ maxConcurrent: 4, minTime: 120 });

async function searchPubMed(query: string) {
  try {
    const url = `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=${encodeURIComponent(query)}&retmax=10&sort=pub+date&api_key=${PUBMED_API_KEY}&retmode=json`;
    const { data } = await limiter.schedule(() => axios.get(url));
    const ids = data.esearchresult.idlist;
    if (!ids.length) return [];
    const fetchUrl = `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=${ids.join(',')}&retmode=xml&api_key=${PUBMED_API_KEY}`;
    const fetchRes = await limiter.schedule(() => axios.get(fetchUrl));
    const parsed = await xml2js.parseStringPromise(fetchRes.data);
    const articles = parsed.PubmedArticleSet.PubmedArticle || [];
    return articles.map((a: any) => ({
      title: a.MedlineCitation[0].Article[0].ArticleTitle[0],
      authors: a.MedlineCitation[0].Article[0].AuthorList?.[0]?.Author?.map((au: any) => `${au.ForeName?.[0] || ''} ${au.LastName?.[0] || ''}`)?.join(', ') || '',
      journal: a.MedlineCitation[0].Article[0].Journal[0].Title[0],
      year: a.MedlineCitation[0].Article[0].Journal[0].JournalIssue[0].PubDate[0].Year?.[0] || '',
      url: `https://pubmed.ncbi.nlm.nih.gov/${a.MedlineCitation[0].PMID[0]._ || a.MedlineCitation[0].PMID[0]}/`,
      abstract: a.MedlineCitation[0].Article[0].Abstract?.[0]?.AbstractText?.join(' ') || '',
      source: 'pubmed',
    }));
  } catch (e) {
    return [];
  }
}

async function searchCrossRef(query: string) {
  try {
    const url = `https://api.crossref.org/works?query=${encodeURIComponent(query)}&rows=10&mailto=${CROSSREF_MAILTO}`;
    const { data } = await limiter.schedule(() => axios.get(url));
    return data.message.items.map((item: any) => ({
      title: item.title?.[0] || '',
      authors: item.author?.map((a: any) => `${a.given} ${a.family}`)?.join(', ') || '',
      journal: item['container-title']?.[0] || '',
      year: item.issued?.['date-parts']?.[0]?.[0] || '',
      url: item.URL,
      abstract: item.abstract || '',
      source: 'crossref',
    }));
  } catch (e) {
    return [];
  }
}

async function searchOpenAlex(query: string) {
  try {
    const url = `https://api.openalex.org/works?search=${encodeURIComponent(query)}&per-page=10`;
    const { data } = await limiter.schedule(() => axios.get(url));
    return data.results.map((item: any) => ({
      title: item.title,
      authors: item.authorships?.map((a: any) => a.author.display_name)?.join(', ') || '',
      journal: item.host_venue?.display_name || '',
      year: item.publication_year || '',
      url: item.id,
      abstract: item.abstract_inverted_index ? Object.keys(item.abstract_inverted_index).join(' ') : '',
      source: 'openalex',
    }));
  } catch (e) {
    return [];
  }
}

async function searchArxiv(query: string) {
  try {
    const url = `http://export.arxiv.org/api/query?search_query=all:${encodeURIComponent(query)}&start=0&max_results=10`;
    const { data } = await limiter.schedule(() => axios.get(url));
    const parsed = await xml2js.parseStringPromise(data);
    const entries = parsed.feed.entry || [];
    return entries.map((item: any) => ({
      title: item.title?.[0]?.trim() || '',
      authors: item.author?.map((a: any) => a.name?.[0])?.join(', ') || '',
      journal: 'arXiv',
      year: item.published?.[0]?.slice(0, 4) || '',
      url: item.id?.[0] || '',
      abstract: item.summary?.[0]?.trim() || '',
      source: 'arxiv',
    }));
  } catch (e) {
    return [];
  }
}

async function searchSemanticScholar(query: string) {
  try {
    const url = `https://api.semanticscholar.org/graph/v1/paper/search?query=${encodeURIComponent(query)}&limit=10&fields=title,authors,year,venue,url,abstract`;
    const { data } = await axios.get(url, {
      headers: { 'x-api-key': SEMANTIC_SCHOLAR_API_KEY }
    });
    return data.data.map((item: any) => ({
      title: item.title,
      authors: item.authors?.map((a: any) => a.name).join(', ') || '',
      journal: item.venue || '',
      year: item.year || '',
      url: item.url,
      abstract: item.abstract || '',
      source: 'semanticscholar',
    }));
  } catch (e) {
    return [];
  }
}

function dedupePapers(papers: any[]) {
  const seen = new Set();
  return papers.filter(p => {
    const key = `${p.title}|${p.authors}|${p.journal}|${p.year}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

export async function searchAllAPIs(processedQuery: { keywords: string[]; category: string; originalQuestion?: string; searchQuery?: string }) {
  const queries = [processedQuery.searchQuery || processedQuery.keywords.join(' ')];
  if (processedQuery.originalQuestion) queries.push(processedQuery.originalQuestion);
  const allResults: any[] = [];
  for (const query of queries) {
    console.log('Searching APIs with query:', query);
    const [pubmed, crossref, openalex, arxiv, semanticscholar] = await Promise.all([
      searchPubMed(query),
      searchCrossRef(query),
      searchOpenAlex(query),
      searchArxiv(query),
      searchSemanticScholar(query),
    ]);
    console.log('PubMed results:', pubmed.map((p: any) => p.title));
    console.log('CrossRef results:', crossref.map((p: any) => p.title));
    console.log('OpenAlex results:', openalex.map((p: any) => p.title));
    console.log('arXiv results:', arxiv.map((p: any) => p.title));
    console.log('Semantic Scholar results:', semanticscholar.map((p: any) => p.title));
    allResults.push(...pubmed, ...crossref, ...openalex, ...arxiv, ...semanticscholar);
  }
  return dedupePapers(allResults);
} 