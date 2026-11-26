import { NextRequest, NextResponse } from 'next/server';
import { processUserQuery } from '@/lib/mental-health-feature/query-processor';
import { searchAllAPIs } from '@/lib/mental-health-feature/search-all-apis';
import { rankAndSelectPapers } from '@/lib/mental-health-feature/paper-ranker';
import { synthesizeResponse } from '@/lib/mental-health-feature/response-synthesizer';
import { detectCrisis } from '@/lib/mental-health-feature/crisis-detector';
import { cacheManager } from '@/lib/mental-health-feature/cache-manager';

export async function POST(req: NextRequest) {
  try {
    const { question } = await req.json();
    if (!question) {
      return NextResponse.json({ error: 'Missing question' }, { status: 400 });
    }

    // Crisis detection
    const crisisResult = detectCrisis(question);
    if (crisisResult.crisisDetected) {
      return NextResponse.json({
        response: crisisResult.message,
        papers: [],
        searchTermsUsed: crisisResult.searchTerms,
        crisisDetected: true,
        crisisResources: crisisResult.resources,
      });
    }

    // Caching
    const cached = await cacheManager.get(question);
    if (cached) {
      return NextResponse.json(cached);
    }

    // Query processing
    const processedQuery = await processUserQuery(question);
    // Parallel API search
    const allResults = await searchAllAPIs(processedQuery);
    // Paper ranking
    const papers = await rankAndSelectPapers(allResults, processedQuery);
    // GPT response synthesis
    const gptResponse = await synthesizeResponse(question, papers);

    const response = {
      response: gptResponse.response,
      papers,
      searchTermsUsed: processedQuery.keywords,
      crisisDetected: false,
      crisisResources: [],
    };
    await cacheManager.set(question, response);
    return NextResponse.json(response);
  } catch (error) {
    return NextResponse.json({ error: error.message || 'Internal server error' }, { status: 500 });
  }
} 