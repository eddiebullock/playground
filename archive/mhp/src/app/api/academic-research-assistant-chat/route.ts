import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@/lib/supabase-server';
import { getRelevantPapers } from '@/lib/simple-academic-search';

export async function POST(req: NextRequest) {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  const authorizedEditors = [
    'eddie@mentalhealthprogram.co.uk',
    'daughterofthes3a@gmail.com'
    // Add more authorized editor emails here as needed
  ];
  if (!user || !authorizedEditors.includes(user.email || '')) {
    return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
  }
  const { question, showMore } = await req.json();
  if (!question) {
    return NextResponse.json({ error: 'Missing question' }, { status: 400 });
  }
  try {
    const result = await getRelevantPapers(question, showMore || false);
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: 'Failed to fetch papers', details: String(e) }, { status: 500 });
  }
} 