import OpenAI from 'openai';

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

const disclaimer = 'This information is for educational purposes only and is not a substitute for professional medical advice. If you are in crisis, please seek immediate help.';

const systemPrompt = 'You are a mental health assistant. Use the provided research to answer the user\'s question in a clear, supportive, and practical way. Reference only the most relevant studies. Focus on actionable advice and avoid technical jargon.';

function shortAbstract(abstract: string) {
  if (!abstract) return '';
  if (abstract.length <= 300) return abstract;
  // Try to cut at the first period after 300 chars
  const cutoff = abstract.indexOf('.', 300);
  if (cutoff !== -1) return abstract.slice(0, cutoff + 1);
  return abstract.slice(0, 300) + '...';
}

export async function synthesizeResponse(question: string, papers: any[]) {
  const context = papers.map((p, i) =>
    `Paper ${i + 1}: ${p.title} (${p.year}) by ${p.authors}. ${shortAbstract(p.abstract)}`
  ).join('\n');
  const prompt = `User question: ${question}\n\nRelevant research papers:\n${context}\n\nBased on the above, provide:\n1. A natural language summary addressing the question\n2. Evidence-based recommendations\n3. Reference at least 2-3 of the provided papers by title or number in your summary and recommendations\n4. Use the provided research as evidence and cite specific studies\n5. Provide practical, actionable advice\n6. Always include this disclaimer: ${disclaimer}\n\nFormat: Summary → Key Findings → Recommendations → Sources`;

  console.log('GPT prompt:', prompt);

  const completion = await openai.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: prompt },
    ],
    max_tokens: 600,
    temperature: 0.7,
  });
  return { response: completion.choices[0].message?.content || '', papers };
} 