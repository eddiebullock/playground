'use client';
import React, { useState, useRef, useEffect } from 'react';
import { PaperCitation } from './PaperCitation';

export default function AcademicResearchAssistantChat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [inputAtBottom, setInputAtBottom] = useState(false);
  const [animatedPapers, setAnimatedPapers] = useState(new Set());
  const inputRef = useRef(null);
  const chatRef = useRef(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [messages]);

  // Animate input bar from center to bottom after first message
  useEffect(() => {
    if (messages.length > 0 && !inputAtBottom) {
      setTimeout(() => setInputAtBottom(true), 150);
    }
  }, [messages, inputAtBottom]);

  // Animation effect for papers
  useEffect(() => {
    const animatePapers = () => {
      const paperElements = document.querySelectorAll('.paper-card');
      paperElements.forEach((element, index) => {
        setTimeout(() => {
          element.classList.add('animate-in');
        }, index * 150); // 150ms delay between each paper
      });
      
      // Animate the show more button after papers
      const showMoreBtn = document.querySelector('.show-more-btn');
      if (showMoreBtn) {
        setTimeout(() => {
          showMoreBtn.classList.add('animate-in');
        }, paperElements.length * 150 + 200); // After all papers + 200ms
      }
    };

    // Trigger animation when papers are rendered
    if (messages.length > 0) {
      const lastMessage = messages[messages.length - 1];
      if (lastMessage?.papers && lastMessage.papers.length > 0) {
        // Small delay to ensure DOM is ready
        setTimeout(animatePapers, 100);
      }
    }
  }, [messages]);

  const sendMessage = async (showMore = false) => {
    if (!input.trim() && !showMore) return;
    
    if (!showMore) {
      setMessages(msgs => [...msgs, { role: 'user', content: input }]);
      setLoading(true);
      setError(null);
      setInput('');
    } else {
      setLoading(true);
      setError(null);
    }
    
    try {
      const res = await fetch('/api/academic-research-assistant-chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          question: showMore ? messages[messages.length - 2]?.content : input,
          showMore 
        })
      });
      if (!res.ok) {
        throw new Error('Error from server');
      }
      const data = await res.json();
      
      // Handle new response format with papers object
      const papers = data.papers || [];
      const searchTerms = data.searchTerms || [];
      const totalFound = data.totalFound || 0;
      const uniqueFound = data.uniqueFound || 0;
      const sourceDistribution = data.sourceDistribution || {};
      const hasMore = data.hasMore || false;
      
      if (showMore) {
        // Update the last assistant message with more papers
        setMessages(msgs => {
          const newMsgs = [...msgs];
          const lastMsg = newMsgs[newMsgs.length - 1];
          if (lastMsg && lastMsg.role === 'assistant') {
            lastMsg.papers = papers;
            lastMsg.hasMore = hasMore;
            lastMsg.sourceDistribution = sourceDistribution;
          }
          return newMsgs;
        });
      } else {
        setMessages(msgs => [
          ...msgs,
          {
            role: 'assistant',
            content: `Found ${papers.length} relevant papers for your query.`,
            papers: papers,
            hasMore: hasMore,
            sourceDistribution: sourceDistribution
          }
        ]);
      }
    } catch (e) {
      setError('Sorry, something went wrong.');
    }
    setLoading(false);
  };

  // Render academic response as plain text for typewriter animation
  const renderAcademicResponsePlain = (academicData) => {
    let text = '';
    if (academicData.executiveSummary) text += `Executive Summary: ${academicData.executiveSummary}\n`;
    if (academicData.currentConsensus) text += `Current Consensus: ${academicData.currentConsensus}\n`;
    if (academicData.effectSizes && academicData.effectSizes.length > 0) {
      text += 'Effect Sizes & Statistics:\n';
      academicData.effectSizes.forEach(es => {
        text += `- ${es.intervention} vs ${es.comparison}: ${es.effectSize} (${es.confidenceInterval}) [n=${es.totalN}, studies=${es.studyCount}]\n`;
      });
    }
    if (academicData.keyFindings && academicData.keyFindings.length > 0) {
      text += 'Key Findings:\n';
      academicData.keyFindings.forEach(kf => {
        text += `- ${kf.finding} [${kf.evidenceLevel}, ${kf.supportingStudies} studies, citation: ${kf.citationKey}]\n`;
      });
    }
    if (Array.isArray(academicData.methodologicalConsiderations) && academicData.methodologicalConsiderations.length > 0) {
      text += 'Methodological Considerations:\n';
      academicData.methodologicalConsiderations.forEach(m => {
        text += `- ${m}\n`;
      });
    }
    if (Array.isArray(academicData.researchGaps) && academicData.researchGaps.length > 0) {
      text += 'Research Gaps:\n';
      academicData.researchGaps.forEach(g => {
        text += `- ${g}\n`;
      });
    }
    if (academicData.practicalImplications) text += `Practical Implications: ${academicData.practicalImplications}\n`;
    if (academicData.citations && academicData.citations.length > 0) {
      text += 'Citations:\n';
      academicData.citations.forEach((c, i) => {
        text += `- ${c.title || ''} ${c.authors || ''} · ${c.journal || ''} · ${c.year || ''}\n`;
      });
    }
    if (academicData.searchStrategy) {
      const joinOrString = (val) => Array.isArray(val) ? val.join(', ') : (typeof val === 'string' ? val : '');
      text += `Search Strategy: Terms: ${joinOrString(academicData.searchStrategy.termsUsed)}; Databases: ${joinOrString(academicData.searchStrategy.databasesSearched)}; Inclusion: ${academicData.searchStrategy.inclusionCriteria}; Papers reviewed: ${academicData.searchStrategy.totalPapersReviewed}, included: ${academicData.searchStrategy.papersIncluded}\n`;
    }
    if (academicData.warnings && academicData.warnings.length > 0) {
      text += 'Warnings:\n';
      academicData.warnings.forEach(w => {
        text += `- ${w}\n`;
      });
    }
    if (academicData.references && academicData.references.length > 0) {
      text += 'References:\n';
      academicData.references.forEach(ref => {
        text += `- ${ref}\n`;
      });
    }
    return text;
  };

  // Render papers in a simple, clean format with animations
  const renderPapers = (papers, hasMore = false, sourceDistribution = {}) => {
    if (!papers || papers.length === 0) {
      return <div className="text-gray-500">No relevant papers found.</div>;
    }

    return (
      <div className="space-y-4">
        <div className="text-sm text-gray-600 mb-3 fade-in">
          Found {papers.length} relevant papers:
          {Object.keys(sourceDistribution).length > 0 && (
            <div className="text-xs text-gray-500 mt-1">
              Sources: {Object.entries(sourceDistribution).map(([source, count]) => `${source}: ${count}`).join(', ')}
            </div>
          )}
        </div>
        <div className="space-y-4">
          {papers.map((paper, index) => (
            <div 
              key={`${paper.title}-${index}`}
              className="paper-card border border-gray-200 dark:border-gray-700 rounded-lg p-4 bg-gray-50 dark:bg-gray-800"
            >
              <div className="font-semibold text-lg mb-2">
                <a 
                  href={paper.url} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-indigo-600 dark:text-indigo-400 hover:underline transition-colors duration-200"
                >
                  {paper.title}
                </a>
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                <strong>Authors:</strong> {paper.authors}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                <strong>Journal:</strong> {paper.journal} • <strong>Year:</strong> {paper.year} • <strong>Source:</strong> {paper.source}
              </div>
              {paper.abstract && (
                <div className="text-sm text-gray-700 dark:text-gray-300 mt-2">
                  <strong>Abstract:</strong> {paper.abstract.length > 300 ? paper.abstract.substring(0, 300) + '...' : paper.abstract}
                </div>
              )}
            </div>
          ))}
        </div>
        {hasMore && (
          <div className="text-center mt-4">
            <button
              onClick={() => sendMessage(true)}
              className="show-more-btn bg-indigo-600 hover:bg-indigo-700 text-white font-semibold px-6 py-2 rounded-lg"
              disabled={loading}
            >
              {loading ? (
                <span className="flex items-center gap-2">
                  <svg className="spinner h-4 w-4" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Loading...
                </span>
              ) : (
                'Show More Papers'
              )}
            </button>
          </div>
        )}
      </div>
    );
  };

  const renderAcademicResponse = (academicData) => {
    if (!academicData) return null;
    const joinOrString = (val) => Array.isArray(val) ? val.join(', ') : (typeof val === 'string' ? val : '');

    // Filter effect sizes and key findings for non-empty entries
    const nonEmptyEffectSizes = (academicData.effectSizes || []).filter(
      es => (es.intervention && es.intervention.trim() !== '') || (es.comparison && es.comparison.trim() !== '') || (es.effectSize && es.effectSize.trim() !== '')
    );
    const nonEmptyKeyFindings = (academicData.keyFindings || []).filter(
      kf => kf.finding && kf.finding.trim() !== ''
    );

    return (
      <div className="space-y-4 mt-2">
        {academicData.executiveSummary && (
          <div><strong>Executive Summary:</strong> {academicData.executiveSummary}</div>
        )}
        {academicData.currentConsensus && (
          <div><strong>Current Consensus:</strong> {academicData.currentConsensus}</div>
        )}
        {nonEmptyEffectSizes.length > 0 && (
          <div>
            <strong>Effect Sizes & Statistics:</strong>
            <ul className="list-disc ml-6">
              {nonEmptyEffectSizes.map((es, i) => (
                <li key={i}>
                  {es.intervention} vs {es.comparison}: {es.effectSize} ({es.confidenceInterval}) [n={es.totalN}, studies={es.studyCount}]
                </li>
              ))}
            </ul>
          </div>
        )}
        {nonEmptyKeyFindings.length > 0 && (
          <div>
            <strong>Key Findings:</strong>
            <ul className="list-disc ml-6">
              {nonEmptyKeyFindings.map((kf, i) => (
                <li key={i}>
                  {kf.finding} <span className="text-xs text-gray-500">[{kf.evidenceLevel}, {kf.supportingStudies} studies, citation: {kf.citationKey}]</span>
                </li>
              ))}
            </ul>
          </div>
        )}
        {Array.isArray(academicData.methodologicalConsiderations) && academicData.methodologicalConsiderations.length > 0 && (
          <div>
            <strong>Methodological Considerations:</strong>
            <ul className="list-disc ml-6">
              {academicData.methodologicalConsiderations.map((m, i) => (
                <li key={i}>{m}</li>
              ))}
            </ul>
          </div>
        )}
        {Array.isArray(academicData.researchGaps) && academicData.researchGaps.length > 0 && (
          <div>
            <strong>Research Gaps:</strong>
            <ul className="list-disc ml-6">
              {academicData.researchGaps.map((g, i) => (
                <li key={i}>{g}</li>
              ))}
            </ul>
          </div>
        )}
        {academicData.practicalImplications && (
          <div><strong>Practical Implications:</strong> {academicData.practicalImplications}</div>
        )}
        {/* References/citations section matching MentalHealthChat style */}
        {academicData.citations && academicData.citations.length > 0 && (
          <div className="mt-3">
            <details className="bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded p-2">
              <summary className="cursor-pointer text-sm text-indigo-700 dark:text-indigo-300 font-semibold">Show research citations</summary>
              {academicData.citations.map((p, j) => <PaperCitation key={j} paper={p} />)}
            </details>
          </div>
        )}
        {academicData.searchStrategy && (
          <div className="text-xs text-gray-500 mt-2">
            <strong>Search Strategy:</strong> Terms: {joinOrString(academicData.searchStrategy.termsUsed)}; Databases: {joinOrString(academicData.searchStrategy.databasesSearched)}; Inclusion: {academicData.searchStrategy.inclusionCriteria}; Papers reviewed: {academicData.searchStrategy.totalPapersReviewed}, included: {academicData.searchStrategy.papersIncluded}
          </div>
        )}
        {academicData.warnings && academicData.warnings.length > 0 && (
          <div className="bg-yellow-100 border-l-4 border-yellow-500 text-yellow-700 p-3 mt-2">
            <strong>Warnings:</strong>
            <ul className="list-disc ml-6">
              {academicData.warnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    );
  };

  // UI: input bar starts centered, animates to bottom after first message
  return (
    <div className="flex flex-col min-h-screen w-full bg-gradient-to-b from-gray-50 to-gray-200 dark:from-gray-900 dark:to-gray-800">
      {/* If no messages, center the input bar vertically */}
      {messages.length === 0 && !inputAtBottom ? (
        <div className="flex-1 flex flex-col items-center justify-center w-full transition-all duration-500 ease-in-out">
          <form
            className="w-full max-w-xl flex items-center gap-2 px-4 py-6 bg-white/95 dark:bg-gray-900/95 border border-gray-200 dark:border-gray-700 rounded-xl shadow-lg transition-all duration-500 ease-in-out"
            style={{ zIndex: 10 }}
            onSubmit={e => { e.preventDefault(); sendMessage(); }}
          >
            <input
              ref={inputRef}
              className="flex-1 rounded-lg border border-gray-300 dark:border-gray-700 px-4 py-3 text-base bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              value={input}
              onChange={e => setInput(e.target.value)}
              placeholder="Type your academic research question..."
              disabled={loading}
              autoComplete="off"
            />
            <button
              className="bg-indigo-600 hover:bg-indigo-700 text-white font-semibold px-5 py-3 rounded-lg transition-colors disabled:opacity-50"
              type="submit"
              disabled={loading || !input.trim()}
            >
              Send
            </button>
          </form>
        </div>
      ) : (
        <>
          {/* Chat area */}
          <main ref={chatRef} className="flex-1 overflow-y-auto px-0 py-4 sm:py-8 w-full transition-all duration-500 ease-in-out" style={{ paddingBottom: inputAtBottom ? 96 : 0 }}>
            <div className="max-w-2xl mx-auto w-full h-full flex flex-col">
              <div className="flex flex-col gap-4 w-full">
                {messages.map((msg, i) => (
                  <div key={i} className={`flex w-full ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`rounded-xl px-4 py-3 shadow-sm max-w-[90vw] sm:max-w-[70%] whitespace-pre-line break-words ${msg.role === 'user' ? 'bg-indigo-600 text-white dark:bg-indigo-500' : 'bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 border border-gray-200 dark:border-gray-700'}`}>
                      <div className="flex items-center gap-2 mb-1">
                        <span className={`text-xs font-semibold ${msg.role === 'user' ? 'text-indigo-200' : 'text-indigo-600 dark:text-indigo-300'}`}>{msg.role === 'user' ? 'You' : 'Academic Assistant'}</span>
                      </div>
                      <div className="text-base leading-relaxed">
                        {msg.role === 'assistant' && msg.papers ? (
                          renderPapers(msg.papers, msg.hasMore, msg.sourceDistribution)
                        ) : (
                          msg.content
                        )}
                      </div>
                    </div>
                  </div>
                ))}
                {loading && (
                  <div className="flex w-full justify-start">
                    <div className="rounded-xl px-4 py-3 shadow-sm max-w-[70%] bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 border border-gray-200 dark:border-gray-700 animate-pulse">
                      <span className="text-xs font-semibold text-indigo-600 dark:text-indigo-300 mb-1 block">Academic Assistant</span>
                      <span className="text-base">Assistant is thinking...</span>
                    </div>
                  </div>
                )}
                {error && (
                  <div className="text-red-600 text-sm mt-2">{error}</div>
                )}
              </div>
            </div>
          </main>
          {/* Input bar fixed to bottom */}
          <form
            className={`w-full flex items-center gap-2 px-4 py-3 bg-white/95 dark:bg-gray-900/95 border-t border-gray-200 dark:border-gray-700 fixed bottom-0 left-0 transition-all duration-500 ease-in-out ${inputAtBottom ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10 pointer-events-none'}`}
            style={{ zIndex: 10 }}
            onSubmit={e => { e.preventDefault(); sendMessage(); }}
          >
            <input
              ref={inputRef}
              className="flex-1 rounded-lg border border-gray-300 dark:border-gray-700 px-4 py-3 text-base bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              value={input}
              onChange={e => setInput(e.target.value)}
              placeholder="Type your academic research question..."
              disabled={loading}
              autoComplete="off"
            />
            <button
              className="bg-indigo-600 hover:bg-indigo-700 text-white font-semibold px-5 py-3 rounded-lg transition-colors disabled:opacity-50"
              type="submit"
              disabled={loading || !input.trim()}
            >
              Send
            </button>
          </form>
        </>
      )}
    </div>
  );
} 