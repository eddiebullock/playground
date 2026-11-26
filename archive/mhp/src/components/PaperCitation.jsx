'use client';

import React, { useState } from 'react';

function stripTags(text) {
  if (!text) return '';
  // Remove XML/HTML tags
  return text.replace(/<[^>]+>/g, '').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');
}

function shortSummary(abstract) {
  if (!abstract) return '';
  const clean = stripTags(abstract).replace(/\s+/g, ' ').trim();
  if (clean.length <= 200) return clean;
  // Find the last space before or at 200 chars to avoid breaking words
  let cutoff = clean.lastIndexOf(' ', 200);
  if (cutoff === -1) cutoff = 200;
  return clean.slice(0, cutoff) + '...';
}

export default function PaperCitation({ paper }) {
  const [showFull, setShowFull] = useState(false);
  const summary = shortSummary(paper.abstract);
  const cleanFull = stripTags(paper.abstract);
  return (
    <div className="border-b py-2">
      <div className="font-semibold">{paper.title}</div>
      <div className="text-xs text-gray-600 mb-1">{paper.authors} &middot; {paper.journal} &middot; {paper.year}</div>
      <div className="text-sm mb-1">
        {showFull ? cleanFull : summary}
        {paper.abstract && cleanFull.length > summary.length && (
          <button
            className="ml-2 text-indigo-600 underline text-xs"
            onClick={() => setShowFull(v => !v)}
          >
            {showFull ? 'Show less' : 'Show more'}
          </button>
        )}
      </div>
      <a href={paper.url} target="_blank" rel="noopener noreferrer" className="text-blue-600 underline text-xs">View Paper</a>
      <div className="text-xs text-gray-400 mt-1">Relevance Score: {paper.relevanceScore?.toFixed(2)}</div>
    </div>
  );
} 