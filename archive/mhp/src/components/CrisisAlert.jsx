'use client';

import React from 'react';

export default function CrisisAlert({ resources }) {
  return (
    <div className="bg-red-100 border border-red-400 text-red-800 px-4 py-3 rounded mb-4">
      <div className="font-bold mb-2">Crisis Resources</div>
      <ul>
        {resources.map((r, i) => (
          <li key={i} className="mb-2">
            <div className="font-semibold">{r.name}</div>
            <div className="text-sm">{r.description}</div>
            <div className="text-xs">Phone: {r.phone}</div>
            <a href={r.url} target="_blank" rel="noopener noreferrer" className="text-blue-700 underline text-xs">{r.url}</a>
          </li>
        ))}
      </ul>
    </div>
  );
} 