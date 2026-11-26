'use client';

import { useState, useEffect } from 'react';
import CollapsibleTable, { Column } from '@/components/tables/CollapsibleTable';
import { Intervention, getInterventions, getInterventionsByCondition } from '@/lib/data/interventions';
import Link from 'next/link';

type TabType = 'lifestyle' | 'clinical' | 'risk_factor';

function ReliabilityHeaderWithTooltip() {
  const [show, setShow] = useState(false);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  
  const handleMouseEnter = (e: React.MouseEvent) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;
    
    // Calculate position to center the tooltip on screen
    const tooltipWidth = 320; // w-80 = 320px
    const tooltipHeight = 200; // Approximate height
    
    let x = rect.left + rect.width / 2 - tooltipWidth / 2;
    let y = rect.bottom + 10; // 10px below the question mark
    
    // Ensure tooltip doesn't go off screen
    if (x < 10) x = 10;
    if (x + tooltipWidth > viewportWidth - 10) x = viewportWidth - tooltipWidth - 10;
    if (y + tooltipHeight > viewportHeight - 10) y = rect.top - tooltipHeight - 10;
    
    setPosition({ x, y });
    setShow(true);
  };
  
  const handleMouseLeave = () => {
    setShow(false);
  };

  return (
    <>
      <span className="relative inline-flex items-center">
        Reliability
        <span
          className="ml-1 cursor-pointer text-gray-400 hover:text-gray-600"
          onMouseEnter={handleMouseEnter}
          onMouseLeave={handleMouseLeave}
          tabIndex={0}
          onFocus={handleMouseEnter}
          onBlur={handleMouseLeave}
          aria-label="How reliability is calculated"
        >
          <svg width="14" height="14" viewBox="0 0 20 20" fill="currentColor" className="inline align-text-bottom">
            <circle cx="10" cy="10" r="9" stroke="currentColor" strokeWidth="2" fill="none" />
            <text x="10" y="15" textAnchor="middle" fontSize="12" fill="currentColor">?</text>
          </svg>
        </span>
      </span>
      
      {show && (
        <div 
          className="fixed z-[9999] bg-white border border-gray-200 rounded-lg shadow-2xl text-xs text-gray-700 pointer-events-none"
          style={{
            left: position.x,
            top: position.y,
            width: '320px',
            padding: '16px'
          }}
        >
          <div className="relative">
            {/* Arrow pointing to question mark */}
            <div 
              className="absolute w-0 h-0 border-l-4 border-r-4 border-b-4 border-transparent border-b-gray-200"
              style={{
                left: '50%',
                transform: 'translateX(-50%)',
                top: position.y < window.innerHeight / 2 ? '-4px' : 'auto',
                bottom: position.y >= window.innerHeight / 2 ? '-4px' : 'auto',
                borderTop: position.y >= window.innerHeight / 2 ? '4px solid #e5e7eb' : 'none',
                borderBottom: position.y < window.innerHeight / 2 ? '4px solid #e5e7eb' : 'none'
              }}
            ></div>
            <div 
              className="absolute w-0 h-0 border-l-4 border-r-4 border-b-4 border-transparent border-b-white"
              style={{
                left: '50%',
                transform: 'translateX(-50%)',
                top: position.y < window.innerHeight / 2 ? '-3px' : 'auto',
                bottom: position.y >= window.innerHeight / 2 ? '-3px' : 'auto',
                borderTop: position.y >= window.innerHeight / 2 ? '4px solid white' : 'none',
                borderBottom: position.y < window.innerHeight / 2 ? '4px solid white' : 'none'
              }}
            ></div>
          </div>
          
          <strong className="block mb-2">How reliability is calculated:</strong>
          <ul className="list-disc pl-4 space-y-1">
            <li>+0.8 for evidence summaries, key studies</li>
            <li>+0.3 for multiple content blocks, detailed summaries</li>
            <li>+0.3 for practical takeaways, implementation guides</li>
            <li>+0.2 for comprehensive tagging</li>
          </ul>
          <div className="mt-2 text-gray-500 border-t pt-2">
            Articles are sorted by reliability score (highest to lowest).
          </div>
        </div>
      )}
    </>
  );
}

function ReliabilityDisplay({ rating }: { rating: number | null | undefined }) {
  if (rating === null || rating === undefined) {
    return <span className="text-sm text-gray-500">N/A</span>;
  }

  // rating is already 0-1 scale from database
  const raw = Math.max(0, Math.min(1, rating));
  const percent = raw * 100;
  const stars = raw * 5;
  const fullStars = Math.floor(stars);
  const partialStar = stars - fullStars;

  return (
    <div className="flex flex-col items-start min-w-[120px]">
      <div className="flex items-center mb-1">
        {[...Array(5)].map((_, i) => {
          if (i < fullStars) {
            return (
              <svg key={i} className="h-4 w-4 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
              </svg>
            );
          } else if (i === fullStars && partialStar > 0) {
            return (
              <svg key={i} className="h-4 w-4 text-yellow-400" viewBox="0 0 20 20">
                <defs>
                  <linearGradient id={`star-gradient-${i}`} x1="0" x2="1" y1="0" y2="0">
                    <stop offset={`${partialStar * 100}%`} stopColor="#facc15" />
                    <stop offset={`${partialStar * 100}%`} stopColor="#d1d5db" />
                  </linearGradient>
                </defs>
                <path fill={`url(#star-gradient-${i})`} d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
              </svg>
            );
          } else {
            return (
              <svg key={i} className="h-4 w-4 text-gray-300" fill="currentColor" viewBox="0 0 20 20">
                <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
              </svg>
            );
          }
        })}
        <span className="ml-2 text-sm text-gray-600">{(stars).toFixed(2)}/5</span>
      </div>
      <span className="text-xs text-gray-500">Raw: {raw.toFixed(2)}</span>
      <span className="text-xs text-gray-400">({percent.toFixed(0)}%)</span>
    </div>
  );
}

const columns: Record<TabType, Column<Intervention>[]> = {
  lifestyle: [
    {
      key: 'title',
      header: 'Intervention',
      sortable: true,
      className: 'w-1/2 max-w-xs',
      render: (value: any, item: Intervention) => (
        <div className="break-words">
          <Link href={`/articles/${item.slug}`} className="text-indigo-600 hover:text-indigo-800 font-medium">
            {item.title}
          </Link>
        </div>
      ),
    },
    {
      key: 'evidenceStrength',
      header: 'Evidence Strength',
      sortable: true,
      render: (value: any, item: Intervention) => {
        const strength = item.evidenceStrength;
        const colors = {
          Strong: 'bg-green-100 text-green-800',
          Moderate: 'bg-blue-100 text-blue-800',
          Limited: 'bg-yellow-100 text-yellow-800',
          Insufficient: 'bg-gray-100 text-gray-800',
        };
        return (
          <span
            className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
              colors[strength]
            }`}
          >
            {strength}
          </span>
        );
      },
    },
    {
      key: 'reliabilityRating',
      header: <ReliabilityHeaderWithTooltip />,
      sortable: true,
      render: (value: any, item: Intervention) => (
        <ReliabilityDisplay rating={item.reliabilityRating} />
      ),
    },
  ],
  clinical: [
    {
      key: 'title',
      header: 'Intervention',
      sortable: true,
      className: 'w-1/2 max-w-xs',
      render: (value: any, item: Intervention) => (
        <div className="break-words">
          <Link href={`/articles/${item.slug}`} className="text-indigo-600 hover:text-indigo-800 font-medium">
            {item.title}
          </Link>
        </div>
      ),
    },
    {
      key: 'evidenceStrength',
      header: 'Evidence Strength',
      sortable: true,
      render: (value: any, item: Intervention) => {
        const strength = item.evidenceStrength;
        const colors = {
          Strong: 'bg-green-100 text-green-800',
          Moderate: 'bg-blue-100 text-blue-800',
          Limited: 'bg-yellow-100 text-yellow-800',
          Insufficient: 'bg-gray-100 text-gray-800',
        };
        return (
          <span
            className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
              colors[strength]
            }`}
          >
            {strength}
          </span>
        );
      },
    },
    {
      key: 'reliabilityRating',
      header: <ReliabilityHeaderWithTooltip />,
      sortable: true,
      render: (value: any, item: Intervention) => (
        <ReliabilityDisplay rating={item.reliabilityRating} />
      ),
    },
  ],
  risk_factor: [
    {
      key: 'title',
      header: 'Risk Factor',
      sortable: true,
      className: 'w-1/2 max-w-xs',
      render: (value: any, item: Intervention) => (
        <div className="break-words">
          <Link href={`/articles/${item.slug}`} className="text-indigo-600 hover:text-indigo-800 font-medium">
            {item.title}
          </Link>
        </div>
      ),
    },
    {
      key: 'evidenceStrength',
      header: 'Evidence Strength',
      sortable: true,
      className: 'w-1/4',
      render: (value: any, item: Intervention) => {
        const strength = item.evidenceStrength;
        const colors = {
          Strong: 'bg-red-100 text-red-800',
          Moderate: 'bg-orange-100 text-orange-800',
          Limited: 'bg-yellow-100 text-yellow-800',
          Insufficient: 'bg-gray-100 text-gray-800',
        };
        return (
          <span
            className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
              colors[strength]
            }`}
          >
            {strength}
          </span>
        );
      },
    },
    {
      key: 'reliabilityRating',
      header: <ReliabilityHeaderWithTooltip />,
      sortable: true,
      render: (value: any, item: Intervention) => (
        <ReliabilityDisplay rating={item.reliabilityRating} />
      ),
    },
  ],
};

const filterOptions = [
  { label: 'All Conditions', value: '' },
  { label: 'Depression', value: 'depression' },
  { label: 'Anxiety', value: 'anxiety' },
  { label: 'PTSD', value: 'ptsd' },
  { label: 'OCD', value: 'ocd' },
  { label: 'Bipolar Disorder', value: 'bipolar' },
  { label: 'Eating Disorders', value: 'eating_disorder' },
  { label: 'Sleep Disorders', value: 'sleep_disorder' },
  { label: 'Substance Use', value: 'substance_use' },
  { label: 'Personality Disorders', value: 'personality_disorder' },
  { label: 'Schizophrenia', value: 'schizophrenia' },
  { label: 'Stress', value: 'stress' },
  { label: 'Trauma', value: 'trauma' },
  { label: 'Mood Disorders', value: 'mood_disorder' },
];

const tabTitles = {
  lifestyle: 'Lifestyle Interventions',
  clinical: 'Clinical Interventions',
  risk_factor: 'Risk Factors',
};

const tabDescriptions = {
  lifestyle: 'Evidence-based lifestyle changes ranked by scientific support and effectiveness',
  clinical: 'Evidence-based clinical and therapeutic interventions ranked by scientific support and effectiveness',
  risk_factor: 'Understanding risk factors is important for prevention and early intervention. Remember that correlation does not imply causation, and many factors can be modified or mitigated.',
};

export default function EvidencePage() {
  const [activeTab, setActiveTab] = useState<TabType>('lifestyle');
  const [interventions, setInterventions] = useState<Intervention[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentFilter, setCurrentFilter] = useState<string>('');

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setCurrentFilter(''); // Reset filter when switching tabs
      try {
        const data = await getInterventions(activeTab);
        // Sort by reliability rating (highest to lowest) as default
        const sortedData = data.sort((a, b) => {
          // Handle null/undefined reliability ratings
          const aRating = a.reliabilityRating || 0;
          const bRating = b.reliabilityRating || 0;
          return bRating - aRating;
        });
        setInterventions(sortedData);
      } catch (err) {
        setError(`Failed to load ${tabTitles[activeTab].toLowerCase()}`);
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [activeTab]);

  const handleFilterChange = async (filter: string) => {
    console.log('Filter changed to:', filter, 'for tab:', activeTab);
    setCurrentFilter(filter);
    setLoading(true);
    try {
      const data = filter
        ? await getInterventionsByCondition(activeTab, filter)
        : await getInterventions(activeTab);
      console.log('Filtered data received:', data.length, 'items');
      // Sort by reliability rating (highest to lowest)
      const sortedData = data.sort((a, b) => {
        // Handle null/undefined reliability ratings
        const aRating = a.reliabilityRating || 0;
        const bRating = b.reliabilityRating || 0;
        return bRating - aRating;
      });
      setInterventions(sortedData);
    } catch (err) {
      console.error('Filter error:', err);
      setError(`Failed to filter ${tabTitles[activeTab].toLowerCase()}`);
    } finally {
      setLoading(false);
    }
  };

  if (error) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-800">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Evidence-Based Mental Health
        </h1>
        <p className="text-xl text-gray-600 leading-relaxed">
          Comprehensive collection of evidence-based interventions, lifestyle factors, and risk factors ranked by scientific reliability
        </p>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200 mb-8">
        <nav className="-mb-px flex space-x-8">
          {Object.entries(tabTitles).map(([key, title]) => (
            <button
              key={key}
              onClick={() => setActiveTab(key as TabType)}
              className={`
                whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm transition-colors
                ${
                  activeTab === key
                    ? 'border-indigo-500 text-indigo-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }
              `}
            >
              {title}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Description */}
      <div className="mb-8">
        <p className="text-lg text-gray-600 leading-relaxed">
          {tabDescriptions[activeTab]}
        </p>
        {activeTab === 'risk_factor' && (
          <div className="mt-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
            <p className="text-blue-800 text-sm">
              Note: This information is presented for educational purposes. 
              If you're concerned about your mental health, please consult with a healthcare professional.
            </p>
          </div>
        )}
      </div>

      {loading ? (
        <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
        </div>
      ) : (
        <CollapsibleTable
          data={interventions}
          columns={columns[activeTab]}
          title={tabTitles[activeTab]}
          filterOptions={filterOptions}
          onFilterChange={handleFilterChange}
          className="mb-8"
          key={activeTab}
          defaultFilter={currentFilter}
        />
      )}
    </div>
  );
} 