'use client';

import { useState, useEffect } from 'react';
import { createClient } from '@/lib/supabase';
import { Database } from '@/types/supabase';
import { QuestionMarkCircleIcon } from '@heroicons/react/24/outline';

type EditorStats = Database['public']['Functions']['get_editor_stats']['Returns'][0];
type EditorRanking = Database['public']['Functions']['get_editor_rankings']['Returns'][0];

interface EditorDashboardProps {
  userId: string;
}

// Tooltip component
function Tooltip({ text }: { text: string }) {
  const [show, setShow] = useState(false);
  return (
    <span className="relative inline-block align-middle">
      <button
        type="button"
        className="ml-1 text-gray-400 hover:text-indigo-600 focus:outline-none"
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
        tabIndex={0}
        aria-label="Show explanation"
      >
        <QuestionMarkCircleIcon className="h-4 w-4 inline" />
      </button>
      {show && (
        <div className="absolute z-50 left-1/2 -translate-x-1/2 mt-2 w-56 bg-white border border-gray-200 rounded-lg shadow-lg p-3 text-xs text-gray-700">
          {text}
        </div>
      )}
    </span>
  );
}

export default function EditorDashboard({ userId }: EditorDashboardProps) {
  const [stats, setStats] = useState<EditorStats | null>(null);
  const [rankings, setRankings] = useState<EditorRanking[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const supabase = createClient();

  useEffect(() => {
    const fetchEditorData = async () => {
      try {
        // Fetch editor stats
        const { data: statsData, error: statsError } = await supabase
          .rpc('get_editor_stats', { editor_user_id: userId });

        if (statsError) throw statsError;
        setStats(statsData?.[0] || null);

        // Fetch editor rankings
        const { data: rankingsData, error: rankingsError } = await supabase
          .rpc('get_editor_rankings');

        if (rankingsError) throw rankingsError;
        setRankings(rankingsData || []);
      } catch (err) {
        console.error('Error fetching editor data:', err);
        setError('Failed to load editor dashboard data');
      } finally {
        setLoading(false);
      }
    };

    fetchEditorData();
  }, [userId]);

  const getEditorLevel = (score: number) => {
    if (score >= 1000) return { level: 'Platinum', color: 'text-purple-600', bgColor: 'bg-purple-100' };
    if (score >= 500) return { level: 'Gold', color: 'text-yellow-600', bgColor: 'bg-yellow-100' };
    if (score >= 100) return { level: 'Silver', color: 'text-gray-600', bgColor: 'bg-gray-100' };
    if (score >= 10) return { level: 'Bronze', color: 'text-orange-600', bgColor: 'bg-orange-100' };
    return { level: 'Novice', color: 'text-blue-600', bgColor: 'bg-blue-100' };
  };

  const getCurrentUserRanking = () => {
    return rankings.find(ranking => ranking.editor_id === userId);
  };

  const getBadges = (stats: EditorStats) => {
    const badges = [];
    
    if (stats.total_edits >= 100) badges.push({ name: 'Century Editor', color: 'bg-purple-500' });
    else if (stats.total_edits >= 50) badges.push({ name: 'Half-Century', color: 'bg-blue-500' });
    else if (stats.total_edits >= 10) badges.push({ name: 'Decade Editor', color: 'bg-green-500' });
    
    if (stats.total_views >= 1000) badges.push({ name: 'Viral Writer', color: 'bg-red-500' });
    else if (stats.total_views >= 500) badges.push({ name: 'Popular Writer', color: 'bg-orange-500' });
    
    if (stats.total_saves >= 100) badges.push({ name: 'Bookmark Master', color: 'bg-yellow-500' });
    else if (stats.total_saves >= 50) badges.push({ name: 'Saved Favorite', color: 'bg-indigo-500' });
    
    return badges;
  };

  const getStatRanking = (statKey: keyof EditorRanking) => {
    if (!currentRanking || !rankings.length) return null;
    const sorted = [...rankings].sort((a, b) => (b[statKey] as number) - (a[statKey] as number));
    const index = sorted.findIndex(r => r.editor_id === userId);
    if (index === -1) return null;
    if (index < 10) return `You're in the top 10 editors for ${statKey.replace(/_/g, ' ')}`;
    if (index < 25) return `You're in the top 25 editors for ${statKey.replace(/_/g, ' ')}`;
    if (index < 50) return `You're in the top 50 editors for ${statKey.replace(/_/g, ' ')}`;
    if (index < 100) return `You're in the top 100 editors for ${statKey.replace(/_/g, ' ')}`;
    return null;
  };

  const getNextLevelProgress = (stats: EditorStats) => {
    const score = stats.editor_score;
    if (score < 10) return `Edit ${10 - stats.total_edits} more articles to achieve Bronze`;
    if (score < 100) return `Edit ${100 - stats.total_edits} more articles to achieve Silver`;
    if (score < 500) return `Edit ${500 - stats.total_edits} more articles to achieve Gold`;
    if (score < 1000) return `Edit ${1000 - stats.total_edits} more articles to achieve Platinum`;
    return null;
  };

  if (loading) {
    return (
      <div className="animate-pulse">
        <div className="h-8 bg-gray-200 rounded w-1/4 mb-4"></div>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="h-24 bg-gray-200 rounded"></div>
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <p className="text-red-800">{error}</p>
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
        <p className="text-yellow-800">No editor stats available yet. Start editing articles to see your dashboard!</p>
      </div>
    );
  }

  const level = getEditorLevel(stats.editor_score);
  const currentRanking = getCurrentUserRanking();
  const badges = getBadges(stats);

  return (
    <div className="space-y-6">
      {/* Editor Level and Ranking */}
      <div className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-lg p-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Editor Dashboard</h2>
            <div className="flex items-center mt-2 space-x-2">
              <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${level.bgColor} ${level.color}`}>
                {level.level} Editor
              </span>
              {/* Gamification progress message */}
              {stats && getNextLevelProgress(stats) && (
                <span className="ml-2 text-xs text-gray-500">{getNextLevelProgress(stats)}</span>
              )}
            </div>
            {/* Personalized top-N stat messages */}
            <div className="flex flex-wrap gap-2 mt-2">
              {stats && getStatRanking('total_edits') && (
                <span className="text-xs text-indigo-600 bg-indigo-50 rounded px-2 py-1">{getStatRanking('total_edits')}</span>
              )}
              {stats && getStatRanking('total_views') && (
                <span className="text-xs text-purple-600 bg-purple-50 rounded px-2 py-1">{getStatRanking('total_views')}</span>
              )}
              {stats && getStatRanking('total_saves') && (
                <span className="text-xs text-yellow-600 bg-yellow-50 rounded px-2 py-1">{getStatRanking('total_saves')}</span>
              )}
              {stats && getStatRanking('editor_score') && (
                <span className="text-xs text-gray-600 bg-gray-50 rounded px-2 py-1">{getStatRanking('editor_score')}</span>
              )}
            </div>
          </div>
          <div className="text-right">
            <div className="text-3xl font-bold text-gray-900">{Math.round(stats.editor_score)}</div>
            <div className="text-sm text-gray-600">Total Score</div>
          </div>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <div className="p-2 bg-blue-100 rounded-lg">
              <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
              </svg>
            </div>
            <div className="ml-4">
              <div className="flex items-center">
                <div className="text-2xl font-bold text-gray-900">{stats.total_edits}</div>
                <Tooltip text="Total number of edits you have made to any article. Multiple edits to the same article are all counted." />
              </div>
              <div className="text-sm text-gray-600">Total Edits</div>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <div className="p-2 bg-green-100 rounded-lg">
              <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <div className="ml-4">
              <div className="flex items-center">
                <div className="text-2xl font-bold text-gray-900">{stats.unique_articles_edited}</div>
                <Tooltip text="Number of unique articles you have edited at least once. Editing the same article multiple times only counts as one here." />
              </div>
              <div className="text-sm text-gray-600">Articles Edited</div>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <div className="p-2 bg-purple-100 rounded-lg">
              <svg className="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              </svg>
            </div>
            <div className="ml-4">
              <div className="flex items-center">
                <div className="text-2xl font-bold text-gray-900">{stats.total_views.toLocaleString()}</div>
                <Tooltip text="Total number of times articles you have edited have been viewed by users (including repeat views)." />
              </div>
              <div className="text-sm text-gray-600">Total Views</div>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <div className="p-2 bg-yellow-100 rounded-lg">
              <svg className="w-6 h-6 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
              </svg>
            </div>
            <div className="ml-4">
              <div className="flex items-center">
                <div className="text-2xl font-bold text-gray-900">{stats.total_saves.toLocaleString()}</div>
                <Tooltip text="Total number of times articles you have edited have been saved by users to their profile." />
              </div>
              <div className="text-sm text-gray-600">Total Saves</div>
            </div>
          </div>
        </div>
      </div>

      {/* Badges */}
      {badges.length > 0 && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Achievements</h3>
          <div className="flex flex-wrap gap-2">
            {badges.map((badge, index) => (
              <span
                key={index}
                className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium text-white ${badge.color}`}
              >
                {badge.name}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
} 