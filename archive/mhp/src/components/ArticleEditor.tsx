'use client';

import { useState, useEffect } from 'react';
import { createBrowserClient } from '@supabase/ssr';
import { Database } from '@/types/supabase';

type Article = Database['public']['Tables']['articles']['Row'];

type ArticleEditorProps = {
  article: Article;
  onSave: () => void;
  onCancel: () => void;
};

// Helper function to convert field names to display names
const getFieldDisplayName = (fieldName: string): string => {
  return fieldName
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

export default function ArticleEditor({ article, onSave, onCancel }: ArticleEditorProps) {
  const [editedArticle, setEditedArticle] = useState(article);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const supabase = createBrowserClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
  );

  const handleSave = async () => {
    setLoading(true);
    setError(null);

    try {
      const { error } = await supabase
        .from('articles')
        .update({
          title: editedArticle.title,
          summary: editedArticle.summary,
          content_blocks: editedArticle.content_blocks,
          tags: editedArticle.tags,
          updated_at: new Date().toISOString(),
        })
        .eq('id', article.id);

      if (error) throw error;

      // Log the edit for tracking editor contributions
      const { data: { user } } = await supabase.auth.getUser();
      if (user) {
        await supabase
          .from('article_edits')
          .insert({
            article_id: article.id,
            editor_id: user.id,
          });
      }

      onSave();
    } catch (err) {
      console.error('Error saving article:', err);
      setError('Failed to save changes. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleContentChange = (field: string, value: string) => {
    setEditedArticle(prev => ({
      ...prev,
      content_blocks: {
        ...prev.content_blocks,
        [field]: value
      }
    }));
  };

  // Get all content_blocks fields that exist in the article
  const contentFields = editedArticle.content_blocks ? Object.keys(editedArticle.content_blocks) : [];

  return (
    <div className="space-y-6">
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-800">{error}</p>
        </div>
      )}

      <div>
        <label htmlFor="title" className="block text-sm font-medium text-gray-700">
          Title
        </label>
        <input
          type="text"
          id="title"
          value={editedArticle.title}
          onChange={(e) => setEditedArticle(prev => ({ ...prev, title: e.target.value }))}
          className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
        />
      </div>

      <div>
        <label htmlFor="summary" className="block text-sm font-medium text-gray-700">
          Summary
        </label>
        <textarea
          id="summary"
          rows={3}
          value={editedArticle.summary}
          onChange={(e) => setEditedArticle(prev => ({ ...prev, summary: e.target.value }))}
          className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
        />
      </div>

      {/* Dynamically render all content_blocks fields */}
      {contentFields.map((fieldName) => {
        const fieldValue = (editedArticle.content_blocks as any)?.[fieldName] || '';
        const displayName = getFieldDisplayName(fieldName);
        
        return (
          <div key={fieldName}>
            <label htmlFor={fieldName} className="block text-sm font-medium text-gray-700">
              {displayName}
            </label>
            <textarea
              id={fieldName}
              rows={4}
              value={fieldValue}
              onChange={(e) => handleContentChange(fieldName, e.target.value)}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
            />
          </div>
        );
      })}

      <div className="flex justify-end space-x-3">
        <button
          type="button"
          onClick={onCancel}
          className="inline-flex justify-center rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
        >
          Cancel
        </button>
        <button
          type="button"
          onClick={handleSave}
          disabled={loading}
          className="inline-flex justify-center rounded-md border border-transparent bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:opacity-50"
        >
          {loading ? 'Saving...' : 'Save Changes'}
        </button>
      </div>
    </div>
  );
} 