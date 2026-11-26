'use client';

import { useState, useEffect } from 'react';
import { Database } from '@/types/supabase';
import ReactMarkdown from 'react-markdown';
import ArticleEditor from '@/components/ArticleEditor';
import { useSearchParams } from 'next/navigation';
import { LockClosedIcon, LockOpenIcon } from '@heroicons/react/24/outline';
import SaveArticleButton from '@/components/SaveArticleButton';
import { createClient } from '@/lib/supabase';

type Article = Database['public']['Tables']['articles']['Row'];

type ArticleClientProps = {
  article: Article;
  isEditor: boolean;
};

export default function ArticleClient({ article, isEditor }: ArticleClientProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [currentArticle, setCurrentArticle] = useState(article);
  const [showLockMessage, setShowLockMessage] = useState(false);
  const searchParams = useSearchParams();
  const [userId, setUserId] = useState<string | null>(null);
  const [initialSaved, setInitialSaved] = useState<boolean>(false);

  useEffect(() => {
    const fetchUserAndSaved = async () => {
      const supabase = createClient();
      const { data: { user } } = await supabase.auth.getUser();
      setUserId(user?.id ?? null);
      if (user) {
        const { data: saved } = await supabase
          .from('saved_articles')
          .select('id')
          .eq('user_id', user.id)
          .eq('article_id', article.id)
          .single();
        setInitialSaved(!!saved);
      }

      // Log the view for tracking article views
      try {
        // Check if user has already viewed this article today
        if (user?.id) {
          const today = new Date().toISOString().split('T')[0]; // Get today's date as YYYY-MM-DD
          const { data: existingView } = await supabase
            .from('article_views')
            .select('id')
            .eq('article_id', article.id)
            .eq('viewer_id', user.id)
            .gte('viewed_at', `${today}T00:00:00Z`)
            .lt('viewed_at', `${today}T23:59:59Z`)
            .single();

          if (!existingView) {
            await supabase
              .from('article_views')
              .insert({
                article_id: article.id,
                viewer_id: user.id,
              });
          }
        } else {
          // For anonymous users, always log the view
          await supabase
            .from('article_views')
            .insert({
              article_id: article.id,
              viewer_id: null,
            });
        }
      } catch (error) {
        // Ignore errors for view logging
        console.log('View logging error:', error);
      }
    };
    fetchUserAndSaved();
  }, [article.id]);

  const handleSave = () => {
    setIsEditing(false);
    // Refresh the page to show updated content
    window.location.reload();
  };

  const handleCancel = () => {
    setIsEditing(false);
  };

  const handleEditClick = () => {
    if (!isEditor) {
      setShowLockMessage(true);
      setTimeout(() => setShowLockMessage(false), 5000); // Hide after 5 seconds
      return;
    }
    setIsEditing(true);
  };

  // Show edit button for editors only
  const showEditButton = isEditor;
  
  // Debug logging
  console.log('ArticleClient - isEditor prop:', isEditor);
  console.log('ArticleClient - showEditButton:', showEditButton);
  console.log('ArticleClient - URL edit param:', searchParams.get('edit'));

  const renderContent = () => {
    if (!currentArticle.content_blocks) {
      return <p className="text-gray-500">No content available.</p>;
    }

    const contentBlocks = currentArticle.content_blocks as Record<string, any>;
    const contentFields = Object.keys(contentBlocks).filter(key => {
      const value = contentBlocks[key];
      return typeof value === 'string' && value.trim().length > 0;
    });

    if (contentFields.length === 0) {
      return <p className="text-gray-500">No content available.</p>;
    }

    // Helper function to convert field names to display names
    const getFieldDisplayName = (fieldName: string): string => {
      return fieldName
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
    };

    return (
      <div className="prose prose-gray max-w-none [&_*]:text-black [&_h2]:text-gray-900 [&_h3]:text-gray-900 [&_h4]:text-gray-900">
        <div className="prose max-w-none">
          {contentFields.map((fieldName) => {
            const content = contentBlocks[fieldName] as string;
            const displayName = getFieldDisplayName(fieldName);
            
            return (
              <section key={fieldName} className="mb-8">
                <h2 className="text-2xl font-bold mb-4">{displayName}</h2>
                <ReactMarkdown>{content}</ReactMarkdown>
              </section>
            );
          })}
        </div>
      </div>
    );
  };

  if (isEditing) {
    return (
      <article className="min-h-screen bg-gray-50">
        <main className="max-w-4xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
          <div className="bg-white shadow rounded-lg overflow-hidden">
            <div className="px-6 py-8">
              <h1 className="text-3xl font-bold text-gray-900 mb-6">Edit Article</h1>
              <ArticleEditor 
                article={currentArticle as any} 
                onSave={handleSave} 
                onCancel={handleCancel} 
              />
            </div>
          </div>
        </main>
      </article>
    );
  }

  return (
    <article className="min-h-screen bg-gray-50">
      <main className="max-w-4xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
        <div className="bg-white shadow rounded-lg overflow-hidden">
          <div className="px-6 py-8">
            <div className="flex justify-between items-center mb-4">
              <h1 className="text-3xl font-bold text-gray-900">{currentArticle.title}</h1>
              <SaveArticleButton
                articleId={article.id}
                userId={userId}
                initialSaved={initialSaved}
              />
            </div>
            <div className="flex justify-between items-start mb-4">
              <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-indigo-100 text-indigo-800">
                {article.category}
              </span>
              {showEditButton ? (
                <button
                  onClick={handleEditClick}
                  className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                >
                  <LockOpenIcon className="h-4 w-4 mr-2" />
                  Edit Article
                </button>
              ) : (
                <button
                  onClick={handleEditClick}
                  className="inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-500 bg-gray-100 hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500 cursor-not-allowed"
                  title="Editor access required"
                >
                  <LockClosedIcon className="h-4 w-4 mr-2" />
                  Edit Article
                </button>
              )}
            </div>
            
            {/* Lock message */}
            {showLockMessage && (
              <div className="mb-4 p-4 bg-yellow-50 border border-yellow-200 rounded-md">
                <div className="flex">
                  <LockClosedIcon className="h-5 w-5 text-yellow-400 mr-2 mt-0.5" />
                  <div>
                    <h3 className="text-sm font-medium text-yellow-800">
                      Editor Access Required
                    </h3>
                    <p className="text-sm text-yellow-700 mt-1">
                      You need to be a verified editor to edit articles. Please contact the platform administrators to apply for editor access.
                    </p>
                  </div>
                </div>
              </div>
            )}

            <p className="text-lg text-gray-900 mb-6">{currentArticle.summary}</p>
            
            <div className="flex flex-wrap gap-2 mb-8">
              {(currentArticle.tags || []).map((tag: string) => (
                <span
                  key={tag}
                  className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-indigo-100 text-indigo-800"
                >
                  {tag}
                </span>
              ))}
            </div>

            <div className="prose prose-gray max-w-none [&_*]:text-black [&_h2]:text-gray-900 [&_h3]:text-gray-900 [&_h4]:text-gray-900">
              {renderContent()}
            </div>

            <div className="mt-8 pt-6 border-t border-gray-200">
              <p className="text-sm text-gray-900">
                Published: {new Date(currentArticle.created_at).toLocaleDateString()}
              </p>
            </div>
          </div>
        </div>
      </main>
    </article>
  );
} 