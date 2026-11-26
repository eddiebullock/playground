'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { createClient } from '@/lib/supabase';
import EditorDashboard from '@/components/EditorDashboard';

interface SavedArticle {
  id: string;
  title: string;
  slug: string;
  summary: string;
  created_at: string;
}

export default function ProfilePage() {
  const router = useRouter();
  const [profile, setProfile] = useState<any>(null);
  const [savedArticles, setSavedArticles] = useState<SavedArticle[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isEditor, setIsEditor] = useState(false);

  const supabase = createClient();

  useEffect(() => {
    const checkSession = async () => {
      try {
        const { data: { session }, error: sessionError } = await supabase.auth.getSession();
        if (sessionError) throw sessionError;
        
        if (!session) {
          router.replace('/auth/login');
          return;
        }

        // Check if user is an editor
        const authorizedEditors = [
          'eddie@mentalhealthprogram.co.uk',
          'daughterofthes3a@gmail.com'
          // Add more authorized editor emails here as needed
        ];
        setIsEditor(authorizedEditors.includes(session.user.email || ''));

        // Fetch profile
        const { data: profileData, error: profileError } = await supabase
          .from('profiles')
          .select('*')
          .eq('id', session.user.id)
          .single();

        if (profileError) {
          if (profileError.code === 'PGRST116') {
            // Profile doesn't exist, create one
            const { data: newProfile, error: createError } = await supabase
              .from('profiles')
              .insert([
                {
                  id: session.user.id,
                  email: session.user.email,
                  username: session.user.email?.split('@')[0],
                },
              ])
              .select()
              .single();

            if (createError) throw createError;
            setProfile(newProfile);
          } else {
            throw profileError;
          }
        } else {
          setProfile(profileData);
        }

        // Fetch saved articles
        const { data: savedData, error: savedError } = await supabase
          .from('saved_articles')
          .select(`
            article:articles (
              id,
              title,
              slug,
              summary,
              created_at
            )
          `)
          .eq('user_id', session.user.id);

        if (savedError) throw savedError;
        setSavedArticles(savedData.map((sa: any) => sa.article));
      } catch (error: any) {
        console.error('Error:', error);
        setError(error.message);
      } finally {
        setLoading(false);
      }
    };

    checkSession();
  }, []);

  const handleSignOut = async () => {
    try {
      await supabase.auth.signOut();
      router.replace('/');
    } catch (error) {
      console.error('Error signing out:', error);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">Error</h2>
          <p className="text-gray-600">{error}</p>
          <button
            onClick={() => router.push('/')}
            className="mt-4 inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700"
          >
            Return Home
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <div className="px-4 py-6 sm:px-0">
          <div className="bg-white shadow rounded-lg p-6">
            <div className="flex justify-between items-center mb-6">
              <h1 className="text-2xl font-bold text-gray-900">
                Welcome back, {profile?.username || profile?.email || 'User'}
              </h1>
              <button
                onClick={handleSignOut}
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-red-600 hover:bg-red-700"
              >
                Sign out
              </button>
            </div>

            {/* Editor Dashboard */}
            {isEditor && profile && (
              <div className="mb-8">
                <EditorDashboard userId={profile.id} />
              </div>
            )}

            <div className="mt-8">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">Saved Articles</h2>
              {savedArticles.length > 0 ? (
                <div className="space-y-4">
                  {savedArticles.map((article) => (
                    <Link
                      key={article.id}
                      href={`/articles/${article.slug}`}
                      className="block p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
                    >
                      <h3 className="text-lg font-medium text-gray-900">
                        {article.title}
                      </h3>
                      <p className="mt-1 text-gray-600">{article.summary}</p>
                      <p className="text-sm text-gray-500 mt-2">
                        Saved on {new Date(article.created_at).toLocaleDateString()}
                      </p>
                    </Link>
                  ))}
                </div>
              ) : (
                <p className="text-gray-500">No saved articles yet.</p>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
} 