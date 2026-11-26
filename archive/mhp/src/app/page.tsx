'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { createClient } from '@/lib/supabase';

export default function HomePage() {
  const router = useRouter();
  const [searchQuery, setSearchQuery] = useState('');
  const [placeholder, setPlaceholder] = useState('');
  const [user, setUser] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  const supabase = createClient();

  const searchTopics = [
    'ADHD',
    'anxiety',
    'neurons',
    'therapy',
    'depression',
    'meditation',
    'stress',
    'sleep',
    'mindfulness',
    'psychology'
  ];

  useEffect(() => {
    const checkUser = async () => {
      try {
        const { data: { session } } = await supabase.auth.getSession();
        setUser(session?.user ?? null);
        
        // If user is logged in, redirect to feed
        if (session?.user) {
          router.replace('/feed');
          return;
        }
      } catch (error) {
        console.error('Error checking user:', error);
      } finally {
        setLoading(false);
      }
    };

    checkUser();

    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ?? null);
      if (session?.user) {
        router.replace('/feed');
      }
    });

    return () => subscription.unsubscribe();
  }, [router, supabase.auth]);

  useEffect(() => {
    let currentIndex = 0;
    let currentText = '';
    let isDeleting = false;
    let typingSpeed = 100;

    const animatePlaceholder = () => {
      const currentTopic = searchTopics[currentIndex];
      
      if (isDeleting) {
        currentText = currentTopic.substring(0, currentText.length - 1);
      } else {
        currentText = currentTopic.substring(0, currentText.length + 1);
      }

      setPlaceholder(currentText);

      if (!isDeleting && currentText === currentTopic) {
        typingSpeed = 2000; // Pause at the end
        isDeleting = true;
      } else if (isDeleting && currentText === '') {
        isDeleting = false;
        currentIndex = (currentIndex + 1) % searchTopics.length;
        typingSpeed = 100;
      } else {
        typingSpeed = isDeleting ? 50 : 100;
      }

      setTimeout(animatePlaceholder, typingSpeed);
    };

    animatePlaceholder();
  }, []);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      router.push(`/feed?search=${encodeURIComponent(searchQuery)}`);
    }
  };

  // Show loading or redirect if user is logged in
  if (loading || user) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <main className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          <h1 className="text-4xl font-extrabold text-gray-900 sm:text-5xl sm:tracking-tight lg:text-6xl">
            Mental Health Platform
          </h1>
          <p className="mt-6 max-w-2xl mx-auto text-xl text-gray-500">
            Evidence-based mental health information and resources
          </p>
          
          {/* Search Bar */}
          <div className="mt-8 max-w-2xl mx-auto">
            <form onSubmit={handleSearch} className="relative">
              <input
                type="text"
                className="w-full px-6 py-4 text-xl rounded-lg border-2 border-gray-300 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500 shadow-lg text-gray-900 placeholder-gray-500"
                placeholder={`Search for ${placeholder}...`}
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
              <button
                type="submit"
                className="absolute right-2 top-1/2 transform -translate-y-1/2 px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
              >
                Search
              </button>
            </form>
          </div>

          <div className="mt-10">
            <Link
              href="/feed"
              className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700"
            >
              Browse Articles
            </Link>
          </div>

          <div className="my-4">
            <Link href="/mental-health-page" className="text-blue-600 underline font-semibold">
              Try the Mental Health Research Chatbot
            </Link>
          </div>
        </div>

        <div className="mt-20">
          <div className="grid grid-cols-1 gap-8 sm:grid-cols-2 lg:grid-cols-3">
            <div className="pt-6">
              <div className="flow-root bg-white rounded-lg px-6 pb-8">
                <div className="-mt-6">
                  <div className="inline-flex items-center justify-center p-3 bg-indigo-500 rounded-md shadow-lg">
                    <svg className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </div>
                  <h3 className="mt-8 text-lg font-medium text-gray-900 tracking-tight">Evidence-Based Content</h3>
                  <p className="mt-5 text-base text-gray-500">
                    Access reliable, research-backed information about mental health topics.
                  </p>
                </div>
              </div>
            </div>

            <div className="pt-6">
              <div className="flow-root bg-white rounded-lg px-6 pb-8">
                <div className="-mt-6">
                  <div className="inline-flex items-center justify-center p-3 bg-indigo-500 rounded-md shadow-lg">
                    <svg className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                    </svg>
                  </div>
                  <h3 className="mt-8 text-lg font-medium text-gray-900 tracking-tight">Personalized Experience</h3>
                  <p className="mt-5 text-base text-gray-500">
                    Save articles and track your reading progress with a personalized profile.
                  </p>
                </div>
              </div>
            </div>

            <div className="pt-6">
              <div className="flow-root bg-white rounded-lg px-6 pb-8">
                <div className="-mt-6">
                  <div className="inline-flex items-center justify-center p-3 bg-indigo-500 rounded-md shadow-lg">
                    <svg className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                  </div>
                  <h3 className="mt-8 text-lg font-medium text-gray-900 tracking-tight">Easy Navigation</h3>
                  <p className="mt-5 text-base text-gray-500">
                    Find the information you need quickly with our intuitive search and filtering system.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
