'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { createClient } from '@/lib/supabase';
import { MagnifyingGlassIcon, UserIcon } from '@heroicons/react/24/outline';
import './globals.css';

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const router = useRouter();
  const [user, setUser] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  const supabase = createClient();

  useEffect(() => {
    const checkUser = async () => {
      try {
        const { data: { session } } = await supabase.auth.getSession();
        setUser(session?.user ?? null);
      } catch (error) {
        console.error('Error checking user:', error);
      } finally {
        setLoading(false);
      }
    };

    checkUser();

    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ?? null);
    });

    return () => subscription.unsubscribe();
  }, []);

  const handleSignOut = async () => {
    try {
      await supabase.auth.signOut();
      router.replace('/');
    } catch (error) {
      console.error('Error signing out:', error);
    }
  };

  return (
    <html lang="en">
      <body 
        className="min-h-screen bg-gray-50"
        suppressHydrationWarning={true}
      >
        <nav className="bg-white shadow">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between h-16">
              <div className="flex">
                <Link href="/" className="flex items-center">
                  <span className="text-xl font-bold text-indigo-600">MHP</span>
                </Link>
                <div className="hidden sm:ml-6 sm:flex sm:space-x-8">
                  <Link
                    href="/feed"
                    className="border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700 inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium"
                  >
                    <MagnifyingGlassIcon className="h-4 w-4 mr-1" />
                    Feed
                  </Link>
                  <Link
                    href="/evidence"
                    className="border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700 inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium"
                  >
                    Evidence
                  </Link>
                  <Link
                    href="/topics"
                    className="border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700 inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium"
                  >
                    Topics
                  </Link>
                  <Link
                    href="/mental-health-page"
                    className="border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700 inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium"
                  >
                    Mental Health Chatbot
                  </Link>
                  {/* Academic Research Assistant link for editors only */}
                  {user && [
                    'eddie@mentalhealthprogram.co.uk',
                    'daughterofthes3a@gmail.com'
                    // Add more authorized editor emails here as needed
                  ].includes(user.email) && (
                    <Link
                      href="/academic-research-assistant"
                      className="border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700 inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium"
                    >
                      Academic Research Assistant
                    </Link>
                  )}
                </div>
              </div>
              <div className="flex items-center">
                {!loading && (
                  <>
                    {user ? (
                      <div className="flex items-center space-x-4">
                        <span className="text-sm text-gray-500">
                          Signed in as {user.email}
                        </span>
                        <Link
                          href="/profile"
                          className="inline-flex items-center px-3 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                        >
                          <UserIcon className="h-4 w-4 mr-1" />
                          Profile
                        </Link>
                        <button
                          onClick={handleSignOut}
                          className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
                        >
                          Sign out
                        </button>
                      </div>
                    ) : (
                      <div className="flex items-center space-x-4">
                        <Link
                          href="/auth/login"
                          className="text-gray-500 hover:text-gray-700 px-3 py-2 rounded-md text-sm font-medium"
                        >
                          Sign in
                        </Link>
                        <Link
                          href="/auth/signup"
                          className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                        >
                          Sign up
                        </Link>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          </div>
        </nav>
        {children}
      </body>
    </html>
  );
}
