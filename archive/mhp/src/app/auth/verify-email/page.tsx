'use client';

import { useEffect, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { createClient } from '@/lib/supabase';

export default function VerifyEmailPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [verifying, setVerifying] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const supabase = createClient();

  useEffect(() => {
    const verifyEmail = async () => {
      try {
        const token = searchParams.get('token');
        const type = searchParams.get('type');

        if (!token || type !== 'email') {
          throw new Error('Invalid verification link');
        }

        const { error } = await supabase.auth.verifyOtp({
          token_hash: token,
          type: 'email',
        });

        if (error) throw error;

        // Redirect to login page with success message
        router.replace('/auth/login?verified=true');
      } catch (error: any) {
        console.error('Verification error:', error);
        setError(error.message);
      } finally {
        setVerifying(false);
      }
    };

    verifyEmail();
  }, []);

  if (verifying) {
    return (
      <div className="min-h-screen bg-gray-50 flex flex-col justify-center py-12 sm:px-6 lg:px-8">
        <div className="sm:mx-auto sm:w-full sm:max-w-md">
          <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900">
            Verifying your email...
          </h2>
          <div className="mt-8 flex justify-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex flex-col justify-center py-12 sm:px-6 lg:px-8">
        <div className="sm:mx-auto sm:w-full sm:max-w-md">
          <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900">
            Verification Failed
          </h2>
          <div className="mt-8 bg-red-50 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
            <span className="block sm:inline">{error}</span>
          </div>
          <div className="mt-6 text-center">
            <a
              href="/auth/login"
              className="font-medium text-indigo-600 hover:text-indigo-500"
            >
              Return to login
            </a>
          </div>
        </div>
      </div>
    );
  }

  return null;
} 