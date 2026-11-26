import { createServerClient } from '@supabase/ssr';
import { cookies } from 'next/headers';

export default async function TestPage() {
  const cookieStore = cookies();
  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    { cookies: { get: (name) => cookieStore.get(name)?.value } }
  );
  const { data: { session } } = await supabase.auth.getSession();
  console.log('DEBUG: Session in test page:', session);
  return <div>Test (Session: { session ? 'Logged in' : 'Not logged in' })</div>;
} 