import AcademicResearchAssistantPage from '@/components/AcademicResearchAssistantPage';
import { createClient } from '@/lib/supabase-server';
import { redirect } from 'next/navigation';

export default async function Page() {
  const supabase = await createClient();
  const { data: { session } } = await supabase.auth.getSession();
  const authorizedEditors = [
    'eddie@mentalhealthprogram.co.uk',
    'daughterofthes3a@gmail.com'
    // Add more authorized editor emails here as needed
  ];
  const userEmail = session?.user?.email || '';
  if (!session || !authorizedEditors.includes(userEmail)) {
    redirect('/');
  }
  return <AcademicResearchAssistantPage />;
} 