import { Metadata } from 'next';
import { createClient } from '@/lib/supabase/server';
import { notFound } from 'next/navigation';
import { Database } from '@/types/supabase';
import ReactMarkdown from 'react-markdown';
import ArticleClient from './ArticleClient';

type Article = Database['public']['Tables']['articles']['Row'];

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
  const supabase = await createClient();
  const { slug } = await params;
  
  const { data: article } = await supabase
    .from('articles')
    .select('*')
    .eq('slug', slug)
    .single();

  if (!article) {
    return {
      title: 'Article Not Found',
      description: 'The requested article could not be found.',
    };
  }

  return {
    title: `${article.title} | Mental Health Platform`,
    description: article.summary,
    openGraph: {
      title: article.title,
      description: article.summary,
      type: 'article',
      publishedTime: article.created_at,
      tags: article.tags || [],
    },
    twitter: {
      card: 'summary_large_image',
      title: article.title,
      description: article.summary,
    },
  };
}

export default async function ArticlePage({ 
  params, 
  searchParams 
}: { 
  params: Promise<{ slug: string }>;
  searchParams: Promise<{ [key: string]: string | string[] | undefined }>;
}) {
  const supabase = await createClient();
  const { slug } = await params;
  const resolvedSearchParams = await searchParams;
  
  const { data: article, error } = await supabase
    .from('articles')
    .select('*')
    .eq('slug', slug)
    .single();

  if (error || !article) {
    notFound();
  }

  // Check if user is an editor
  const { data: { user }, error: userError } = await supabase.auth.getUser();
  let isEditor = false;

  console.log('Article page - User:', user?.id);
  console.log('Article page - User email:', user?.email);
  console.log('Article page - User error:', userError);
  console.log('Article page - User session exists:', !!user);

  if (user) {
    console.log('Article page - Checking editor for user ID:', user.id);
    console.log('Article page - Checking editor for user email:', user.email);
    
    // Simple role-based system: Check if user email is in authorized editors list
    // This bypasses the RLS recursion issue and gives you full control
    const authorizedEditors = [
      'eddie@mentalhealthprogram.co.uk',
      'daughterofthes3a@gmail.com'
      // Add more authorized editor emails here as needed
    ];
    
    isEditor = authorizedEditors.includes(user.email || '');
    
    console.log('Article page - Authorized editors:', authorizedEditors);
    console.log('Article page - User email in authorized list:', user.email);
    console.log('Article page - Is user authorized editor:', isEditor);
  }

  console.log('Article page - Final isEditor result:', isEditor);

  // Check for edit mode from search params
  const editMode = resolvedSearchParams.edit === 'true';
  // Only allow editing for users in the article_editors table
  const finalIsEditor = isEditor;

  console.log('Article page - Edit mode from URL:', editMode);
  console.log('Article page - Final editor status:', finalIsEditor);

  return <ArticleClient article={article} isEditor={finalIsEditor} />;
} 