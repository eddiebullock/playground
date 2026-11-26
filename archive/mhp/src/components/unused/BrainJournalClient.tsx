'use client';

import { useState, useEffect, useCallback } from 'react';
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs';
import { useRouter } from 'next/navigation';
import JournalEntryForm from './components/JournalEntryForm';
import BrainVisualization from './components/BrainVisualization';
import type { APIExperience } from './types';
import type { User } from '@supabase/supabase-js';

interface AnalysisData {
    experiences: APIExperience[];
}

interface BrainJournalClientProps {
    initialUser: User;
}

export default function BrainJournalClient({ initialUser }: BrainJournalClientProps) {
    const [user, setUser] = useState<User>(initialUser);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [analysisData, setAnalysisData] = useState<AnalysisData | null>(null);
    const [error, setError] = useState<string | null>(null);
    const router = useRouter();
    const supabase = createClientComponentClient();

    useEffect(() => {
        const {
            data: { subscription },
        } = supabase.auth.onAuthStateChange((event, session) => {
            console.log('BrainJournalClient: Auth state changed:', { event, session });
            if (event === 'SIGNED_OUT') {
                router.push('/login');
            } else if (session?.user) {
                console.log('BrainJournalClient: User updated:', session.user);
                setUser(session.user);
            }
        });

        return () => {
            subscription.unsubscribe();
        };
    }, [router, supabase.auth]);

    const handleJournalSubmit = useCallback(async (content: string) => {
        if (!content?.trim()) {
            throw new Error('Journal entry is required');
        }

        setIsAnalyzing(true);
        setError(null);

        try {
            console.log('Submitting journal entry:', { content: content.trim() });
            
            const response = await fetch('/api/analyze-journal', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    journalEntry: content.trim()
                }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to analyze journal entry');
            }

            const data = await response.json();
            console.log('Analysis result:', data);
            setAnalysisData(data);
        } catch (err) {
            console.error('Error analyzing journal:', err);
            setError(err instanceof Error ? err.message : 'Failed to analyze journal entry');
            throw err;
        } finally {
            setIsAnalyzing(false);
        }
    }, []);

    return (
        <div className="space-y-8">
            <div className="bg-white rounded-lg shadow-lg p-6">
                <h2 className="text-2xl font-bold mb-4">Brain Journal</h2>
                <JournalEntryForm
                    onSubmit={handleJournalSubmit}
                    isSubmitting={isAnalyzing}
                />
                {error && (
                    <div className="mt-4 p-4 bg-red-50 text-red-700 rounded-md">
                        {error}
                    </div>
                )}
            </div>

            {analysisData && (
                <div className="bg-white rounded-lg shadow-lg p-6">
                    <h2 className="text-2xl font-bold mb-4">Analysis Results</h2>
                    <div className="space-y-4">
                        {analysisData.experiences.map((experience, index) => (
                            <div key={index} className="p-4 bg-gray-50 rounded-lg">
                                <h3 className="font-semibold capitalize">{experience.type}</h3>
                                <p className="text-sm text-gray-600">
                                    Intensity: {experience.intensity}/5
                                </p>
                                <div className="mt-2">
                                    <h4 className="text-sm font-medium">Brain Regions:</h4>
                                    <div className="flex flex-wrap gap-2 mt-1">
                                        {experience.brain_regions.map((region) => (
                                            <span
                                                key={region}
                                                className="px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-sm"
                                            >
                                                {region.replace('_', ' ')}
                                            </span>
                                        ))}
                                    </div>
                                </div>
                                {experience.explanation && (
                                    <p className="mt-2 text-sm text-gray-600">
                                        {experience.explanation}
                                    </p>
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {analysisData && (
                <div className="bg-white rounded-lg shadow-lg p-6">
                    <h2 className="text-2xl font-bold mb-4">Brain Visualization</h2>
                    <BrainVisualization experiences={analysisData.experiences} />
                </div>
            )}
        </div>
    );
} 