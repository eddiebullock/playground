import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { createServerClient } from '@supabase/ssr';
import { analyzeWithAI } from '@/lib/ai';

// Brain region mapping for different experiences
const BRAIN_REGION_MAPPING: Record<string, string[]> = {
    anxiety: ['amygdala', 'prefrontal_cortex', 'hippocampus'],
    stress: ['amygdala', 'hypothalamus', 'prefrontal_cortex'],
    happiness: ['nucleus_accumbens', 'ventral_tegmental_area', 'prefrontal_cortex'],
    exercise: ['motor_cortex', 'cerebellum', 'basal_ganglia'],
    social_interaction: ['temporal_lobe', 'prefrontal_cortex', 'mirror_neurons'],
    learning: ['hippocampus', 'prefrontal_cortex', 'cerebellum'],
    creativity: ['prefrontal_cortex', 'temporal_lobe', 'parietal_lobe'],
    meditation: ['prefrontal_cortex', 'insula', 'anterior_cingulate_cortex']
};

// GPT system prompt for analysis
const SYSTEM_PROMPT = `You are a specialized AI that analyzes journal entries and identifies brain regions involved in the described experiences. 
Your task is to analyze the journal entry and output a JSON array of experiences, where each experience has:
- type: string (one of: happiness, sadness, anxiety, stress, anger, exercise, social_interaction, learning, creativity, meditation, sleep_quality, memory, focus, relaxation, emotion, decision_making, physical_activity, eating, music, art)
- intensity: number (1-5, where 5 is highest)
- brain_regions: string[] (one or more of: prefrontal_cortex, temporal_lobe, parietal_lobe, occipital_lobe, amygdala, hippocampus, cerebellum, brainstem, thalamus)

For each experience in the journal entry:
1. Identify the type of experience
2. Assess its intensity (1-5)
3. List the brain regions involved
4. Provide a brief explanation for each region's involvement

Output format example:
{
  "experiences": [
    {
      "type": "happiness",
      "intensity": 4,
      "brain_regions": ["prefrontal_cortex", "amygdala"],
      "explanation": "The prefrontal cortex is involved in positive emotion regulation, while the amygdala processes emotional responses."
    }
  ]
}

Ensure your analysis is scientifically accurate and based on current neuroscience research.`;

export async function POST(request: Request) {
    try {
        const cookieStore = await cookies();
        const supabase = createServerClient(
            process.env.NEXT_PUBLIC_SUPABASE_URL!,
            process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
            {
                cookies: {
                    async get(name: string) {
                        const cookie = await cookieStore.get(name);
                        return cookie?.value;
                    },
                    set(name: string, value: string, options: any) {
                        // Server components can't set cookies directly
                        // The middleware will handle this
                    },
                    remove(name: string, options: any) {
                        // Server components can't remove cookies directly
                        // The middleware will handle this
                    },
                },
            }
        );

        const { data: { user }, error: authError } = await supabase.auth.getUser();
        if (authError || !user) {
            return NextResponse.json(
                { error: 'Unauthorized' },
                { status: 401 }
            );
        }

        const body = await request.json();
        const { journalEntry } = body;

        // Validate journal entry
        if (!journalEntry || typeof journalEntry !== 'string' || journalEntry.trim().length === 0) {
            console.error('Invalid journal entry:', { journalEntry });
            return NextResponse.json(
                { error: 'Journal entry is required and must be a non-empty string' },
                { status: 400 }
            );
        }

        // Analyze with GPT-4o-mini
        const analysis = await analyzeWithAI(journalEntry.trim(), SYSTEM_PROMPT, 'gpt-4o-mini');

        // Start a transaction to save all data
        const { data: entry, error: entryError } = await supabase
            .from('journal_entries')
            .insert({
                user_id: user.id,
                content: journalEntry.trim(),
            })
            .select()
            .single();

        if (entryError) {
            console.error('Error saving journal entry:', entryError);
            return NextResponse.json(
                { error: 'Failed to save journal entry' },
                { status: 500 }
            );
        }

        // Save the analysis
        const { data: analysisData, error: analysisError } = await supabase
            .from('entry_analyses')
            .insert({
                entry_id: entry.id,
                gpt_response: analysis,
                brain_data: analysis, // Store the same data in both fields for now
            })
            .select()
            .single();

        if (analysisError) {
            console.error('Error saving analysis:', analysisError);
            // Try to clean up the journal entry
            await supabase
                .from('journal_entries')
                .delete()
                .eq('id', entry.id);
            return NextResponse.json(
                { error: 'Failed to save analysis' },
                { status: 500 }
            );
        }

        // Save the experiences
        const experiences = analysis.experiences.map((exp: any) => {
            // Convert experience type to match the enum
            let experienceType = exp.type.toLowerCase().replace(/\s+/g, '_');
            // Ensure it's one of the valid enum values
            if (!['anxiety', 'stress', 'happiness', 'exercise', 'social_interaction', 'learning', 'creativity', 'meditation'].includes(experienceType)) {
                experienceType = 'happiness'; // Default to happiness if type doesn't match
            }

            return {
                analysis_id: analysisData.id,
                experience_type: experienceType,
                intensity: Math.min(Math.max(exp.intensity / 5, 0), 1), // Ensure between 0 and 1
                evidence_quote: exp.explanation || '',
                brain_regions: exp.brain_regions || [],
            };
        });

        console.log('Saving experiences:', experiences);

        const { error: experiencesError } = await supabase
            .from('experiences')
            .insert(experiences);

        if (experiencesError) {
            console.error('Error saving experiences:', experiencesError);
            // Try to clean up the journal entry and analysis
            await supabase
                .from('entry_analyses')
                .delete()
                .eq('id', analysisData.id);
            await supabase
                .from('journal_entries')
                .delete()
                .eq('id', entry.id);
            return NextResponse.json(
                { error: 'Failed to save experiences: ' + experiencesError.message },
                { status: 500 }
            );
        }

        return NextResponse.json(analysis);
    } catch (error) {
        console.error('Error in analyze-journal:', error);
        return NextResponse.json(
            { error: 'Failed to analyze journal entry' },
            { status: 500 }
        );
    }
} 