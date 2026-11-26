import OpenAI from 'openai';

// Initialize OpenAI client
const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

// Brain region mapping for different experiences
const BRAIN_REGION_MAPPING: { [key: string]: string[] } = {
    'stress': ['amygdala', 'prefrontal_cortex'],
    'anxiety': ['amygdala', 'prefrontal_cortex'],
    'fear': ['amygdala'],
    'happiness': ['prefrontal_cortex', 'temporal_lobe'],
    'sadness': ['amygdala', 'prefrontal_cortex'],
    'anger': ['amygdala', 'prefrontal_cortex'],
    'memory': ['hippocampus', 'temporal_lobe'],
    'learning': ['hippocampus', 'prefrontal_cortex'],
    'movement': ['cerebellum', 'motor_cortex'],
    'emotion': ['amygdala', 'prefrontal_cortex'],
    'decision_making': ['prefrontal_cortex'],
    'creativity': ['prefrontal_cortex', 'temporal_lobe'],
    'focus': ['prefrontal_cortex'],
    'relaxation': ['prefrontal_cortex', 'amygdala'],
    'social_interaction': ['prefrontal_cortex', 'temporal_lobe'],
    'physical_activity': ['cerebellum', 'motor_cortex'],
    'sleep': ['prefrontal_cortex', 'hippocampus'],
    'eating': ['prefrontal_cortex', 'amygdala'],
    'music': ['temporal_lobe', 'prefrontal_cortex'],
    'art': ['prefrontal_cortex', 'temporal_lobe'],
};

const SYSTEM_PROMPT = `You are an AI that analyzes journal entries to identify experiences and their associated brain regions.
For each experience mentioned in the journal entry, identify:
1. The type of experience (e.g., stress, happiness, learning)
2. The intensity (1-5, where 5 is most intense)
3. A direct quote from the text that provides evidence for this experience

Return the analysis as a JSON object with this structure:
{
    "experiences": [
        {
            "type": "string (one of the predefined types)",
            "intensity": number (1-5),
            "evidence": "string (direct quote from the text)"
        }
    ]
}

Only include experiences that are explicitly mentioned in the text.`;

export async function analyzeJournalEntry(content: string) {
    try {
        // Analyze the journal entry with GPT
        const completion = await openai.chat.completions.create({
            model: 'gpt-4o-mini',
            messages: [
                { role: "system", content: SYSTEM_PROMPT },
                { role: "user", content: content }
            ],
            response_format: { type: "json_object" }
        });

        const messageContent = completion.choices[0].message.content;
        if (!messageContent) {
            throw new Error('No content in GPT response');
        }

        const analysis = JSON.parse(messageContent);

        // Process experiences and create brain data
        return analysis.experiences.map((exp: any) => ({
            type: exp.type,
            intensity: exp.intensity,
            evidence: exp.evidence,
            brain_regions: BRAIN_REGION_MAPPING[exp.type] || []
        }));
    } catch (error) {
        console.error('Error analyzing journal entry:', error);
        throw error;
    }
}

export async function analyzeWithAI(
    journalEntry: string,
    systemPrompt: string,
    model: string = 'gpt-4o-mini' // Default to GPT-4o-mini
) {
    try {
        const completion = await openai.chat.completions.create({
            model,
            messages: [
                { role: 'system', content: systemPrompt },
                { role: 'user', content: journalEntry }
            ],
            temperature: 0.7,
            max_tokens: 1000,
        });

        const response = completion.choices[0]?.message?.content;
        if (!response) {
            throw new Error('No response from AI');
        }

        // Parse the JSON response
        const analysis = JSON.parse(response);
        return analysis;
    } catch (error) {
        console.error('Error in AI analysis:', error);
        throw error;
    }
} 