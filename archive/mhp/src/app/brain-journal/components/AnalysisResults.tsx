'use client';

interface Experience {
    type: string;
    intensity: number;
    evidence: string;
    brain_regions: string[];
}

interface JournalEntry {
    id: string;
    content: string;
    created_at: string;
}

interface AnalysisResultsProps {
    experiences: Experience[];
    journalEntry: JournalEntry;
}

export default function AnalysisResults({ experiences, journalEntry }: AnalysisResultsProps) {
    // Format experience type for display
    const formatExperienceType = (type: string | undefined) => {
        if (!type || typeof type !== 'string') return '';
        return type.split('_').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    };

    // Get color based on intensity
    const getIntensityColor = (intensity: number) => {
        if (intensity >= 0.7) return 'bg-red-100 text-red-800';
        if (intensity >= 0.4) return 'bg-yellow-100 text-yellow-800';
        return 'bg-green-100 text-green-800';
    };

    return (
        <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-semibold text-gray-900 mb-4">
                Analysis Results
            </h2>
            <div className="space-y-6">
                {experiences.map((experience, index) => (
                    <div 
                        key={index}
                        className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors"
                    >
                        <div className="flex items-center justify-between mb-2">
                            <h3 className="text-lg font-medium text-gray-900">
                                {formatExperienceType(experience.type)}
                            </h3>
                            <span className={`
                                px-2 py-1 rounded-full text-sm font-medium
                                ${getIntensityColor(experience.intensity)}
                            `}>
                                Intensity: {(experience.intensity * 100).toFixed(0)}%
                            </span>
                        </div>
                        <p className="text-gray-600 italic mb-2">
                            "{experience.evidence}"
                        </p>
                        <div className="mt-2">
                            <h4 className="text-sm font-medium text-gray-700 mb-1">
                                Related Brain Regions:
                            </h4>
                            <div className="flex flex-wrap gap-2">
                                {experience.brain_regions.map((region, idx) => (
                                    <span
                                        key={idx}
                                        className="px-2 py-1 bg-indigo-100 text-indigo-800 rounded-full text-sm"
                                    >
                                        {region.split('_').map(word => 
                                            word.charAt(0).toUpperCase() + word.slice(1)
                                        ).join(' ')}
                                    </span>
                                ))}
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
} 