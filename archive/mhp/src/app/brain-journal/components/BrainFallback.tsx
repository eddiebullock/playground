interface BrainFallbackProps {
    experiences: Array<{
        type: string;
        intensity: number;
        evidence: string;
        brain_regions: string[];
    }>;
}

export default function BrainFallback({ experiences }: BrainFallbackProps) {
    return (
        <div className="bg-white rounded-lg p-6 shadow-lg">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Brain Region Analysis
            </h3>
            <div className="space-y-4">
                {experiences.map((experience, index) => (
                    <div key={index} className="border border-gray-200 rounded-lg p-4">
                        <div className="flex items-center justify-between mb-2">
                            <h4 className="font-medium text-gray-900">
                                {experience.type.split('_').map(word => 
                                    word.charAt(0).toUpperCase() + word.slice(1)
                                ).join(' ')}
                            </h4>
                            <span className="text-sm text-gray-600">
                                Intensity: {(experience.intensity * 100).toFixed(0)}%
                            </span>
                        </div>
                        <p className="text-gray-600 italic mb-2">
                            "{experience.evidence}"
                        </p>
                        <div className="mt-2">
                            <h5 className="text-sm font-medium text-gray-700 mb-1">
                                Activated Brain Regions:
                            </h5>
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
            <div className="mt-6 p-4 bg-gray-50 rounded-lg">
                <p className="text-sm text-gray-600">
                    Note: Your device doesn't support 3D visualization. 
                    This is a simplified view of the brain region analysis.
                </p>
            </div>
        </div>
    );
} 