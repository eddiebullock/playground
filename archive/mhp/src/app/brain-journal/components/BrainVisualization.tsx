import { useEffect, useState } from 'react';
import type { APIExperience } from '../types';

interface BrainVisualizationProps {
    experiences: Array<{
        type: string;
        intensity: number;
        brain_regions: string[];
        explanation?: string;
    }>;
}

interface VisualizationData {
    visualization: string;
    message: string;
}

export default function BrainVisualization({ experiences }: BrainVisualizationProps) {
    const [visualizationData, setVisualizationData] = useState<VisualizationData | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);

    useEffect(() => {
        async function generateVisualization() {
            if (!experiences.length) return;

            setIsLoading(true);
            setError(null);

            try {
                const response = await fetch('/api/visualize-brain', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ experiences }),
                });

                if (!response.ok) {
                    throw new Error('Failed to generate visualization');
                }

                const data = await response.json();
                setVisualizationData(data);
            } catch (err) {
                console.error('Error generating visualization:', err);
                setError('Failed to generate brain visualization');
            } finally {
                setIsLoading(false);
            }
        }

        generateVisualization();
    }, [experiences]);

    if (isLoading) {
        return (
            <div className="w-full h-full min-h-[500px] bg-white rounded-lg shadow-lg flex items-center justify-center">
                <div className="text-gray-500">Generating brain visualization...</div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="w-full h-full min-h-[500px] bg-white rounded-lg shadow-lg flex items-center justify-center">
                <div className="text-red-500">{error}</div>
            </div>
        );
    }

    if (!visualizationData) {
        return null;
    }

    // Extract unique brain regions from experiences
    const uniqueRegions = Array.from(new Set(
        experiences.flatMap(exp => exp.brain_regions)
    ));

    return (
        <div className="space-y-6">
            {/* Brain Visualization Image */}
            <div className="w-full bg-white rounded-lg shadow-lg overflow-hidden">
                <img 
                    src={visualizationData.visualization} 
                    alt="Brain Activity Visualization"
                    className="w-full h-auto"
                />
            </div>

            {/* Brain Regions Summary */}
            <div className="bg-white rounded-lg shadow-lg p-6">
                <h3 className="text-xl font-semibold mb-4">Brain Activity Summary</h3>
                <div className="space-y-4">
                    <div>
                        <h4 className="font-medium mb-2">Active Brain Regions:</h4>
                        <div className="flex flex-wrap gap-2">
                            {uniqueRegions.map((region) => (
                                <span
                                    key={region}
                                    className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm"
                                >
                                    {region.replace('_', ' ')}
                                </span>
                            ))}
                        </div>
                    </div>
                    <div>
                        <h4 className="font-medium mb-2">Experiences Detected:</h4>
                        <div className="space-y-2">
                            {experiences.map((exp, index) => (
                                <div key={index} className="p-3 bg-gray-50 rounded-lg">
                                    <div className="flex justify-between items-start">
                                        <span className="font-medium capitalize">
                                            {exp.type.replace('_', ' ')}
                                        </span>
                                        <span className="text-sm text-gray-600">
                                            Intensity: {exp.intensity}/5
                                        </span>
                                    </div>
                                    {exp.explanation && (
                                        <p className="mt-1 text-sm text-gray-600">
                                            {exp.explanation}
                                        </p>
                                    )}
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
} 