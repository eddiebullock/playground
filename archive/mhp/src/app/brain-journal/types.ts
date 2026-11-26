// Types for brain journal application

// Experience types
export type ExperienceType = 
    | 'anxiety'
    | 'stress'
    | 'happiness'
    | 'exercise'
    | 'social_interaction'
    | 'learning'
    | 'creativity'
    | 'meditation';

// API Experience type (from backend)
export interface APIExperience {
    type: string;
    intensity: number;
    brain_regions: string[];
    explanation?: string;
}

// Internal Experience type (for frontend)
export interface Experience {
    type: ExperienceType;
    intensity: number;
    evidence: string;
    brain_regions: string[];
}

// Tooltip data type
export interface TooltipData {
    region: string;
    experience: Experience;
    position: {
        x: number;
        y: number;
    };
}

export interface BrainRegion {
    name: string;
    description: string;
    position: [number, number, number];
    scale: [number, number, number];
} 