import type { APIExperience } from '../types';
import type { ExperienceType } from './experience-colors';

export interface Experience {
    type: string;
    intensity: number;
    evidence: string;
    brain_regions: string[];
}

// Helper function to convert APIExperience to Experience
export function convertToExperience(apiExperience: APIExperience): Experience {
    return {
        type: apiExperience.experience_type,
        intensity: apiExperience.intensity,
        evidence: apiExperience.evidence_quote,
        brain_regions: apiExperience.brain_regions
    };
}

// Convert multiple API experiences
export function convertToExperiences(apiExperiences: APIExperience[]): Experience[] {
    return apiExperiences.map(convertToExperience);
} 