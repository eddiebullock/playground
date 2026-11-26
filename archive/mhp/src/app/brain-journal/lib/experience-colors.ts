import * as THREE from 'three';

export type ExperienceType = 'learning' | 'memory' | 'emotion' | 'attention' | 'perception';

// Experience type to color mapping
const EXPERIENCE_COLORS: Record<string, string> = {
    anxiety: '#ff6b6b',
    stress: '#ff9f43',
    happiness: '#20bf6b',
    sadness: '#3867d6',
    anger: '#eb3b5a',
    exercise: '#45aaf2',
    social_interaction: '#8854d0',
    learning: '#4b6584',
    creativity: '#f7b731',
    meditation: '#2ecc71',
    sleep_quality: '#a55eea',
    memory: '#00b894',
    focus: '#0984e3',
    relaxation: '#00cec9',
    emotion: '#6c5ce7',
    decision_making: '#e17055',
    physical_activity: '#fdcb6e',
    eating: '#e84393',
    music: '#00b894',
    art: '#6c5ce7'
};

export function getExperienceColor(type: string): THREE.Color {
    const hexColor = EXPERIENCE_COLORS[type.toLowerCase()] || '#cccccc';
    return new THREE.Color(hexColor);
} 