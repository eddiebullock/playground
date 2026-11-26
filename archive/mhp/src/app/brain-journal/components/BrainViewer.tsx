'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader, GLTF } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { useMediaQuery } from 'react-responsive';
import BrainLoadingState from './BrainLoadingState';
import BrainFallback from './BrainFallback';

interface Experience {
    type: string;
    intensity: number;
    evidence: string;
    brain_regions: string[];
}

interface BrainViewerProps {
    experiences: Experience[];
}

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
    sleep_quality: '#a55eea'
};

// Brain region descriptions
const BRAIN_REGION_DESCRIPTIONS: Record<string, string> = {
    amygdala: "Involved in processing emotions, particularly fear and anxiety",
    prefrontal_cortex: "Responsible for decision-making, planning, and emotional regulation",
    hippocampus: "Critical for memory formation and spatial navigation",
    hypothalamus: "Controls basic bodily functions and hormone regulation",
    nucleus_accumbens: "Plays a key role in reward and pleasure processing",
    ventral_tegmental_area: "Important for motivation and reward processing",
    motor_cortex: "Controls voluntary movement and coordination",
    cerebellum: "Essential for balance, coordination, and motor learning",
    basal_ganglia: "Involved in movement control and habit formation",
    temporal_lobe: "Processes auditory information and language",
    mirror_neurons: "Help understand others' actions and emotions",
    parietal_lobe: "Processes sensory information and spatial awareness",
    insula: "Involved in emotional awareness and interoception",
    anterior_cingulate_cortex: "Plays a role in emotional regulation and decision-making"
};

interface TooltipData {
    region: string;
    experience: Experience;
    position: { x: number; y: number };
}

export default function BrainViewer({ experiences }: BrainViewerProps) {
    const containerRef = useRef<HTMLDivElement>(null);
    const sceneRef = useRef<THREE.Scene | null>(null);
    const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
    const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
    const controlsRef = useRef<OrbitControls | null>(null);
    const brainModelRef = useRef<THREE.Group | null>(null);
    const raycasterRef = useRef<THREE.Raycaster>(new THREE.Raycaster());
    const mouseRef = useRef<THREE.Vector2>(new THREE.Vector2());
    const [tooltip, setTooltip] = useState<TooltipData | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [hasWebGL, setHasWebGL] = useState(true);
    const [isReplaying, setIsReplaying] = useState(false);
    const isMobile = useMediaQuery({ maxWidth: 768 });

    // Check WebGL support
    useEffect(() => {
        try {
            const canvas = document.createElement('canvas');
            const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
            setHasWebGL(!!gl);
        } catch (e) {
            setHasWebGL(false);
        }
    }, []);

    // Handle region activation animation
    const activateRegions = useCallback((regions: THREE.Mesh[], delay: number = 0) => {
        regions.forEach((region, index) => {
            setTimeout(() => {
                const material = region.material as THREE.MeshPhongMaterial;
                if (material) {
                    const originalColor = region.userData.originalColor;
                    const targetColor = new THREE.Color(EXPERIENCE_COLORS[region.userData.experience.type]);
                    
                    // Animate color transition
                    const duration = 1000;
                    const startTime = Date.now();
                    
                    const animate = () => {
                        const elapsed = Date.now() - startTime;
                        const progress = Math.min(elapsed / duration, 1);
                        
                        material.color.lerpColors(originalColor, targetColor, progress);
                        material.opacity = 0.6 + (progress * 0.2);
                        
                        if (progress < 1) {
                            requestAnimationFrame(animate);
                        }
                    };
                    
                    animate();
                }
            }, delay * index);
        });
    }, []);

    // Replay activation sequence
    const handleReplay = useCallback(() => {
        if (!brainModelRef.current) return;
        
        setIsReplaying(true);
        const regions: THREE.Mesh[] = [];
        
        brainModelRef.current.traverse((child) => {
            if (child instanceof THREE.Mesh && child.userData.region) {
                regions.push(child);
                // Reset to original state
                const material = child.material as THREE.MeshPhongMaterial;
                if (material) {
                    material.color.copy(child.userData.originalColor);
                    material.opacity = 0.6;
                }
            }
        });
        
        activateRegions(regions, 500);
        setTimeout(() => setIsReplaying(false), regions.length * 500 + 1000);
    }, [activateRegions]);

    // Load brain model
    useEffect(() => {
        if (!containerRef.current) return;

        // Initialize scene
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xf8fafc);
        sceneRef.current = scene;

        // Initialize camera
        const camera = new THREE.PerspectiveCamera(
            45,
            containerRef.current.clientWidth / containerRef.current.clientHeight,
            0.1,
            1000
        );
        camera.position.set(0, 0, 5);
        cameraRef.current = camera;

        // Initialize renderer
        const renderer = new THREE.WebGLRenderer({ 
            antialias: true,
            alpha: true 
        });
        renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        containerRef.current.appendChild(renderer.domElement);
        rendererRef.current = renderer;

        // Add orbit controls
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.rotateSpeed = 0.5;
        controls.minDistance = 3;
        controls.maxDistance = 10;
        controlsRef.current = controls;

        // Lighting setup
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(1, 1, 1);
        scene.add(directionalLight);

        const backLight = new THREE.DirectionalLight(0xffffff, 0.3);
        backLight.position.set(-1, -1, -1);
        scene.add(backLight);

        // Load brain model
        const loader = new GLTFLoader();
        console.log('Attempting to load brain model...');
        loader.load(
            '/models/brain_areas.glb', // Try brain_areas.glb first
            (gltf: GLTF) => {
                console.log('Brain model loaded successfully!');
                console.log('Model structure:');
                gltf.scene.traverse((node) => {
                    if (node instanceof THREE.Mesh) {
                        console.log(`Found mesh: ${node.name}`);
                    }
                });
                
                const brainModel = gltf.scene;
                brainModel.scale.set(2, 2, 2);
                brainModel.position.set(0, 0, 0);
                scene.add(brainModel);
                brainModelRef.current = brainModel;
                setIsLoading(false);

                // Add region meshes
                experiences.forEach(experience => {
                    experience.brain_regions.forEach(region => {
                        const regionMesh = brainModel.getObjectByName(region);
                        if (regionMesh && regionMesh instanceof THREE.Mesh) {
                            console.log(`Found region mesh for: ${region}`);
                            const material = new THREE.MeshPhongMaterial({
                                color: new THREE.Color(EXPERIENCE_COLORS[experience.type]),
                                transparent: true,
                                opacity: 0.6,
                                side: THREE.DoubleSide
                            });
                            regionMesh.material = material;
                            regionMesh.userData = {
                                region,
                                experience,
                                originalColor: material.color.clone()
                            };
                        } else {
                            console.warn(`Region mesh not found for: ${region}`);
                        }
                    });
                });
            },
            (progress) => {
                console.log('Loading progress:', (progress.loaded / progress.total * 100).toFixed(2) + '%');
            },
            (error: unknown) => {
                console.error('Error loading brain model:', error instanceof Error ? error.message : String(error));
                setIsLoading(false);
            }
        );

        // Mouse move handler for tooltips
        const handleMouseMove = (event: MouseEvent) => {
            if (!containerRef.current) return;

            const rect = containerRef.current.getBoundingClientRect();
            mouseRef.current.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            mouseRef.current.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

            raycasterRef.current.setFromCamera(mouseRef.current, camera);

            if (brainModelRef.current) {
                const intersects = raycasterRef.current.intersectObjects(
                    brainModelRef.current.children,
                    true
                );

                if (intersects.length > 0) {
                    const object = intersects[0].object;
                    if (object.userData.region) {
                        const { region, experience } = object.userData;
                        setTooltip({
                            region,
                            experience,
                            position: {
                                x: event.clientX,
                                y: event.clientY
                            }
                        });

                        // Highlight effect
                        if (object instanceof THREE.Mesh) {
                            const material = object.material as THREE.MeshPhongMaterial;
                            material.emissive = new THREE.Color(0x333333);
                            material.opacity = 0.8;
                        }
                    }
                } else {
                    setTooltip(null);
                    // Reset all materials
                    brainModelRef.current.traverse((child) => {
                        if (child instanceof THREE.Mesh && child.userData.region) {
                            const material = child.material as THREE.MeshPhongMaterial;
                            material.emissive = new THREE.Color(0x000000);
                            material.opacity = 0.6;
                        }
                    });
                }
            }
        };

        // Animation loop
        const animate = () => {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        };
        animate();

        // Event listeners
        window.addEventListener('mousemove', handleMouseMove);
        window.addEventListener('resize', handleResize);

        // Cleanup
        return () => {
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('resize', handleResize);
            if (containerRef.current && renderer.domElement) {
                containerRef.current.removeChild(renderer.domElement);
            }
            renderer.dispose();
        };
    }, [experiences]);

    // Handle window resize
    const handleResize = () => {
        if (!containerRef.current || !cameraRef.current || !rendererRef.current) return;
        
        const camera = cameraRef.current;
        const renderer = rendererRef.current;
        
        camera.aspect = containerRef.current.clientWidth / containerRef.current.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    };

    // Update brain regions based on experiences
    useEffect(() => {
        if (!brainModelRef.current || !sceneRef.current) return;
        
        const regions: THREE.Mesh[] = [];
        brainModelRef.current.traverse((child) => {
            if (child instanceof THREE.Mesh && child.userData.region) {
                regions.push(child);
            }
        });
        
        if (!isReplaying) {
            activateRegions(regions);
        }
    }, [experiences, activateRegions, isReplaying]);

    if (!hasWebGL) {
        return <BrainFallback experiences={experiences} />;
    }

    return (
        <div className="relative w-full h-full min-h-[500px]">
            {isLoading && <BrainLoadingState />}
            <div 
                ref={containerRef} 
                className="w-full h-full rounded-lg overflow-hidden"
            />
            {tooltip && (
                <div
                    className={`
                        absolute z-10 bg-white rounded-lg shadow-lg p-4 max-w-xs
                        transform -translate-x-1/2 -translate-y-full
                        transition-opacity duration-200
                        ${isMobile ? 'bottom-4 left-4 translate-y-0' : ''}
                    `}
                    style={{
                        left: isMobile ? 'auto' : tooltip.position.x,
                        top: isMobile ? 'auto' : tooltip.position.y - 10,
                        pointerEvents: 'none'
                    }}
                >
                    <h3 className="font-semibold text-gray-900 mb-1">
                        {tooltip.region.split('_').map(word => 
                            word.charAt(0).toUpperCase() + word.slice(1)
                        ).join(' ')}
                    </h3>
                    <p className="text-sm text-gray-600 mb-2">
                        {BRAIN_REGION_DESCRIPTIONS[tooltip.region]}
                    </p>
                    <div className="text-sm">
                        <p className="font-medium text-gray-700">
                            {tooltip.experience.type.split('_').map(word => 
                                word.charAt(0).toUpperCase() + word.slice(1)
                            ).join(' ')}
                        </p>
                        <p className="text-gray-600 italic">
                            "{tooltip.experience.evidence}"
                        </p>
                    </div>
                </div>
            )}
            <div className="absolute bottom-4 left-4 bg-white/90 rounded-lg p-3 shadow-lg">
                <div className="flex items-center justify-between mb-2">
                    <h4 className="text-sm font-medium text-gray-900">Experience Colors</h4>
                    <button
                        onClick={handleReplay}
                        disabled={isReplaying}
                        className={`
                            px-2 py-1 text-xs rounded
                            ${isReplaying 
                                ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                                : 'bg-indigo-100 text-indigo-600 hover:bg-indigo-200'
                            }
                        `}
                    >
                        {isReplaying ? 'Replaying...' : 'Replay Activation'}
                    </button>
                </div>
                <div className="grid grid-cols-2 gap-2">
                    {Object.entries(EXPERIENCE_COLORS).map(([type, color]) => (
                        <div key={type} className="flex items-center space-x-2">
                            <div 
                                className="w-3 h-3 rounded-full" 
                                style={{ backgroundColor: color }}
                            />
                            <span className="text-xs text-gray-600">
                                {type.split('_').map(word => 
                                    word.charAt(0).toUpperCase() + word.slice(1)
                                ).join(' ')}
                            </span>
                        </div>
                    ))}
                </div>
                <p className="mt-2 text-xs text-gray-500 italic">
                    This visualization shows research about typical brain activity patterns, 
                    not your specific brain activity
                </p>
            </div>
        </div>
    );
} 