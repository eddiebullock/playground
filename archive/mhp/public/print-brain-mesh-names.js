// Load this file in your browser (e.g., open /print-brain-mesh-names.js in your browser)
// Make sure Three.js and GLTFLoader are available (via CDN or your project)

(async function() {
  // Dynamically load Three.js and GLTFLoader if not already present
  if (!window.THREE) {
    const threeScript = document.createElement('script');
    threeScript.src = 'https://unpkg.com/three@0.152.2/build/three.min.js';
    document.head.appendChild(threeScript);
    await new Promise(res => threeScript.onload = res);
  }
  if (!window.THREE.GLTFLoader) {
    const loaderScript = document.createElement('script');
    loaderScript.src = 'https://unpkg.com/three@0.152.2/examples/js/loaders/GLTFLoader.js';
    document.head.appendChild(loaderScript);
    await new Promise(res => loaderScript.onload = res);
  }

  const loader = new window.THREE.GLTFLoader();
  loader.load('/models/brain_areas.glb', (gltf) => {
    console.log('--- Mesh names in brain_areas.glb ---');
    gltf.scene.traverse((node) => {
      if (node.isMesh) {
        console.log(node.name);
      }
    });
    console.log('--- End of mesh list ---');
  }, undefined, (err) => {
    console.error('Error loading model:', err);
  });
})(); 