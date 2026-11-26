// (This script is intended to be run in a browser via a dynamic import, e.g. from scripts/print-brain-mesh-names.html)
// Dynamically load Three.js and GLTFLoader if not already present (using CDN)
(async function() {
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
  // Use a relative URL (e.g. /models/brain_areas.glb) so that the browser loads the model from your public folder.
  const modelUrl = '/models/brain_areas.glb';
  console.log("Attempting to load model from:", modelUrl);
  loader.load(modelUrl, (gltf) => {
    console.log("--- Mesh names in brain_areas.glb ---");
    gltf.scene.traverse((node) => {
      if (node.isMesh) {
        console.log("- " + node.name);
      }
    });
    console.log("--- End of mesh list ---");
  }, undefined, (err) => {
    console.error("Error loading model:", err);
  });
})(); 