#!/usr/bin/env python3

import sys
import json
import numpy as np
import nibabel as nib
from nilearn import datasets, plotting, surface, image
from nilearn.image import new_img_like, math_img
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.patches as mpatches
import seaborn as sns
import logging
import nilearn
import os
import shutil
import cortex
from PIL import Image
import traceback
import urllib.request
import gzip

# Set the style to a more paper-like appearance
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Set up logging with more detail
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants for visualization
FIGURE_SIZE = (36, 12)
DPI = 300

# Load the MNI152 template and anatomical image
mni_template = datasets.load_mni152_template()
anatomical_img = datasets.load_mni152_template(resolution=2)

# Define vibrant colors for each brain region with increased opacity
REGION_COLORS = {
    'prefrontal_cortex': '#FF0000',  # Pure red
    'temporal_lobe': '#0000FF',      # Pure blue
    'parietal_lobe': '#00FF00',      # Pure green
    'occipital_lobe': '#FFFF00',     # Pure yellow
    'amygdala': '#FF00FF',           # Pure magenta
    'hippocampus': '#800080',        # Deep purple
    'cerebellum': '#00FFFF',         # Pure cyan
    'brainstem': '#FFA500',          # Orange
    'thalamus': '#0080FF',           # Bright blue
    'hypothalamus': '#8000FF',       # Bright purple
    'insula': '#FFD700',             # Gold
    'motor_cortex': '#00FF80',       # Bright green
    'basal_ganglia': '#FF0080'       # Bright pink
}

# Define descriptions for each brain region
BRAIN_REGION_DESCRIPTIONS = {
    'prefrontal_cortex': 'Executive functions, decision making, and personality',
    'temporal_lobe': 'Auditory processing, memory, and language comprehension',
    'parietal_lobe': 'Sensory integration, spatial awareness, and attention',
    'occipital_lobe': 'Visual processing and perception',
    'amygdala': 'Emotional processing, fear, and memory formation',
    'hippocampus': 'Memory formation, learning, and spatial navigation',
    'cerebellum': 'Motor coordination, balance, and movement control',
    'brainstem': 'Basic life functions, breathing, and heart rate',
    'thalamus': 'Sensory relay station and consciousness regulation',
    'hypothalamus': 'Hormone regulation, body temperature, and basic drives',
    'insula': 'Interoception, emotional awareness, and empathy',
    'motor_cortex': 'Voluntary movement control and planning',
    'basal_ganglia': 'Movement control, habit formation, and reward processing'
}

# Create custom colormaps for each region
REGION_CMAPS = {
    region: LinearSegmentedColormap.from_list(
        f'{region}_cmap',
        ['#FFFFFF', color],
        N=256
    )
    for region, color in REGION_COLORS.items()
}

# Define brain region coordinates (MNI space)
REGION_COORDS = {
    'prefrontal_cortex': [(0, 50, 0), (0, 45, 10), (0, 40, 20)],
    'temporal_lobe': [(-50, -20, -10), (50, -20, -10), (-45, -30, 0), (45, -30, 0)],
    'parietal_lobe': [(-30, -50, 50), (30, -50, 50), (-25, -60, 40), (25, -60, 40)],
    'occipital_lobe': [(-20, -80, 0), (20, -80, 0), (0, -90, 10)],
    'amygdala': [(-20, -10, -15), (20, -10, -15)],
    'hippocampus': [(-25, -20, -15), (25, -20, -15)],
    'cerebellum': [(0, -50, -30), (-10, -55, -25), (10, -55, -25)],
    'brainstem': [(0, -30, -40)],
    'thalamus': [(0, -20, 0), (-5, -25, 5), (5, -25, 5)],
    'hypothalamus': [(0, -5, -10)],
    'insula': [(-35, 0, 0), (35, 0, 0)],
    'motor_cortex': [(-40, -20, 50), (40, -20, 50)],
    'basal_ganglia': [(-15, 0, 0), (15, 0, 0)]
}

def download_brain_template():
    """Download and prepare the MNI152 brain template."""
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(template_dir, exist_ok=True)
    
    template_path = os.path.join(template_dir, 'mni152.nii.gz')
    if os.path.exists(template_path):
        try:
            # Verify existing template
            with gzip.open(template_path, 'rb') as f:
                f.read(1)
            logger.info("Using existing template file")
            return nib.load(template_path)
        except Exception as e:
            logger.warning(f"Existing template file is invalid: {str(e)}")
            os.remove(template_path)
    
    logger.info("Downloading MNI152 brain template...")
    # Using direct link to MNI152 template from NeuroVault
    url = "https://neurovault.org/media/images/35/35_MNI152_T1_1mm.nii.gz"
    temp_path = os.path.join(template_dir, 'temp_template.nii.gz')
    
    try:
        logger.debug("Attempting to download template...")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req) as response, open(temp_path, 'wb') as out_file:
            # Read in chunks to handle large files
            chunk_size = 8192
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
        
        # Verify the downloaded file
        try:
            # First verify it's a valid gzip file
            with gzip.open(temp_path, 'rb') as f:
                f.read(1)
            
            # Then verify it's a valid NIfTI file
            test_load = nib.load(temp_path)
            if not isinstance(test_load, nib.Nifti1Image):
                raise ValueError("Downloaded file is not a valid NIfTI image")
            
            # If we get here, both checks passed
            os.rename(temp_path, template_path)
            logger.info("Template downloaded and verified successfully")
            return test_load
            
        except Exception as e:
            logger.error(f"Downloaded file verification failed: {str(e)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise Exception("Failed to download a valid brain template file")
            
    except Exception as e:
        logger.error(f"Error downloading template: {str(e)}")
        # Fallback to a simpler template
        logger.info("Attempting to download fallback template...")
        fallback_url = "https://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nl_asym_09c_t1_tal_nlin_sym_09c.nii.gz"
        try:
            req = urllib.request.Request(fallback_url, headers=headers)
            with urllib.request.urlopen(req) as response, open(temp_path, 'wb') as out_file:
                chunk_size = 8192
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
            
            # Verify the fallback file
            try:
                with gzip.open(temp_path, 'rb') as f:
                    f.read(1)
                test_load = nib.load(temp_path)
                if not isinstance(test_load, nib.Nifti1Image):
                    raise ValueError("Fallback file is not a valid NIfTI image")
                os.rename(temp_path, template_path)
                logger.info("Fallback template downloaded and verified successfully")
                return test_load
            except Exception as e2:
                logger.error(f"Fallback file verification failed: {str(e2)}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise Exception("Failed to download any valid brain template file")
                
        except Exception as e2:
            logger.error(f"Error downloading fallback template: {str(e2)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise Exception("Failed to download any brain template. Please check your internet connection and try again.")

def create_activation_volume(experiences):
    """Create a 3D volume of brain activation."""
    try:
        logger.debug(f"Creating activation volume for {len(experiences)} experiences")
        # Initialize volume (91x109x91 MNI space)
        volume = np.zeros((91, 109, 91))
        
        # Create activation spheres for each experience
        for i, exp in enumerate(experiences):
            logger.debug(f"Processing experience {i+1}/{len(experiences)}")
            intensity = (exp['intensity'] / 5.0) * 2.0  # Scale intensity
            for region in exp['brain_regions']:
                if region in REGION_COORDS:
                    logger.debug(f"Adding activation for region: {region}")
                    for x, y, z in REGION_COORDS[region]:
                        # Convert MNI coordinates to volume indices
                        x_idx = int((x + 45) * 91/90)  # Scale to volume dimensions
                        y_idx = int((y + 90) * 109/180)
                        z_idx = int((z + 45) * 91/90)
                        
                        # Create activation sphere
                        xx, yy, zz = np.ogrid[:91, :109, :91]
                        distance = np.sqrt((xx - x_idx)**2 + (yy - y_idx)**2 + (zz - z_idx)**2)
                        sphere = np.exp(-distance**2 / (2 * 5**2))  # Gaussian sphere
                        volume = np.maximum(volume, sphere * intensity)
        
        # Normalize volume
        if np.max(volume) > 0:
            volume = volume / np.max(volume)
            logger.debug(f"Volume normalized. Max value: {np.max(volume)}")
        else:
            logger.warning("Volume is empty - no activation detected")
        
        return volume
    except Exception as e:
        logger.error(f"Error in create_activation_volume: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def plot_slice(ax, template_data, activation_data, slice_idx, view, title):
    """Plot a single slice of the brain with activation overlay."""
    try:
        logger.debug(f"Plotting {view} slice at index {slice_idx}")
        
        # Get the appropriate slice for both template and activation
        if view == 'sagittal':
            template_slice = template_data[slice_idx, :, :]
            activation_slice = activation_data[slice_idx, :, :]
        elif view == 'coronal':
            template_slice = template_data[:, slice_idx, :]
            activation_slice = activation_data[:, slice_idx, :]
        else:  # axial
            template_slice = template_data[:, :, slice_idx]
            activation_slice = activation_data[:, :, slice_idx]
        
        # Normalize template data for better visibility
        template_slice = (template_slice - template_slice.min()) / (template_slice.max() - template_slice.min())
        
        # Create RGB image with template as grayscale background
        rgb = np.zeros((*template_slice.shape, 3))
        rgb[..., 0] = template_slice  # Red channel
        rgb[..., 1] = template_slice  # Green channel
        rgb[..., 2] = template_slice  # Blue channel
        
        # Add activation overlay in red
        activation_mask = activation_slice > 0.1  # Threshold for visibility
        rgb[activation_mask, 0] = np.maximum(rgb[activation_mask, 0], activation_slice[activation_mask])
        rgb[activation_mask, 1] = np.minimum(rgb[activation_mask, 1], 1 - activation_slice[activation_mask])
        rgb[activation_mask, 2] = np.minimum(rgb[activation_mask, 2], 1 - activation_slice[activation_mask])
        
        # Plot the slice
        im = ax.imshow(rgb, 
                      origin='lower',
                      aspect='equal')
        
        # Add title
        ax.set_title(title, fontsize=24, pad=20)
        
        # Remove axis
        ax.axis('off')
        
        logger.debug(f"Successfully plotted {view} slice")
        return im
    except Exception as e:
        logger.error(f"Error in plot_slice for {view}: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def generate_visualization(experiences):
    """Generate high-quality brain visualization using matplotlib."""
    try:
        logger.info("Starting visualization generation")
        logger.debug(f"Received {len(experiences)} experiences")
        
        # Load brain template
        logger.info("Loading brain template")
        template = download_brain_template()
        template_data = template.get_fdata()
        
        # Create activation volume
        logger.info("Creating activation volume")
        activation_volume = create_activation_volume(experiences)
        
        # Create figure
        logger.info("Creating visualization")
        fig = plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
        
        # Define views and their slice indices
        views = [
            ('sagittal', 'Sagittal View (Left-Right)', 45),  # Middle slice
            ('coronal', 'Coronal View (Front-Back)', 54),    # Middle slice
            ('axial', 'Axial View (Top-Bottom)', 45)         # Middle slice
        ]
        
        # Plot each view
        for i, (view, title, slice_idx) in enumerate(views):
            logger.debug(f"Processing view {i+1}/3: {view}")
            ax = fig.add_subplot(1, 3, i+1)
            try:
                im = plot_slice(ax, template_data, activation_volume, slice_idx, view, title)
            except Exception as e:
                logger.error(f"Error plotting {view} view: {str(e)}")
                logger.error(traceback.format_exc())
                raise
        
        # Add a note about the visualization
        plt.figtext(
            0.5, 0.01,
            'Note: Red highlights indicate brain regions activated during your experiences. Brighter red represents stronger activation.',
            ha='center',
            fontsize=16,
            style='italic'
        )
        
        # Add colorbar
        logger.debug("Adding colorbar")
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Activation Intensity', fontsize=14)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.1, 0.9, 0.9])
        
        # Save to buffer
        logger.info("Saving visualization")
        buf = BytesIO()
        plt.savefig(
            buf,
            format='png',
            dpi=DPI,
            bbox_inches='tight',
            pad_inches=0.1,
            facecolor='white',
            edgecolor='none',
            transparent=False
        )
        plt.close(fig)
        
        # Convert to base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        logger.info("Visualization generated successfully")
        return {
            'visualization': f'data:image/png;base64,{img_str}',
            'message': 'Brain visualization generated successfully'
        }
        
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}")
        logger.error(traceback.format_exc())
        raise Exception(f"Failed to generate brain visualization: {str(e)}")

if __name__ == '__main__':
    try:
        # Read input from stdin
        logger.debug("Reading input from stdin")
        input_data = json.loads(sys.stdin.read())
        logger.debug(f"Received input data: {input_data}")
        
        if not input_data.get('experiences'):
            logger.error("No experiences found in input data")
            raise ValueError("No experiences provided in input data")
            
        result = generate_visualization(input_data['experiences'])
        print(json.dumps(result))
        logger.info("Successfully completed visualization generation")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON input: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1) 