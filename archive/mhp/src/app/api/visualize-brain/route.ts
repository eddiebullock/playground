import { NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';

interface VisualizationRequest {
    experiences: {
        type: string;
        intensity: number;
        brain_regions: string[];
    }[];
}

export async function POST(request: Request) {
    try {
        const data: VisualizationRequest = await request.json();
        
        // Get the path to the Python executable in the virtual environment
        const pythonPath = path.join(process.cwd(), 'venv', 'bin', 'python3');
        const pythonScript = path.join(process.cwd(), 'scripts', 'visualize_brain.py');

        console.log('Using Python at:', pythonPath);
        console.log('Script path:', pythonScript);
        console.log('Input data:', JSON.stringify(data, null, 2));

        // Call Python script with virtual environment Python
        const pythonProcess = spawn(pythonPath, [pythonScript]);

        let visualizationData = '';
        let errorData = '';

        // Write input data to Python script's stdin
        pythonProcess.stdin.write(JSON.stringify(data));
        pythonProcess.stdin.end();

        // Collect data from Python script
        pythonProcess.stdout.on('data', (data) => {
            const chunk = data.toString();
            console.log('Python stdout chunk:', chunk);
            visualizationData += chunk;
        });

        pythonProcess.stderr.on('data', (data) => {
            const chunk = data.toString();
            console.error('Python stderr chunk:', chunk);
            errorData += chunk;
        });

        // Wait for Python script to complete
        await new Promise((resolve, reject) => {
            pythonProcess.on('close', (code) => {
                if (code === 0) {
                    // Clean up the visualization data - remove any non-JSON content
                    visualizationData = visualizationData.trim();
                    try {
                        // Try to parse the data to validate it's JSON
                        JSON.parse(visualizationData);
                        resolve(null);
                    } catch (e: unknown) {
                        console.error('Invalid JSON output:', visualizationData);
                        const errorMessage = e instanceof Error ? e.message : String(e);
                        reject(new Error(`Invalid JSON output from Python script: ${errorMessage}`));
                    }
                } else {
                    console.error('Python script failed with code:', code);
                    console.error('Error output:', errorData);
                    reject(new Error(`Python script exited with code ${code}: ${errorData}`));
                }
            });
        });

        // Parse the visualization data
        const result = JSON.parse(visualizationData);
        console.log('Visualization result:', { 
            ...result, 
            visualization: result.visualization ? `${result.visualization.substring(0, 50)}...` : null 
        });

        return NextResponse.json(result);
    } catch (error) {
        console.error('Error in visualize-brain:', error);
        return NextResponse.json(
            { error: 'Failed to generate brain visualization: ' + (error instanceof Error ? error.message : String(error)) },
            { status: 500 }
        );
    }
} 