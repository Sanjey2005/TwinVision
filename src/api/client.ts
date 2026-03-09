// Types based on expected backend ComparisonPayload schema

export type GeneratorModel = 'flux_schnell' | 'sd_3_5';

export interface GenerationMetrics {
    clip_sim?: number;
    brisque?: number;
    niqe?: number;
    ssim?: number;
    lpips?: number;
}

export interface ModelResult {
    video_url: string;
    metrics: GenerationMetrics;
}

export interface ComparisonPayload {
    job_id: string;
    status: 'pending' | 'processing' | 'completed' | 'failed';
    prompt: string;
    num_images: number;
    results?: {
        flux: ModelResult;
        sd35: ModelResult;
        winner?: 'flux' | 'sd35' | 'tie';
    };
    error?: string;
}

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

/**
 * Initiates a new video generation pipeline comparison.
 * @param prompt The descriptive text prompt to generate images from.
 * @param nImages The number of frames/images to generate for the video (default 24).
 * @returns The initialized ComparisonPayload containing the job_id.
 */
export const startGeneration = async (prompt: string, nImages: number = 24): Promise<ComparisonPayload> => {
    const response = await fetch(`${API_BASE_URL}/generate`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt, n_images: nImages }),
    });

    if (!response.ok) {
        throw new Error(`Failed to start generation: ${response.statusText}`);
    }

    return response.json();
};

/**
 * Polls the current status of an active comparison job.
 * @param jobId The unique identifier for the job.
 * @returns The current status within a ComparisonPayload.
 */
export const pollStatus = async (jobId: string): Promise<ComparisonPayload> => {
    const response = await fetch(`${API_BASE_URL}/status/${jobId}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
        },
    });

    if (!response.ok) {
        throw new Error(`Failed to poll status: ${response.statusText}`);
    }

    return response.json();
};

/**
 * Retrieves the final compiled videos and metrics for a completed job.
 * @param jobId The unique identifier for the job.
 * @returns The populated ComparisonPayload with URLs and metrics.
 */
export const getResults = async (jobId: string): Promise<ComparisonPayload> => {
    const response = await fetch(`${API_BASE_URL}/results/${jobId}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
        },
    });

    if (!response.ok) {
        throw new Error(`Failed to fetch results: ${response.statusText}`);
    }

    return response.json();
};
