import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Suspense, lazy, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import Preloader from './components/Preloader';
import LoadingOverlay from './components/LoadingOverlay';
import { startGeneration, pollStatus, getResults } from './api/client';
import type { ComparisonPayload } from './api/client';

// Lazy load pages for better performance
const HomePage = lazy(() => import('./pages/HomePage'));

function App() {
  const [loading, setLoading] = useState(true);

  // App-level Orchestration State
  const [comparisonData, setComparisonData] = useState<ComparisonPayload | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [jobStatus, setJobStatus] = useState<string>('');

  const handleGenerate = async (prompt: string) => {
    try {
      setIsGenerating(true);
      setJobStatus('Initializing Pipeline...');

      // 1. Start generation
      const job = await startGeneration(prompt);
      const jobId = job.job_id;

      // 2. Poll Status
      setJobStatus('Awaiting Dispatch...');
      const pollInterval = setInterval(async () => {
        try {
          const statusResult = await pollStatus(jobId);

          if (statusResult.status === 'processing') {
            setJobStatus('Synthesizing & Compiling...');
          } else if (statusResult.status === 'completed') {
            clearInterval(pollInterval);
            setJobStatus('Finalizing Metrics...');

            // 3. Get final results
            const finalResults = await getResults(jobId);
            setComparisonData(finalResults);
            setIsGenerating(false);
          } else if (statusResult.status === 'failed') {
            clearInterval(pollInterval);
            setJobStatus('Generation Failed.');
            console.error("Pipeline Error:", statusResult.error);
            setTimeout(() => setIsGenerating(false), 3000);
          }
        } catch (pollErr) {
          console.error("Polling Error:", pollErr);
        }
      }, 2000); // Poll every 2 seconds

    } catch (error) {
      console.error("Failed to start pipeline:", error);
      setJobStatus('Failed to connect to API.');
      setTimeout(() => setIsGenerating(false), 3000);
    }
  };

  return (
    <>
      <AnimatePresence mode="wait">
        {loading && (
          <Preloader key="preloader" onComplete={() => setLoading(false)} />
        )}
      </AnimatePresence>

      <AnimatePresence>
        {isGenerating && (
          <LoadingOverlay key="loadingOverlay" isVisible={isGenerating} status={jobStatus} />
        )}
      </AnimatePresence>

      {!loading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.2 }}
        >
          <Router>
            <Suspense fallback={<div className="h-screen bg-background" />}>
              <Routes>
                <Route
                  path="/"
                  element={
                    <HomePage
                      onGenerate={handleGenerate}
                      comparisonData={comparisonData}
                    />
                  }
                />
              </Routes>
            </Suspense>
          </Router>
        </motion.div>
      )}
    </>
  );
}

export default App;
