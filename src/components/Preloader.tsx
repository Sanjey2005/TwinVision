import { motion } from 'framer-motion';
import { useEffect, useState } from 'react';

interface PreloaderProps {
    onComplete: () => void;
}

const Preloader = ({ onComplete }: PreloaderProps) => {
    const [progress, setProgress] = useState(0);

    useEffect(() => {
        // Simulate loading progress
        const duration = 2500; // 2.5 seconds total loading time
        const interval = 25; // Update every 25ms
        const steps = duration / interval;
        let currentStep = 0;

        const timer = setInterval(() => {
            currentStep++;
            const newProgress = Math.min(Math.round((currentStep / steps) * 100), 100);
            setProgress(newProgress);

            if (currentStep >= steps) {
                clearInterval(timer);
                // Small delay before unmounting to show 100%
                setTimeout(() => {
                    onComplete();
                }, 400);
            }
        }, interval);

        return () => clearInterval(timer);
    }, [onComplete]);

    return (
        <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center bg-background overflow-hidden"
            initial={{ opacity: 1 }}
            exit={{
                y: '-100%',
                transition: {
                    duration: 0.8,
                    ease: [0.76, 0, 0.24, 1] // Custom easing for smooth slide-up
                }
            }}
        >
            {/* Background Marquee Text */}
            <div className="absolute inset-0 flex items-center justify-center opacity-10 select-none pointer-events-none overflow-hidden">
                <h1 className="text-[15vw] font-heading font-black whitespace-nowrap text-outline" style={{ display: 'inline-block' }}>
                    TWINVISION // AI MODEL COMPARISON //
                </h1>
            </div>

            {/* Main Counter */}
            <div className="relative z-10 flex flex-col items-center">
                <div className="overflow-hidden">
                    <motion.div
                        initial={{ y: '100%' }}
                        animate={{ y: '0%' }}
                        transition={{ duration: 0.5, ease: 'easeOut' }}
                        className="flex items-baseline"
                    >
                        <h2 className="text-sm tracking-[0.3em] text-muted mb-2 font-body uppercase">
                            Loading pipeline
                        </h2>
                    </motion.div>
                </div>

                <div className="overflow-hidden">
                    <motion.div
                        initial={{ y: '100%' }}
                        animate={{ y: '0%' }}
                        transition={{ duration: 0.5, delay: 0.1, ease: 'easeOut' }}
                        className="text-7xl md:text-9xl font-heading font-bold text-cream"
                    >
                        {progress.toString().padStart(2, '0')}<span className="text-primary-neon">%</span>
                    </motion.div>
                </div>
            </div>
        </motion.div>
    );
};

export default Preloader;
