import { motion } from 'framer-motion';
import { Loader2, Zap } from 'lucide-react';

interface LoadingOverlayProps {
    status: string;
    isVisible: boolean;
}

const LoadingOverlay = ({ status, isVisible }: LoadingOverlayProps) => {
    if (!isVisible) return null;

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-6 bg-background/80 backdrop-blur-xl"
        >
            <div className="max-w-md w-[95%] sm:w-full glass-card rounded-3xl p-6 sm:p-10 flex flex-col items-center text-center relative overflow-hidden border-primary-neon/30">
                {/* Background Ambient Glow */}
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-48 h-48 bg-primary-neon/10 rounded-full blur-[60px] pointer-events-none"></div>

                <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ repeat: Infinity, duration: 4, ease: "linear" }}
                    className="relative z-10 p-6 rounded-full bg-black/40 border border-white/5 shadow-2xl mb-8"
                >
                    <Loader2 className="w-12 h-12 text-primary-neon" />

                    {/* Inner counter-rotating core */}
                    <motion.div
                        animate={{ rotate: -360 }}
                        transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
                        className="absolute inset-0 flex items-center justify-center text-white/50"
                    >
                        <Zap className="w-5 h-5 opacity-50" />
                    </motion.div>
                </motion.div>

                <h3 className="text-2xl font-heading font-bold text-cream mb-2 uppercase tracking-wide relative z-10 flex flex-col items-center">
                    Generating <span className="text-primary-neon mt-1">Comparisons</span>
                </h3>

                <div className="font-body text-sm text-muted/80 h-10 flex items-center justify-center relative z-10">
                    <motion.span
                        key={status}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                    >
                        {status || "Synthesizing latents..."}
                    </motion.span>
                </div>

                {/* Progress Bar (Simulated Indeterminate) */}
                <div className="w-full h-1 bg-black/50 rounded-full mt-6 overflow-hidden relative z-10">
                    <motion.div
                        className="h-full bg-primary-neon shadow-[0_0_10px_#ccff00]"
                        animate={{
                            x: ["-100%", "100%"]
                        }}
                        transition={{
                            repeat: Infinity,
                            duration: 2,
                            ease: "easeInOut"
                        }}
                        style={{ width: "50%" }}
                    />
                </div>
            </div>
        </motion.div>
    );
};

export default LoadingOverlay;
