import { useState } from 'react';
import { motion } from 'framer-motion';
import type { Variants } from 'framer-motion';
import { Terminal, Sparkles } from 'lucide-react';

interface HeroProps {
    onGenerate?: (prompt: string) => void;
}

const Hero = ({ onGenerate }: HeroProps) => {
    const [prompt, setPrompt] = useState('');

    const handleGenerate = (e: React.FormEvent) => {
        e.preventDefault();
        if (!prompt.trim()) return;

        if (onGenerate) {
            onGenerate(prompt);
        }
    };

    // Animation variants
    const containerVariants: Variants = {
        hidden: { opacity: 0 },
        visible: {
            opacity: 1,
            transition: {
                staggerChildren: 0.1,
                delayChildren: 0.2, // Wait for preloader to finish
            },
        },
    };

    const itemVariants: Variants = {
        hidden: { opacity: 0, y: 50 },
        visible: {
            opacity: 1,
            y: 0,
            transition: { duration: 0.8, ease: [0.76, 0, 0.24, 1] },
        },
    };

    return (
        <section className="relative min-h-[90vh] flex flex-col justify-center pt-20" id="hero">
            <motion.div
                className="w-full"
                variants={containerVariants}
                initial="hidden"
                animate="visible"
            >
                {/* Massive Typography */}
                <div className="mb-16">
                    <motion.h1 className="flex flex-col font-heading font-black tracking-tighter uppercase leading-[0.85] text-[12vw] sm:text-[10vw] md:text-[8vw] lg:text-[7.5rem] xl:text-[9rem]">
                        <motion.span variants={itemVariants} className="text-cream">
                            PROMPT
                        </motion.span>
                        <motion.span variants={itemVariants} className="text-outline">
                            TO VIDEO
                        </motion.span>
                        <motion.span variants={itemVariants} className="text-cream">
                            EVALUATION.
                        </motion.span>
                    </motion.h1>
                </div>

                {/* Terminal Input Form */}
                <motion.div variants={itemVariants} className="max-w-4xl">
                    <form
                        onSubmit={handleGenerate}
                        className="relative flex flex-col sm:flex-row items-center w-full p-2 rounded-2xl glass-card transition-all duration-300 focus-within:border-primary-neon/50 focus-within:shadow-[0_0_30px_rgba(204,255,0,0.15)] group"
                    >
                        {/* Terminal Icon (Hidden on very small screens to save space) */}
                        <div className="hidden sm:block pl-6 pr-4 text-muted group-focus-within:text-primary-neon transition-colors duration-300">
                            <Terminal className="w-6 h-6" />
                        </div>

                        {/* Input Field */}
                        <input
                            type="text"
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                            placeholder="Enter a prompt to compare generation pipelines..."
                            className="flex-1 bg-transparent border-none outline-none text-base sm:text-lg md:text-xl font-body text-cream placeholder:text-muted/50 py-4 px-4 sm:px-0 w-full"
                        />

                        {/* Submit Button */}
                        <button
                            type="submit"
                            disabled={!prompt.trim()}
                            className={`w-full sm:w-auto mt-2 sm:mt-0 sm:ml-4 flex items-center justify-center space-x-2 px-6 md:px-8 py-4 rounded-xl font-heading font-bold tracking-widest uppercase transition-all duration-300 ${prompt.trim()
                                ? 'bg-primary-neon text-background hover:scale-105 neon-glow cursor-pointer'
                                : 'bg-white/5 text-muted cursor-not-allowed border border-white/10'
                                }`}
                        >
                            <span>
                                GENERATE
                            </span>
                            <Sparkles className="w-5 h-5 ml-2" />
                        </button>
                    </form>

                    <div className="mt-4 flex items-center space-x-6 text-sm font-body text-muted/60 pl-6">
                        <span className="flex items-center"><span className="w-1.5 h-1.5 rounded-full bg-primary-neon/50 mr-2"></span> FLUX.1 [schnell]</span>
                        <span className="flex items-center"><span className="w-1.5 h-1.5 rounded-full bg-primary-neon/50 mr-2"></span> Stable Diffusion 3.5</span>
                    </div>
                </motion.div>
            </motion.div>

            {/* Background ambient glow */}
            <div className="absolute top-1/2 left-1/4 -translate-y-1/2 w-96 h-96 bg-primary-neon/10 rounded-full blur-[120px] pointer-events-none -z-10"></div>
        </section>
    );
};

export default Hero;
