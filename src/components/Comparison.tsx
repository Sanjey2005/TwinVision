import { useRef } from 'react';
import { motion, useInView } from 'framer-motion';
import type { Variants } from 'framer-motion';
import { Maximize2, Play } from 'lucide-react';

export interface ComparisonProps {
    fluxVideoUrl?: string;
    sd35VideoUrl?: string;
    winner?: 'flux' | 'sd35' | 'tie';
}

const Comparison = ({ fluxVideoUrl, sd35VideoUrl, winner }: ComparisonProps) => {
    const containerRef = useRef(null);
    const isInView = useInView(containerRef, { once: true, margin: "-50px" });

    const cardVariants: Variants = {
        hidden: { opacity: 0, y: 30 },
        visible: { opacity: 1, y: 0, transition: { duration: 0.8, ease: "easeOut" } }
    };

    return (
        <section className="py-24 relative overflow-hidden" id="comparison">
            <div className="max-w-7xl mx-auto px-6 md:px-12 relative z-10" ref={containerRef}>
                <div className="mb-16">
                    <h2 className="text-4xl md:text-6xl font-heading font-bold text-cream uppercase tracking-tighter">
                        Comparison <span className="text-outline">Showcase</span>
                    </h2>
                    <p className="text-muted font-body mt-4 max-w-2xl text-lg">
                        Side-by-side temporal and aesthetic analysis of the latest diffusion models.
                    </p>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    {/* FLUX Card */}
                    <motion.div
                        variants={cardVariants}
                        initial="hidden"
                        animate={isInView ? "visible" : "hidden"}
                        className={`relative glass-card rounded-3xl overflow-hidden aspect-video group ${winner === 'flux' || winner === 'tie' ? 'border-primary-neon/50 shadow-[0_0_30px_rgba(204,255,0,0.15)]' : ''}`}
                    >
                        {/* Status Badge */}
                        <div className="absolute top-6 left-6 z-20 flex gap-2">
                            <div className="px-3 py-1.5 rounded-full bg-black/60 backdrop-blur-md border border-white/10 font-heading text-xs font-bold tracking-widest text-cream uppercase">
                                FLUX.1 [schnell]
                            </div>
                            {winner === 'flux' && (
                                <div className="px-3 py-1.5 rounded-full bg-primary-neon text-black font-heading text-xs font-bold tracking-widest uppercase neon-glow flex items-center">
                                    <span className="w-1.5 h-1.5 rounded-full bg-black mr-2 animate-pulse"></span>
                                    Winner: Clarity
                                </div>
                            )}
                        </div>

                        {/* Video Player / Fallback */}
                        <div className="absolute inset-0 bg-[#0a0a0a] flex items-center justify-center">
                            {fluxVideoUrl ? (
                                <video
                                    src={fluxVideoUrl}
                                    className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity duration-500"
                                    autoPlay
                                    loop
                                    muted
                                    playsInline
                                />
                            ) : (
                                <div className="text-muted/30 font-heading text-4xl font-bold uppercase tracking-tighter mix-blend-overlay flex flex-col items-center">
                                    <Play className="w-16 h-16 mb-4 opacity-50" />
                                    <span>Awaiting Generation</span>
                                </div>
                            )}
                        </div>

                        {/* Overlay Controls (Decorative) */}
                        <div className="absolute bottom-6 right-6 z-20 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                            <button className="p-3 rounded-full bg-black/60 backdrop-blur-md border border-white/10 text-cream hover:bg-white/10 transition-colors">
                                <Maximize2 className="w-5 h-5" />
                            </button>
                        </div>
                    </motion.div>

                    {/* SD3.5 Card */}
                    <motion.div
                        variants={cardVariants}
                        initial="hidden"
                        animate={isInView ? "visible" : "hidden"}
                        transition={{ delay: 0.2 }}
                        className={`relative glass-card rounded-3xl overflow-hidden aspect-video group ${winner === 'sd35' || winner === 'tie' ? 'border-primary-blue/50 shadow-[0_0_30px_rgba(0,240,255,0.15)]' : ''}`}
                    >
                        {/* Status Badge */}
                        <div className="absolute top-6 left-6 z-20 flex gap-2">
                            <div className="px-3 py-1.5 rounded-full bg-black/60 backdrop-blur-md border border-white/10 font-heading text-xs font-bold tracking-widest text-cream uppercase">
                                Stable Diffusion 3.5
                            </div>
                            {winner === 'sd35' && (
                                <div className="px-3 py-1.5 rounded-full bg-primary-blue text-black font-heading text-xs font-bold tracking-widest uppercase blue-glow flex items-center bg-[#00f0ff]">
                                    <span className="w-1.5 h-1.5 rounded-full bg-black mr-2 animate-pulse"></span>
                                    Winner: Art Style
                                </div>
                            )}
                        </div>

                        {/* Video Player / Fallback */}
                        <div className="absolute inset-0 bg-[#0a0a0a] flex items-center justify-center">
                            {sd35VideoUrl ? (
                                <video
                                    src={sd35VideoUrl}
                                    className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity duration-500"
                                    autoPlay
                                    loop
                                    muted
                                    playsInline
                                />
                            ) : (
                                <div className="text-muted/30 font-heading text-4xl font-bold uppercase tracking-tighter mix-blend-overlay flex flex-col items-center">
                                    <Play className="w-16 h-16 mb-4 opacity-50" />
                                    <span>Awaiting Generation</span>
                                </div>
                            )}
                        </div>

                        {/* Overlay Controls (Decorative) */}
                        <div className="absolute bottom-6 right-6 z-20 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                            <button className="p-3 rounded-full bg-black/60 backdrop-blur-md border border-white/10 text-cream hover:bg-white/10 transition-colors">
                                <Maximize2 className="w-5 h-5" />
                            </button>
                        </div>
                    </motion.div>
                </div>
            </div>

            {/* Background elements */}
            <div className="absolute top-0 right-0 w-1/2 h-full bg-gradient-to-l from-white/[0.02] to-transparent pointer-events-none"></div>
        </section>
    );
};

export default Comparison;
