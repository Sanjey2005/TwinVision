import { useRef } from 'react';
import gsap from 'gsap';
import { useGSAP } from '@gsap/react';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { TerminalSquare, ImagePlus, Film, BarChart3 } from 'lucide-react';

gsap.registerPlugin(ScrollTrigger);

const Pipeline = () => {
    const containerRef = useRef<HTMLDivElement>(null);
    const scrollWrapperRef = useRef<HTMLDivElement>(null);

    useGSAP(() => {
        let mm = gsap.matchMedia();

        mm.add("(min-width: 768px)", () => {
            // 1. Get the actual scroll distance (total width of all cards minus viewport width)
            const scrollWidth = scrollWrapperRef.current?.scrollWidth || 0;
            const windowWidth = window.innerWidth;

            // Only animate if the content is wider than the screen
            if (scrollWidth > windowWidth) {
                const xDistance = -(scrollWidth - windowWidth + 100); // 100px extra padding

                // 2. Setup the GSAP Pin and Horizontal Translate
                gsap.to(scrollWrapperRef.current, {
                    x: xDistance,
                    ease: "none", // important for consistent scroll speed
                    scrollTrigger: {
                        trigger: containerRef.current,
                        start: "top top", // When the top of container hits top of viewport
                        end: `+=${scrollWidth}`, // Scroll distance equals content width
                        pin: true, // Lock the container in place
                        scrub: 1, // Smooth scrubbing (1 second catchup time)
                        anticipatePin: 1,
                    }
                });
            }
        });

        return () => mm.revert();
    }, { scope: containerRef }); // Scope animations to this component

    const cards = [
        {
            id: "01",
            title: "PROMPT  INPUT",
            desc: "System receives the creative direction. Normalizes text and evaluates context parameters for both Generation engines.",
            icon: <TerminalSquare className="w-10 h-10 mb-6 text-primary-neon" />,
            delay: "0.1s"
        },
        {
            id: "02",
            title: "IMAGE  GEN (FLUX VS SD3.5)",
            desc: "Simultaneous execution of prompt against Black Forest Labs' FLUX.1 [schnell] and Stability AI's SD3.5 for direct A/B visual comparison.",
            icon: <ImagePlus className="w-10 h-10 mb-6 text-primary-neon" />,
            delay: "0.2s"
        },
        {
            id: "03",
            title: "FFMPEG  COMPILATION",
            desc: "High-speed temporal alignment. Stitches latent frame outputs into synchronized, side-by-side MP4 video layouts.",
            icon: <Film className="w-10 h-10 mb-6 text-primary-neon" />,
            delay: "0.3s"
        },
        {
            id: "04",
            title: "AUTOMATED  METRICS",
            desc: "Continuous evaluation module calculates objective aesthetic scores globally: CLIP similarity, BRISQUE, and Niqe quality markers.",
            icon: <BarChart3 className="w-10 h-10 mb-6 text-primary-neon" />,
            delay: "0.4s"
        }
    ];

    return (
        <section
            ref={containerRef}
            className="relative min-h-screen bg-background flex flex-col items-start justify-center overflow-hidden border-t border-white/[0.05] pt-24"
            id="pipeline"
        >
            <div className="relative z-10 w-full px-6 md:px-24 mb-10">
                <h2 className="text-4xl md:text-5xl font-heading font-bold text-muted uppercase tracking-tighter">
                    The <span className="text-cream">Pipeline</span>
                </h2>
                <div className="h-[1px] w-1/4 bg-primary-neon mt-4"></div>
            </div>

            {/* The scrolling wrapper */}
            <div
                ref={scrollWrapperRef}
                className="flex flex-col md:flex-row gap-8 px-6 md:px-24 mt-10 md:mt-20 pb-20 md:pb-0 md:w-max"
            >
                {cards.map((card) => (
                    <div
                        key={card.id}
                        className="group relative w-full sm:w-[85vw] md:w-[450px] h-auto md:h-[500px] aspect-[4/5] md:aspect-auto glass-card rounded-3xl p-8 md:p-10 flex flex-col justify-end overflow-hidden transition-all duration-500 hover:border-primary-neon/30 hover:bg-white/[0.05]"
                    >
                        {/* Massive Background Number */}
                        <div className="absolute top-4 -right-4 font-heading font-black text-[12rem] text-white/[0.02] group-hover:text-primary-neon/[0.05] transition-colors duration-500 leading-none pointer-events-none select-none">
                            {card.id}
                        </div>

                        {/* Wireframe / Grid Effect */}
                        <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.03)_1px,transparent_1px)] bg-[size:40px_40px] pointer-events-none opacity-20 group-hover:opacity-40 transition-opacity duration-500"></div>

                        {/* Content */}
                        <div className="relative z-10">
                            {card.icon}
                            <div className="text-primary-neon font-body text-xs tracking-[0.35em] mb-3 font-bold">
                                STEP {card.id}
                            </div>
                            <h3 className="text-3xl font-heading font-bold text-cream mb-4 uppercase tracking-wider">
                                {card.title}
                            </h3>
                            <p className="text-muted font-body leading-relaxed text-base">
                                {card.desc}
                            </p>
                        </div>
                    </div>
                ))}
            </div>

            {/* Background ambient glow */}
            <div className="absolute bottom-[-20%] left-[20%] w-[500px] h-[500px] bg-primary-blue/5 rounded-full blur-[150px] pointer-events-none -z-10"></div>
        </section>
    );
};

export default Pipeline;
