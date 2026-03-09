import { useState } from 'react';
import { motion, useScroll, useMotionValueEvent } from 'framer-motion';
import { Github } from 'lucide-react';

const navLinks = [
    { name: '[ THE PIPELINE ]', href: '#pipeline' },
    { name: '[ COMPARISON ]', href: '#comparison' },
    { name: '[ METRICS ]', href: '#metrics' },
];

const Navbar = () => {
    const { scrollY } = useScroll();
    const [scrolled, setScrolled] = useState(false);

    useMotionValueEvent(scrollY, "change", (latest) => {
        if (latest > 50) {
            setScrolled(true);
        } else {
            setScrolled(false);
        }
    });

    return (
        <motion.nav
            initial={{ y: -100 }}
            animate={{ y: 0 }}
            transition={{ duration: 0.8, ease: [0.76, 0, 0.24, 1], delay: 0.5 }} // delayed to sync with preloader reveal
            className={`fixed top-0 left-0 right-0 z-40 transition-all duration-500 ease-in-out ${scrolled ? 'py-4 glass-card border-b border-white/[0.08]' : 'py-8 bg-transparent border-b border-transparent'
                }`}
        >
            <div className="max-w-7xl mx-auto px-6 md:px-12 flex items-center justify-between">
                {/* Left: Logo */}
                <a
                    href="#"
                    className="font-heading font-bold text-2xl tracking-tighter text-cream hover:text-primary-neon transition-colors duration-300"
                >
                    TWINVISION.
                </a>

                {/* Center: Links */}
                <div className="hidden lg:flex items-center space-x-12">
                    {navLinks.map((link) => (
                        <a
                            key={link.name}
                            href={link.href}
                            className="relative font-body text-sm font-medium tracking-widest text-muted hover:text-cream transition-colors duration-300 group overflow-hidden"
                        >
                            <span>{link.name}</span>
                            {/* Hover Underline / Strikethrough Effect */}
                            <span className="absolute left-0 bottom-0 w-full h-[1px] bg-primary-neon -translate-x-[101%] group-hover:translate-x-0 transition-transform duration-300 ease-out"></span>
                        </a>
                    ))}
                </div>

                {/* Right: GitHub */}
                <a
                    href="https://github.com"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center space-x-3 text-cream group"
                >
                    <span className="hidden sm:inline-block font-body text-sm font-bold tracking-widest uppercase text-muted group-hover:text-cream transition-colors duration-300">
                        Source
                    </span>
                    <div className="p-2.5 rounded-full bg-white/[0.03] border border-white/[0.1] group-hover:border-primary-neon/50 group-hover:neon-glow group-hover:text-primary-neon transition-all duration-300">
                        <Github className="w-5 h-5" />
                    </div>
                </a>
            </div>
        </motion.nav>
    );
};

export default Navbar;
