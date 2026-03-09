import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { motion } from 'framer-motion';
import { Activity, Eye, Zap, Layers } from 'lucide-react';
import type { ComparisonPayload } from '../api/client';

const performanceData = [
    { metric: 'CLIP Sim', flux: 0.32, sd35: 0.28, higherIsBetter: true },
    { metric: 'BRISQUE', flux: 25.4, sd35: 31.2, higherIsBetter: false },
    { metric: 'NIQE', flux: 4.1, sd35: 5.8, higherIsBetter: false },
    { metric: 'SSIM', flux: 0.88, sd35: 0.81, higherIsBetter: true },
    { metric: 'LPIPS', flux: 0.12, sd35: 0.18, higherIsBetter: false }
];

interface MetricsProps {
    metricsData?: ComparisonPayload['results'];
}

const StatCard = ({ title, valueFlux, valueSD, higherIsBetter, icon: Icon, delay }: any) => {
    const fluxWins = higherIsBetter ? valueFlux > valueSD : valueFlux < valueSD;

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ delay, duration: 0.5 }}
            className="glass-card rounded-3xl p-6 relative overflow-hidden group"
        >
            <div className="flex justify-between items-start mb-8">
                <div className="p-3 bg-white/5 rounded-xl text-muted group-hover:text-cream transition-colors">
                    <Icon className="w-6 h-6" />
                </div>
                <span className="text-xs font-heading tracking-widest text-muted/50 uppercase">
                    {higherIsBetter ? 'Higher is Better' : 'Lower is Better'}
                </span>
            </div>

            <div>
                <h4 className="text-muted font-body text-sm tracking-wide uppercase mb-4">{title}</h4>
                <div className="flex justify-between items-end border-b border-white/5 pb-4 mb-4">
                    <span className="text-xs font-heading tracking-widest text-muted">FLUX.1 [S]</span>
                    <span className={`text-4xl font-heading font-black tracking-tighter ${fluxWins ? 'text-primary-neon text-glow' : 'text-white'}`}>
                        {valueFlux}
                    </span>
                </div>
                <div className="flex justify-between items-end">
                    <span className="text-xs font-heading tracking-widest text-muted">SD 3.5</span>
                    <span className={`text-3xl font-heading font-bold tracking-tighter ${!fluxWins ? 'text-primary-blue blue-glow' : 'text-white/50'}`}>
                        {valueSD}
                    </span>
                </div>
            </div>

            {/* Subtle background glow for winner */}
            <div className={`absolute -right-10 -bottom-10 w-32 h-32 rounded-full blur-[60px] pointer-events-none transition-opacity duration-500 opacity-0 group-hover:opacity-100 ${fluxWins ? 'bg-primary-neon/20' : 'bg-primary-blue/20'}`}></div>
        </motion.div>
    );
};

const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
        return (
            <div className="bg-black/90 border border-white/10 p-4 rounded-xl backdrop-blur-md shadow-2xl">
                <p className="font-heading font-bold text-cream mb-2 uppercase tracking-wide">{label}</p>
                {payload.map((entry: any, index: number) => (
                    <div key={index} className="flex items-center space-x-4 mb-1">
                        <span
                            className="w-3 h-3 rounded-full"
                            style={{ backgroundColor: entry.color }}
                        ></span>
                        <span className="text-muted text-sm capitalize">{entry.name}:</span>
                        <span className="text-cream font-bold font-heading">{entry.value}</span>
                    </div>
                ))}
            </div>
        );
    }
    return null;
};

const Metrics = ({ metricsData }: MetricsProps) => {
    const activeData = metricsData ? [
        { metric: 'CLIP Sim', flux: metricsData.flux.metrics.clip_sim || 0, sd35: metricsData.sd35.metrics.clip_sim || 0, higherIsBetter: true },
        { metric: 'BRISQUE', flux: metricsData.flux.metrics.brisque || 0, sd35: metricsData.sd35.metrics.brisque || 0, higherIsBetter: false },
        { metric: 'NIQE', flux: metricsData.flux.metrics.niqe || 0, sd35: metricsData.sd35.metrics.niqe || 0, higherIsBetter: false },
        { metric: 'SSIM', flux: metricsData.flux.metrics.ssim || 0, sd35: metricsData.sd35.metrics.ssim || 0, higherIsBetter: true },
        { metric: 'LPIPS', flux: metricsData.flux.metrics.lpips || 0, sd35: metricsData.sd35.metrics.lpips || 0, higherIsBetter: false }
    ] : performanceData;

    return (
        <section className="py-24 relative overflow-hidden" id="metrics">
            <div className="max-w-7xl mx-auto px-6 md:px-12 relative z-10">
                <div className="mb-16">
                    <h2 className="text-4xl md:text-6xl font-heading font-bold text-cream uppercase tracking-tighter">
                        Automated <span className="text-primary-neon text-glow drop-shadow-[0_0_15px_rgba(204,255,0,0.5)]">Metrics</span>
                    </h2>
                    <p className="text-muted font-body mt-4 max-w-2xl text-lg">
                        Quantitative analysis of spatial-temporal consistencies, aesthetic scores, and generation fidelity.
                    </p>
                </div>

                {/* Bento Box Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">

                    {/* Large Chart Container */}
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        whileInView={{ opacity: 1, scale: 1 }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.7 }}
                        className="col-span-1 md:col-span-2 lg:col-span-4 glass-card rounded-3xl p-6 md:p-10"
                    >
                        <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 gap-4">
                            <div>
                                <h3 className="text-2xl font-heading font-bold uppercase tracking-tighter flex items-center">
                                    <Activity className="w-6 h-6 mr-3 text-primary-neon" />
                                    Performance Overview
                                </h3>
                                <p className="text-muted text-sm mt-1">Aggregated scoring across standard evaluation dimensions.</p>
                            </div>
                            <div className="flex space-x-4 text-xs font-heading font-bold tracking-widest uppercase">
                                <div className="flex items-center">
                                    <span className="w-3 h-3 rounded-full bg-primary-neon mr-2"></span>
                                    FLUX.1 [S]
                                </div>
                                <div className="flex items-center">
                                    <span className="w-3 h-3 rounded-full bg-primary-blue mr-2"></span>
                                    SD 3.5
                                </div>
                            </div>
                        </div>

                        <div className="h-[400px] w-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart
                                    data={activeData}
                                    margin={{ top: 20, right: 30, left: 0, bottom: 5 }}
                                    barGap={8}
                                >
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                                    <XAxis
                                        dataKey="metric"
                                        stroke="#666"
                                        tick={{ fill: '#888', fontSize: 12, fontFamily: 'monospace' }}
                                        axisLine={false}
                                        tickLine={false}
                                        dy={10}
                                    />
                                    <YAxis
                                        stroke="#666"
                                        tick={{ fill: '#888', fontSize: 12, fontFamily: 'monospace' }}
                                        axisLine={false}
                                        tickLine={false}
                                    />
                                    <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.02)' }} />
                                    <Bar dataKey="flux" name="FLUX" fill="#ccff00" radius={[4, 4, 0, 0]} maxBarSize={50} />
                                    <Bar dataKey="sd35" name="SD3.5" fill="#00f0ff" radius={[4, 4, 0, 0]} maxBarSize={50} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </motion.div>

                    {/* Stat Cards Row */}
                    <StatCard
                        title="CLIP Similarity"
                        valueFlux={activeData[0].flux}
                        valueSD={activeData[0].sd35}
                        higherIsBetter={true}
                        icon={Zap}
                        delay={0.1}
                    />
                    <StatCard
                        title="BRISQUE Score"
                        valueFlux={activeData[1].flux}
                        valueSD={activeData[1].sd35}
                        higherIsBetter={false}
                        icon={Eye}
                        delay={0.2}
                    />
                    <StatCard
                        title="NIQE Score"
                        valueFlux={activeData[2].flux}
                        valueSD={activeData[2].sd35}
                        higherIsBetter={false}
                        icon={Eye}
                        delay={0.3}
                    />
                    <StatCard
                        title="Temporal SSIM"
                        valueFlux={activeData[3].flux}
                        valueSD={activeData[3].sd35}
                        higherIsBetter={true}
                        icon={Layers}
                        delay={0.4}
                    />

                </div>
            </div>

            {/* Background elements */}
            <div className="absolute top-1/2 left-0 w-96 h-96 bg-primary-neon/5 rounded-full blur-[150px] pointer-events-none -z-10"></div>
        </section>
    );
};

export default Metrics;
