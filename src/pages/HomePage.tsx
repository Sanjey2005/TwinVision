import Navbar from '../components/Navbar';
import Hero from '../components/Hero';
import Pipeline from '../components/Pipeline';
import Comparison from '../components/Comparison';
import Metrics from '../components/Metrics';
import type { ComparisonPayload } from '../api/client';

interface HomePageProps {
    onGenerate?: (prompt: string) => void;
    comparisonData?: ComparisonPayload | null;
}

const HomePage = ({ onGenerate, comparisonData }: HomePageProps) => {
    return (
        <div className="bg-background text-cream">
            <Navbar />

            <main className="min-h-[200vh]">
                <div className="px-6 md:px-12">
                    <Hero onGenerate={onGenerate} />
                </div>

                <Pipeline />

                {/* Pass URLs safely if results are present */}
                <Comparison
                    winner={comparisonData?.results?.winner || 'sd35'} // Keep visual fallback purely for local rendering if no API
                    fluxVideoUrl={comparisonData?.results?.flux?.video_url}
                    sd35VideoUrl={comparisonData?.results?.sd35?.video_url}
                />

                <Metrics metricsData={comparisonData?.results} />
            </main>
        </div>
    );
};

export default HomePage;
