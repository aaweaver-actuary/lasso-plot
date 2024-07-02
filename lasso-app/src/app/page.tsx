import LassoPathPlot from '@/components/LassoPathPlot.component';

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <h1 className="text-6xl font-bold">PLOTLY</h1>
      <LassoPathPlot />
    </main>
  );
}
