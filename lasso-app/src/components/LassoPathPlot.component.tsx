'use client';

import React, { useEffect, useState } from 'react';
import Plot from 'react-plotly.js';

type LassoPathPlotInputData = {
  coefficients: number[][];
  alphas: number[];
  feature_names: string[];
};

type PlotData = {
  x: number[];
  y: number[];
  mode: string;
  name: string;
};

const LassoPathPlot = () => {
  const [plotData, setPlotData] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch('http://localhost:8000/lasso-path-data');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setPlotData(data);
      } catch (error) {
        console.error('Error fetching the plot data:', error);
      }
    };

    fetchData();
  }, []);

  const buildPlotData = (data: LassoPathPlotInputData): PlotData[] => {
    const traces = data.coefficients.map((coeff, index) => ({
      x: data.alphas,
      y: coeff,
      mode: 'lines',
      name: data.feature_names[index],
    }));
    return traces;
  };

  return (
    <div>
      {plotData ? (
        <Plot
          data={buildPlotData(plotData)}
          layout={{ title: 'Lasso Path Plot' }}
        />
      ) : (
        <p>Loading...</p>
      )}
    </div>
  );
};

export default LassoPathPlot;
