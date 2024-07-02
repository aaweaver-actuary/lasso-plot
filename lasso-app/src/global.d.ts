// Global type declarations for plotly.js and react-plotly.js:

declare module 'plotly.js' {
    export = Plotly;
}

declare module 'react-plotly.js/factory' {
    import Plotly from 'plotly.js';
    const createPlotlyComponent: (Plotly: any) => any;
    export default createPlotlyComponent;
}