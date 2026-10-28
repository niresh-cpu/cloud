import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';

// Pages
import Home from './pages/Home';
import DiseaseDetection from './pages/DiseaseDetection';
import DiseaseInformation from './pages/DiseaseInformation';
import SupportedPlants from './pages/SupportedPlants';
import ModelPerformance from './pages/ModelPerformance';
import AboutModel from './pages/AboutModel';

function App() {
  return (
    <Router>
      <div className="flex flex-col min-h-screen bg-slate-50 text-slate-800">
        <Navbar />
        <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/detect" element={<DiseaseDetection />} />
            <Route path="/diseases" element={<DiseaseInformation />} />
            <Route path="/crops" element={<SupportedPlants />} />
            <Route path="/performance" element={<ModelPerformance />} />
            <Route path="/about" element={<AboutModel />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
}

export default App;
