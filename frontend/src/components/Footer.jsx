import React from 'react';
import { Leaf, Shield, Cpu, ExternalLink } from 'lucide-react';

const Footer = () => {
  return (
    <footer className="bg-white border-t border-slate-200 mt-auto py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row justify-between items-center gap-4 text-center md:text-left">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-emerald-600 flex items-center justify-center text-white">
              <Leaf className="w-5 h-5" />
            </div>
            <div>
              <p className="text-sm font-bold text-slate-800">
                AgriVision AI · ResNet-50 Multi-Crop Disease Classifier
              </p>
              <p className="text-xs text-slate-500">
                IA-3 Deep Learning Skill-Based Evaluation · 25 Marks Comprehensive Project
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center justify-center gap-2 text-xs text-slate-600 font-medium">
            <span className="px-2.5 py-1 bg-slate-100 rounded-md border border-slate-200">ResNet-50</span>
            <span className="px-2.5 py-1 bg-slate-100 rounded-md border border-slate-200">TensorFlow 2.x</span>
            <span className="px-2.5 py-1 bg-slate-100 rounded-md border border-slate-200">FastAPI</span>
            <span className="px-2.5 py-1 bg-slate-100 rounded-md border border-slate-200">React + Vite</span>
            <span className="px-2.5 py-1 bg-emerald-50 text-emerald-700 font-semibold rounded-md border border-emerald-200">
              OOD Uncertainty Guard
            </span>
          </div>
        </div>

        <div className="mt-6 pt-4 border-t border-slate-100 flex flex-col sm:flex-row justify-between items-center text-xs text-slate-400 gap-2">
          <p>© 2026 Deep Learning Project. Developed for academic evaluation & agricultural research.</p>
          <p className="flex items-center gap-1 text-slate-500">
            <span>Powered by PlantVillage & Rice Leaf Pathology Datasets</span>
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
