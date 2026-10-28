import React, { useState, useEffect } from 'react';
import { Sprout, CheckCircle2, AlertTriangle, ArrowRight, ShieldCheck, Layers } from 'lucide-react';
import { Link } from 'react-router-dom';
import { getSupportedCrops } from '../services/api';

const SupportedPlants = () => {
  const [cropsData, setCropsData] = useState([]);
  const [selectedCrop, setSelectedCrop] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getSupportedCrops()
      .then((data) => {
        const crops = data.crops || [];
        setCropsData(crops);
        if (crops.length > 0) {
          setSelectedCrop(crops[0]);
        }
        setLoading(false);
      })
      .catch((err) => {
        console.error("Error fetching supported crops:", err);
        setLoading(false);
      });
  }, []);

  return (
    <div className="space-y-8 py-4 pb-16 max-w-6xl mx-auto">
      {/* Header */}
      <div className="text-center space-y-2">
        <span className="text-xs font-bold text-emerald-600 uppercase tracking-widest">
          Dynamic Model Inventory
        </span>
        <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">
          Supported Agricultural Crop Species
        </h1>
        <p className="text-sm text-slate-600 max-w-2xl mx-auto">
          Every crop and disease listed below has been verified in the training dataset and
          evaluated in the ResNet-50 transfer learning classifier.
        </p>
      </div>

      {loading ? (
        <div className="text-center py-16 text-slate-500 text-sm">
          Loading crop registry from backend...
        </div>
      ) : cropsData.length === 0 ? (
        <div className="bg-white rounded-3xl p-12 text-center border border-slate-200">
          <p className="text-slate-600">No supported crop data returned by backend API.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Crop Selection Column */}
          <div className="lg:col-span-5 space-y-3">
            <h2 className="text-xs font-bold text-slate-500 uppercase tracking-wider px-1">
              Select Crop Species ({cropsData.length} available)
            </h2>
            <div className="space-y-2">
              {cropsData.map((c) => {
                const isSelected = selectedCrop?.crop === c.crop;
                const healthyCount = c.diseases.filter((d) => d.is_healthy).length;
                const diseasedCount = c.diseases.length - healthyCount;

                return (
                  <button
                    key={c.crop}
                    onClick={() => setSelectedCrop(c)}
                    className={`w-full text-left p-4 rounded-2xl border transition flex items-center justify-between ${
                      isSelected
                        ? 'bg-emerald-50 border-emerald-400 shadow-sm'
                        : 'bg-white border-slate-200 hover:border-slate-300 hover:bg-slate-50/70'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <div
                        className={`w-10 h-10 rounded-xl flex items-center justify-center font-bold text-sm ${
                          isSelected
                            ? 'bg-emerald-600 text-white'
                            : 'bg-slate-100 text-slate-700'
                        }`}
                      >
                        🌱
                      </div>
                      <div>
                        <h3 className="font-bold text-slate-900 text-sm">{c.crop}</h3>
                        <p className="text-xs text-slate-500">
                          {c.total_categories} classes ({diseasedCount} pathologies, {healthyCount} healthy)
                        </p>
                      </div>
                    </div>
                    <ArrowRight
                      className={`w-4 h-4 transition ${
                        isSelected ? 'text-emerald-700 translate-x-1' : 'text-slate-300'
                      }`}
                    />
                  </button>
                );
              })}
            </div>
          </div>

          {/* Detailed Categories Column */}
          <div className="lg:col-span-7">
            {selectedCrop ? (
              <div className="bg-white rounded-3xl p-6 border border-slate-200 shadow-sm space-y-6">
                <div className="flex justify-between items-start border-b border-slate-100 pb-4">
                  <div>
                    <span className="text-xs font-bold text-emerald-600 uppercase tracking-widest">
                      Crop Profile
                    </span>
                    <h2 className="text-2xl font-black text-slate-900 mt-1">
                      {selectedCrop.crop}
                    </h2>
                    <p className="text-xs text-slate-500 mt-0.5">
                      {selectedCrop.diseases.length} total classes mapped in model output softmax layer
                    </p>
                  </div>
                  <Link
                    to="/detect"
                    className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl text-xs font-bold transition shadow-xs flex items-center gap-1.5"
                  >
                    Diagnose {selectedCrop.crop} →
                  </Link>
                </div>

                <div className="space-y-3">
                  <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider">
                    Associated Pathological & Healthy Categories
                  </h4>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {selectedCrop.diseases.map((d, idx) => (
                      <div
                        key={idx}
                        className={`p-3.5 rounded-xl border flex items-start gap-2.5 ${
                          d.is_healthy
                            ? 'bg-emerald-50/50 border-emerald-200 text-emerald-950'
                            : 'bg-slate-50 border-slate-200 text-slate-800'
                        }`}
                      >
                        {d.is_healthy ? (
                          <CheckCircle2 className="w-4 h-4 text-emerald-600 mt-0.5 shrink-0" />
                        ) : (
                          <AlertTriangle className="w-4 h-4 text-rose-500 mt-0.5 shrink-0" />
                        )}
                        <div>
                          <p className="text-xs font-bold leading-tight">{d.disease}</p>
                          <p className="text-[11px] text-slate-500 mt-0.5 font-mono">
                            {d.class_id}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Academic Context Notice */}
                <div className="bg-slate-50 border border-slate-200 rounded-2xl p-4 text-xs text-slate-600 flex items-start gap-2.5">
                  <ShieldCheck className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
                  <p>
                    Classes are synchronized directly with <code className="text-emerald-700 font-mono">models/class_names.json</code>.
                    Unsupported crops or unverified disease labels are strictly excluded to preserve scientific integrity.
                  </p>
                </div>
              </div>
            ) : null}
          </div>
        </div>
      )}
    </div>
  );
};

export default SupportedPlants;
