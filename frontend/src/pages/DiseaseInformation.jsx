import React, { useState, useEffect } from 'react';
import {
  BookOpen,
  Search,
  Filter,
  CheckCircle2,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  Shield,
  Sprout
} from 'lucide-react';
import { getAllDiseases } from '../services/api';

const DiseaseInformation = () => {
  const [diseases, setDiseases] = useState({});
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCrop, setSelectedCrop] = useState('ALL');
  const [filterType, setFilterType] = useState('ALL'); // ALL, HEALTHY, DISEASED
  const [expandedId, setExpandedId] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getAllDiseases()
      .then((data) => {
        setDiseases(data.diseases || {});
        setLoading(false);
      })
      .catch((err) => {
        console.error("Error loading disease knowledge base:", err);
        setLoading(false);
      });
  }, []);

  const diseaseEntries = Object.entries(diseases).map(([id, info]) => ({
    id,
    ...info,
  }));

  const availableCrops = ['ALL', ...Array.from(new Set(diseaseEntries.map((d) => d.crop))).sort()];

  const filteredEntries = diseaseEntries.filter((item) => {
    const matchesSearch =
      item.disease_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.crop.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.description.toLowerCase().includes(searchTerm.toLowerCase());

    const matchesCrop = selectedCrop === 'ALL' || item.crop === selectedCrop;
    const matchesType =
      filterType === 'ALL' ||
      (filterType === 'HEALTHY' && item.is_healthy) ||
      (filterType === 'DISEASED' && !item.is_healthy);

    return matchesSearch && matchesCrop && matchesType;
  });

  const toggleExpand = (id) => {
    setExpandedId(expandedId === id ? null : id);
  };

  return (
    <div className="space-y-8 py-4 pb-16 max-w-6xl mx-auto">
      {/* Header */}
      <div className="text-center space-y-2">
        <span className="text-xs font-bold text-emerald-600 uppercase tracking-widest">
          Agronomic Knowledge Base
        </span>
        <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">
          Plant Disease Reference Encyclopedia
        </h1>
        <p className="text-sm text-slate-600 max-w-2xl mx-auto">
          Scientific botanical reference of crop pathogens, distinguishing foliar symptoms,
          preventative cultural practices, and Integrated Pest Management (IPM) protocols.
        </p>
      </div>

      {/* Filter and Search Bar */}
      <div className="bg-white rounded-3xl p-5 border border-slate-200 shadow-sm space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-12 gap-3">
          {/* Search Box */}
          <div className="md:col-span-6 relative">
            <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-3.5" />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Search by disease, pathogen, or symptoms..."
              className="w-full pl-10 pr-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 transition"
            />
          </div>

          {/* Crop Dropdown */}
          <div className="md:col-span-3">
            <select
              value={selectedCrop}
              onChange={(e) => setSelectedCrop(e.target.value)}
              className="w-full px-3.5 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 font-medium text-slate-700"
            >
              {availableCrops.map((c) => (
                <option key={c} value={c}>
                  {c === 'ALL' ? 'All Crop Species' : `Crop: ${c}`}
                </option>
              ))}
            </select>
          </div>

          {/* Type Toggle */}
          <div className="md:col-span-3 flex rounded-xl bg-slate-100 p-1 border border-slate-200 text-xs font-bold">
            <button
              onClick={() => setFilterType('ALL')}
              className={`flex-1 py-1.5 rounded-lg transition ${
                filterType === 'ALL' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600'
              }`}
            >
              All ({diseaseEntries.length})
            </button>
            <button
              onClick={() => setFilterType('DISEASED')}
              className={`flex-1 py-1.5 rounded-lg transition ${
                filterType === 'DISEASED' ? 'bg-white text-rose-700 shadow-xs' : 'text-slate-600'
              }`}
            >
              Diseases
            </button>
            <button
              onClick={() => setFilterType('HEALTHY')}
              className={`flex-1 py-1.5 rounded-lg transition ${
                filterType === 'HEALTHY' ? 'bg-white text-emerald-700 shadow-xs' : 'text-slate-600'
              }`}
            >
              Healthy
            </button>
          </div>
        </div>

        <div className="flex justify-between items-center text-xs text-slate-500 px-1 pt-1">
          <span>Showing {filteredEntries.length} verified categories</span>
          <span>Integrated with ResNet-50 class mapping</span>
        </div>
      </div>

      {/* Disease Cards List */}
      {loading ? (
        <div className="text-center py-12 text-slate-500 text-sm">
          Loading agricultural encyclopedia...
        </div>
      ) : filteredEntries.length === 0 ? (
        <div className="bg-white rounded-3xl p-12 text-center border border-slate-200 text-slate-500 space-y-2">
          <BookOpen className="w-10 h-10 text-slate-300 mx-auto" />
          <p className="font-bold text-slate-800">No disease records matched your filter criteria.</p>
          <p className="text-xs">Try clearing the search term or resetting the crop selector.</p>
        </div>
      ) : (
        <div className="space-y-4">
          {filteredEntries.map((item) => {
            const isExpanded = expandedId === item.id;
            return (
              <div
                key={item.id}
                className="bg-white rounded-2xl border border-slate-200/90 shadow-sm overflow-hidden hover:border-slate-300 transition"
              >
                {/* Collapsible Header */}
                <div
                  onClick={() => toggleExpand(item.id)}
                  className="p-5 flex flex-wrap items-center justify-between gap-4 cursor-pointer hover:bg-slate-50/50 transition"
                >
                  <div className="flex items-center gap-3.5">
                    <div
                      className={`w-10 h-10 rounded-xl flex items-center justify-center shrink-0 ${
                        item.is_healthy
                          ? 'bg-emerald-100 text-emerald-700'
                          : 'bg-rose-100 text-rose-700'
                      }`}
                    >
                      {item.is_healthy ? (
                        <CheckCircle2 className="w-5 h-5" />
                      ) : (
                        <AlertTriangle className="w-5 h-5" />
                      )}
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-bold px-2 py-0.5 rounded-md bg-slate-100 text-slate-700 border border-slate-200">
                          {item.crop}
                        </span>
                        <h3 className="text-base font-bold text-slate-900">
                          {item.disease_name}
                        </h3>
                      </div>
                      <p className="text-xs text-slate-500 mt-0.5">
                        Pathogen: <span className="font-semibold text-slate-700">{item.pathogen_type}</span>
                        {item.severity && !item.is_healthy && (
                          <span className="ml-2 font-semibold text-rose-600">
                            · Severity: {item.severity}
                          </span>
                        )}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-3">
                    <span
                      className={`text-xs font-bold px-2.5 py-1 rounded-full ${
                        item.is_healthy
                          ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                          : 'bg-rose-50 text-rose-700 border border-rose-200'
                      }`}
                    >
                      {item.is_healthy ? 'Healthy' : 'Diseased'}
                    </span>
                    <button className="text-slate-400 hover:text-slate-600 p-1">
                      {isExpanded ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
                    </button>
                  </div>
                </div>

                {/* Expanded Details */}
                {isExpanded && (
                  <div className="px-5 pb-6 pt-2 border-t border-slate-100 bg-slate-50/40 space-y-5">
                    <div>
                      <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-1">
                        Agronomic Description
                      </h4>
                      <p className="text-sm text-slate-700 leading-relaxed">
                        {item.description}
                      </p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      {/* Symptoms */}
                      <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-2xs space-y-2">
                        <h5 className="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center gap-1.5">
                          <AlertTriangle className="w-3.5 h-3.5 text-amber-500" />
                          Diagnostic Symptoms
                        </h5>
                        <ul className="space-y-1.5">
                          {item.symptoms.map((s, idx) => (
                            <li key={idx} className="text-xs text-slate-600 flex items-start gap-1.5">
                              <span className="w-1 h-1 rounded-full bg-slate-400 mt-1.5 shrink-0" />
                              <span>{s}</span>
                            </li>
                          ))}
                        </ul>
                      </div>

                      {/* Cultural Prevention */}
                      <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-2xs space-y-2">
                        <h5 className="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center gap-1.5">
                          <Shield className="w-3.5 h-3.5 text-emerald-600" />
                          Cultural Prevention
                        </h5>
                        <ul className="space-y-1.5">
                          {item.prevention.map((p, idx) => (
                            <li key={idx} className="text-xs text-slate-600 flex items-start gap-1.5">
                              <span className="w-1 h-1 rounded-full bg-emerald-500 mt-1.5 shrink-0" />
                              <span>{p}</span>
                            </li>
                          ))}
                        </ul>
                      </div>

                      {/* IPM Management */}
                      <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-2xs space-y-2">
                        <h5 className="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center gap-1.5">
                          <Sprout className="w-3.5 h-3.5 text-teal-600" />
                          IPM Management
                        </h5>
                        <ul className="space-y-1.5">
                          {item.management.map((m, idx) => (
                            <li key={idx} className="text-xs text-slate-600 flex items-start gap-1.5">
                              <span className="w-1 h-1 rounded-full bg-teal-500 mt-1.5 shrink-0" />
                              <span>{m}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default DiseaseInformation;
