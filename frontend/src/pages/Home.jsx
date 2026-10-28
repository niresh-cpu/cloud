import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import {
  Leaf,
  ShieldCheck,
  Cpu,
  Layers,
  ArrowRight,
  CheckCircle2,
  AlertTriangle,
  Zap,
  BookOpen,
  BarChart3,
  Search
} from 'lucide-react';
import { getSupportedCrops, getSupportedClasses, healthCheck } from '../services/api';

const Home = () => {
  const [stats, setStats] = useState({
    cropCount: 0,
    classCount: 0,
    cropsList: [],
    loading: true,
  });

  useEffect(() => {
    Promise.all([getSupportedCrops(), getSupportedClasses(), healthCheck()])
      .then(([cropsData, classesData]) => {
        setStats({
          cropCount: cropsData.total_crops || 0,
          classCount: classesData.total || 0,
          cropsList: (cropsData.crops || []).map((c) => c.crop),
          loading: false,
        });
      })
      .catch((err) => {
        console.error("Error loading home stats:", err);
        setStats((prev) => ({ ...prev, loading: false }));
      });
  }, []);

  const features = [
    {
      icon: <Layers className="w-6 h-6 text-emerald-600" />,
      title: "ResNet-50 Transfer Learning",
      description: "Harnesses deep residual connections with 50 layers to accurately extract complex foliar lesion and necrosis textures.",
    },
    {
      icon: <ShieldCheck className="w-6 h-6 text-emerald-600" />,
      title: "Leaf Validation Guard",
      description: "Dedicated validation pipeline that rejects cars, buildings, animals, and non-plant inputs before classification.",
    },
    {
      icon: <AlertTriangle className="w-6 h-6 text-emerald-600" />,
      title: "Uncertainty Quantification",
      description: "Calculates prediction entropy and confidence margins to alert farmers when an image is ambiguous or out-of-distribution.",
    },
    {
      icon: <Zap className="w-6 h-6 text-emerald-600" />,
      title: "Real-Time Fast Inference",
      description: "Optimized model caching delivers instant diagnostic predictions with top-3 candidate confidence rankings.",
    },
    {
      icon: <BookOpen className="w-6 h-6 text-emerald-600" />,
      title: "Integrated IPM Advisory",
      description: "Provides verified botanical descriptions, symptoms, cultural prevention, and integrated pest management strategies.",
    },
    {
      icon: <BarChart3 className="w-6 h-6 text-emerald-600" />,
      title: "Full Evaluation Transparency",
      description: "Live performance dashboard reporting actual test set accuracy, precision, recall, confusion matrix, and training curves.",
    },
  ];

  const workflowSteps = [
    {
      num: "01",
      title: "Input Acquisition",
      desc: "Upload leaf photographs across multiple formats (JPG, PNG, WEBP, BMP, TIFF) or capture live via camera.",
    },
    {
      num: "02",
      title: "Foliage Verification",
      desc: "Image integrity check and plant foliage verification filter out non-agricultural artifacts and invalid files.",
    },
    {
      num: "03",
      title: "Deep Feature Extraction",
      desc: "ResNet-50 backbone evaluates deep spatial representations, followed by GAP, Dense(256), and Softmax layers.",
    },
    {
      num: "04",
      title: "Diagnosis & Advisory",
      desc: "View crop species, primary disease, confidence score, top-3 candidates, and agronomic management advice.",
    },
  ];

  return (
    <div className="space-y-16 py-6 pb-16">
      {/* Hero Section */}
      <section className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-emerald-900 via-emerald-800 to-slate-900 text-white p-8 md:p-14 shadow-xl">
        <div className="absolute top-0 right-0 -mt-10 -mr-10 w-96 h-96 bg-emerald-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="relative z-10 max-w-3xl space-y-6">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/20 border border-emerald-400/30 text-emerald-300 text-xs font-semibold uppercase tracking-wider">
            <Leaf className="w-4 h-4 text-emerald-400" />
            IA-3 Deep Learning Evaluation Project
          </div>
          <h1 className="text-3xl md:text-5xl font-black tracking-tight leading-tight">
            Real-Time Multi-Crop Plant Leaf Disease Detection Using ResNet-50
          </h1>
          <p className="text-slate-200 text-base md:text-lg leading-relaxed">
            An end-to-end intelligent computer vision system combining deep transfer learning,
            rigorous plant leaf validation, and automated agronomic disease management protocols
            for multi-crop agricultural monitoring.
          </p>

          <div className="pt-2 flex flex-wrap gap-4 items-center">
            <Link
              to="/detect"
              className="inline-flex items-center gap-2 px-6 py-3.5 bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold rounded-xl shadow-lg shadow-emerald-500/20 transition"
            >
              <Search className="w-5 h-5" />
              Diagnose Leaf Image
              <ArrowRight className="w-5 h-5" />
            </Link>
            <Link
              to="/performance"
              className="inline-flex items-center gap-2 px-6 py-3.5 bg-white/10 hover:bg-white/20 border border-white/20 text-white font-semibold rounded-xl transition backdrop-blur-sm"
            >
              <BarChart3 className="w-5 h-5" />
              View Model Performance
            </Link>
          </div>
        </div>

        {/* Dynamic Live Stats Cards */}
        <div className="relative z-10 grid grid-cols-2 sm:grid-cols-4 gap-4 mt-10 pt-8 border-t border-white/15">
          <div className="bg-white/5 backdrop-blur-md rounded-2xl p-4 border border-white/10">
            <p className="text-xs text-slate-300 font-medium">Supported Crops</p>
            <p className="text-2xl md:text-3xl font-extrabold text-emerald-400 mt-1">
              {stats.loading ? "..." : stats.cropCount}
            </p>
            <p className="text-xs text-slate-400 mt-0.5">Diverse Species</p>
          </div>
          <div className="bg-white/5 backdrop-blur-md rounded-2xl p-4 border border-white/10">
            <p className="text-xs text-slate-300 font-medium">Trained Classes</p>
            <p className="text-2xl md:text-3xl font-extrabold text-emerald-400 mt-1">
              {stats.loading ? "..." : stats.classCount}
            </p>
            <p className="text-xs text-slate-400 mt-0.5">Pathologies & Healthy</p>
          </div>
          <div className="bg-white/5 backdrop-blur-md rounded-2xl p-4 border border-white/10">
            <p className="text-xs text-slate-300 font-medium">Architecture</p>
            <p className="text-xl md:text-2xl font-extrabold text-white mt-1">ResNet-50</p>
            <p className="text-xs text-slate-400 mt-0.5">ImageNet Backbone</p>
          </div>
          <div className="bg-white/5 backdrop-blur-md rounded-2xl p-4 border border-white/10">
            <p className="text-xs text-slate-300 font-medium">Leaf Guard</p>
            <p className="text-xl md:text-2xl font-extrabold text-emerald-400 mt-1">Active</p>
            <p className="text-xs text-slate-400 mt-0.5">Non-plant Rejection</p>
          </div>
        </div>
      </section>

      {/* Main Features Grid */}
      <section className="space-y-6">
        <div className="text-center max-w-2xl mx-auto space-y-2">
          <h2 className="text-2xl md:text-3xl font-extrabold text-slate-900 tracking-tight">
            Key Engineering & Research Capabilities
          </h2>
          <p className="text-slate-600 text-sm md:text-base">
            Engineered to meet the stringent criteria of IA-3 Deep Learning evaluation: data integrity,
            fine-tuning, out-of-distribution handling, and real-time inference.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((f, i) => (
            <div
              key={i}
              className="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-sm hover:shadow-md transition hover:border-emerald-300 group"
            >
              <div className="w-12 h-12 rounded-xl bg-emerald-50 border border-emerald-100 flex items-center justify-center mb-4 group-hover:scale-105 transition">
                {f.icon}
              </div>
              <h3 className="text-base font-bold text-slate-900 mb-2">{f.title}</h3>
              <p className="text-sm text-slate-600 leading-relaxed">{f.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* How the System Works */}
      <section className="bg-white rounded-3xl p-8 md:p-12 border border-slate-200 shadow-sm space-y-8">
        <div className="text-center max-w-2xl mx-auto space-y-2">
          <span className="text-xs font-bold text-emerald-600 uppercase tracking-widest">
            Pipeline Architecture
          </span>
          <h2 className="text-2xl md:text-3xl font-extrabold text-slate-900 tracking-tight">
            End-to-End Decision Workflow
          </h2>
          <p className="text-slate-600 text-sm">
            How uploaded agricultural imagery is validated, analyzed, and mapped to disease protocols.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {workflowSteps.map((s, i) => (
            <div key={i} className="relative p-5 rounded-2xl bg-slate-50 border border-slate-200/70">
              <span className="text-3xl font-black text-emerald-600/40 block mb-2">{s.num}</span>
              <h4 className="text-base font-bold text-slate-900 mb-1.5">{s.title}</h4>
              <p className="text-xs text-slate-600 leading-relaxed">{s.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Supported Crops Banner */}
      {stats.cropsList.length > 0 && (
        <section className="bg-emerald-50/70 border border-emerald-200/60 rounded-3xl p-8 text-center space-y-4">
          <h3 className="text-lg font-bold text-emerald-950">
            Broad Multi-Crop Coverage Across Verified Datasets
          </h3>
          <p className="text-xs text-emerald-800 max-w-2xl mx-auto">
            Supporting actual evaluated categories across PlantVillage and Rice Leaf Disease datasets:
          </p>
          <div className="flex flex-wrap justify-center gap-2 max-w-4xl mx-auto">
            {stats.cropsList.map((crop, idx) => (
              <span
                key={idx}
                className="px-3 py-1 bg-white border border-emerald-200 rounded-full text-xs font-semibold text-emerald-800 shadow-2xs"
              >
                🌱 {crop}
              </span>
            ))}
          </div>
          <div className="pt-2">
            <Link
              to="/crops"
              className="inline-flex items-center gap-1.5 text-xs font-bold text-emerald-700 hover:text-emerald-800 underline"
            >
              Explore all supported diseases by crop →
            </Link>
          </div>
        </section>
      )}
    </div>
  );
};

export default Home;
