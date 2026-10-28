import React, { useState, useRef } from 'react';
import {
  UploadCloud,
  Camera,
  RefreshCw,
  Search,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Clock,
  ShieldCheck,
  Activity,
  Layers,
  ChevronDown,
  Info
} from 'lucide-react';
import { predictDisease } from '../services/api';
import CameraModal from '../components/CameraModal';

const DiseaseDetection = () => {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [activeTab, setActiveTab] = useState('symptoms');
  const fileInputRef = useRef(null);

  const handleFileSelect = (selectedFile) => {
    if (!selectedFile) return;
    setFile(selectedFile);
    setPrediction(null);
    setErrorMsg(null);
    const url = URL.createObjectURL(selectedFile);
    setPreviewUrl(url);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleCameraCapture = (capturedFile, capturedPreviewUrl) => {
    setFile(capturedFile);
    setPreviewUrl(capturedPreviewUrl);
    setPrediction(null);
    setErrorMsg(null);
  };

  const handlePredict = async () => {
    if (!file) return;
    setLoading(true);
    setErrorMsg(null);

    try {
      const data = await predictDisease(file);
      setPrediction(data);
    } catch (err) {
      console.error("Prediction error:", err);
      const msg = err.response?.data?.detail || err.message || "Network or inference service error occurred.";
      setErrorMsg(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreviewUrl(null);
    setPrediction(null);
    setErrorMsg(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <div className="space-y-8 py-4 pb-16 max-w-5xl mx-auto">
      {/* Page Header */}
      <div className="text-center space-y-2">
        <span className="text-xs font-bold text-emerald-600 uppercase tracking-widest">
          Computer Vision Diagnostics
        </span>
        <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">
          Real-Time Multi-Crop Disease Detection
        </h1>
        <p className="text-sm text-slate-600 max-w-2xl mx-auto">
          Upload or capture a high-resolution leaf photograph. The system will validate foliage
          authenticity and classify specific plant diseases using ResNet-50 transfer learning.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Upload & Input Column */}
        <div className="lg:col-span-5 space-y-4">
          <div className="bg-white rounded-3xl p-6 border border-slate-200 shadow-sm space-y-5">
            <h2 className="text-base font-bold text-slate-800 flex items-center justify-between">
              <span>Input Leaf Image</span>
              {file && (
                <button
                  onClick={handleReset}
                  className="text-xs text-slate-500 hover:text-rose-600 flex items-center gap-1 font-semibold"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                  Reset
                </button>
              )}
            </h2>


            {/* Drop Zone */}
            {!previewUrl ? (
              <div
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-emerald-200 hover:border-emerald-500 bg-emerald-50/30 hover:bg-emerald-50/60 transition rounded-2xl p-8 text-center cursor-pointer flex flex-col items-center justify-center space-y-3 min-h-[260px]"
              >
                <div className="w-14 h-14 rounded-2xl bg-emerald-100/80 text-emerald-700 flex items-center justify-center shadow-inner">
                  <UploadCloud className="w-7 h-7" />
                </div>
                <div className="space-y-1">
                  <p className="text-sm font-bold text-slate-800">
                    Click to browse or drag and drop
                  </p>
                  <p className="text-xs text-slate-500">
                    Supports JPG, JPEG, PNG, WEBP, BMP, TIFF, GIF
                  </p>
                </div>
                <span className="text-[11px] text-emerald-700 font-medium px-2 py-0.5 bg-emerald-100 rounded-md">
                  Max file size: 20MB
                </span>
              </div>
            ) : (
              <div className="relative rounded-2xl overflow-hidden border border-slate-200 bg-slate-900 aspect-square flex items-center justify-center shadow-inner">
                <img
                  src={previewUrl}
                  alt="Leaf Preview"
                  className="w-full h-full object-contain"
                />
                <button
                  onClick={handleReset}
                  className="absolute top-3 right-3 p-1.5 bg-black/60 hover:bg-black/80 text-white rounded-lg backdrop-blur-sm transition"
                >
                  <RefreshCw className="w-4 h-4" />
                </button>
              </div>
            )}

            {/* Hidden Input */}
            <input
              type="file"
              ref={fileInputRef}
              onChange={(e) => handleFileSelect(e.target.files[0])}
              accept=".jpg,.jpeg,.png,.webp,.bmp,.tiff,.tif,.gif"
              className="hidden"
            />

            {/* Action Buttons */}
            <div className="grid grid-cols-2 gap-3 pt-2">
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="w-full py-2.5 px-3 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-xl font-semibold text-xs transition flex items-center justify-center gap-1.5"
              >
                <UploadCloud className="w-4 h-4" />
                Browse File
              </button>
              <button
                type="button"
                onClick={() => setIsCameraOpen(true)}
                className="w-full py-2.5 px-3 bg-emerald-50 hover:bg-emerald-100 text-emerald-700 border border-emerald-200 rounded-xl font-semibold text-xs transition flex items-center justify-center gap-1.5"
              >
                <Camera className="w-4 h-4" />
                Live Camera
              </button>
            </div>

            {/* Predict Button */}
            <button
              type="button"
              onClick={() => handlePredict()}
              disabled={!file || loading}
              className="w-full py-3.5 px-4 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl font-bold text-sm shadow-md transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Running ResNet-50 Diagnostics...
                </>
              ) : (
                <>
                  <Search className="w-4 h-4" />
                  Predict Disease
                </>
              )}
            </button>
          </div>

          {/* Validation Notice Box */}
          <div className="bg-slate-50 border border-slate-200 rounded-2xl p-4 text-xs text-slate-600 space-y-1.5">
            <div className="flex items-center gap-1.5 font-bold text-slate-800">
              <ShieldCheck className="w-4 h-4 text-emerald-600" />
              Active Out-of-Distribution Guard
            </div>
            <p>
              Non-agricultural objects (cars, buildings, animals, people) will be actively
              rejected by our foliage validation pipeline to avoid false classifications.
            </p>
          </div>
        </div>

        {/* Prediction Results Column */}
        <div className="lg:col-span-7 space-y-5">
          {errorMsg && (
            <div className="p-5 rounded-2xl bg-rose-50 border border-rose-200 text-rose-800 flex items-start gap-3">
              <XCircle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
              <div>
                <p className="font-bold text-sm">Inference Service Warning</p>
                <p className="text-xs mt-1">{errorMsg}</p>
              </div>
            </div>
          )}

          {/* Case 1: Rejection - Not a valid plant leaf */}
          {prediction && !prediction.success && prediction.status === 'not_a_plant' && (
            <div className="p-6 rounded-3xl bg-rose-50 border-2 border-rose-200 text-rose-900 shadow-sm space-y-3">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-rose-600 text-white flex items-center justify-center shrink-0">
                  <XCircle className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="font-extrabold text-base text-rose-900">
                    Invalid Image: Please upload a valid plant or leaf image.
                  </h3>
                  <p className="text-xs text-rose-700 mt-0.5">
                    Foliage validation check failed ({prediction.details || "Non-plant pattern detected"}).
                  </p>
                </div>
              </div>
              <p className="text-xs text-rose-800 bg-white/60 p-3 rounded-xl border border-rose-100">
                The image upload does not display characteristic plant leaf structures, chloroplast
                reflections, or crop vein architectures. Please upload an image of a supported crop leaf.
              </p>
              <div className="text-[11px] text-rose-600 flex items-center gap-1 font-semibold">
                <Clock className="w-3.5 h-3.5" />
                Validation elapsed time: {prediction.inference_time_ms} ms
              </div>
            </div>
          )}

          {/* Case 2: High Uncertainty / Out-of-Distribution */}
          {prediction && !prediction.success && prediction.status === 'uncertain_prediction' && (
            <div className="p-6 rounded-3xl bg-amber-50 border-2 border-amber-200 text-amber-900 shadow-sm space-y-3">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-amber-500 text-white flex items-center justify-center shrink-0">
                  <AlertTriangle className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="font-extrabold text-base text-amber-900">
                    High Prediction Uncertainty
                  </h3>
                  <p className="text-xs text-amber-700 mt-0.5">
                    {prediction.message}
                  </p>
                </div>
              </div>
              <p className="text-xs text-amber-800 bg-white/60 p-3 rounded-xl border border-amber-100">
                The ResNet-50 predictive entropy is too high or confidence margin between candidate
                classes is below reliable threshold. Provisional candidate was <strong>{prediction.provisional_class}</strong> ({Math.round(prediction.confidence * 100)}%).
              </p>
              <div className="text-[11px] text-amber-700 flex items-center gap-1 font-semibold">
                <Clock className="w-3.5 h-3.5" />
                Inference elapsed time: {prediction.inference_time_ms} ms
              </div>
            </div>
          )}

          {/* Case 3: Success Prediction Result Card */}
          {prediction && prediction.success && (
            <div className="bg-white rounded-3xl border border-slate-200 shadow-sm overflow-hidden space-y-6">
              {/* Top Banner */}
              <div
                className={`p-6 border-b ${
                  prediction.is_healthy
                    ? 'bg-gradient-to-r from-emerald-500 to-green-600 text-white'
                    : 'bg-gradient-to-r from-rose-600 to-orange-600 text-white'
                }`}
              >
                <div className="flex flex-wrap justify-between items-start gap-4">
                  <div>
                    <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 bg-white/20 backdrop-blur-md rounded-full text-xs font-semibold uppercase tracking-wider mb-2">
                      {prediction.is_healthy ? (
                        <>
                          <CheckCircle2 className="w-3.5 h-3.5" /> Healthy Foliage
                        </>
                      ) : (
                        <>
                          <AlertTriangle className="w-3.5 h-3.5" /> Disease Identified
                        </>
                      )}
                    </div>
                    <h2 className="text-2xl font-black tracking-tight">
                      {prediction.crop} · {prediction.disease}
                    </h2>
                    <p className="text-xs text-white/90 mt-1">
                      Pathogen: {prediction.pathogen_type} · Severity: {prediction.severity}
                    </p>
                  </div>

                  {/* Confidence Pill */}
                  <div className="bg-white/15 backdrop-blur-md px-4 py-2 rounded-2xl border border-white/20 text-right">
                    <p className="text-[10px] uppercase font-bold text-white/80">Confidence</p>
                    <p className="text-2xl font-black">{prediction.confidence_percentage}</p>
                    <p className="text-[10px] text-white/70">
                      ⏱ {prediction.inference_time_ms} ms
                    </p>
                  </div>
                </div>
              </div>

              <div className="p-6 space-y-6">
                {/* Detected Species Candidates */}
                {prediction.crop_breakdown && prediction.crop_breakdown.length > 0 && (
                  <div className="space-y-2 bg-emerald-50/60 p-3.5 rounded-2xl border border-emerald-200/80">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-bold text-slate-800 flex items-center gap-1.5">
                        <span>🌱</span> Detected Plant Species Candidates:
                      </span>
                      <span className="text-[10px] text-emerald-700 font-semibold">
                        Multi-Crop Visual Distribution
                      </span>
                    </div>
                    <div className="flex flex-wrap gap-2 pt-1">
                      {prediction.crop_breakdown.map((item, idx) => {
                        const isPrimary = prediction.crop.toLowerCase() === item.crop.toLowerCase();
                        return (
                          <div
                            key={idx}
                            className={`text-xs px-3 py-1.5 rounded-xl font-bold transition flex items-center gap-1.5 shadow-sm ${
                              isPrimary
                                ? 'bg-emerald-600 text-white shadow-emerald-200'
                                : 'bg-white text-slate-700 border border-slate-200'
                            }`}
                          >
                            <span>{item.crop}</span>
                            <span className={isPrimary ? 'text-emerald-100 text-[10px]' : 'text-emerald-600 text-[10px]'}>
                              {item.score}%
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Top 3 Predictions Bar */}
                <div className="space-y-3 bg-slate-50 p-4 rounded-2xl border border-slate-200/80">
                  <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider flex items-center gap-1.5">
                    <Layers className="w-3.5 h-3.5 text-emerald-600" />
                    Top-3 Model Candidate Predictions
                  </h4>
                  <div className="space-y-2">
                    {prediction.top_predictions?.map((cand, idx) => (
                      <div key={idx} className="space-y-1">
                        <div className="flex justify-between text-xs font-semibold text-slate-700">
                          <span>
                            {idx + 1}. {cand.crop} — {cand.disease}
                          </span>
                          <span className="text-emerald-700 font-bold">
                            {cand.confidence_percentage}
                          </span>
                        </div>
                        <div className="w-full bg-slate-200 rounded-full h-2 overflow-hidden">
                          <div
                            className={`h-2 rounded-full transition-all duration-500 ${
                              idx === 0 ? 'bg-emerald-600' : 'bg-slate-400'
                            }`}
                            style={{ width: `${Math.round(cand.confidence * 100)}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Botanical Description */}
                <div className="space-y-1.5">
                  <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wider">
                    Pathological Description
                  </h4>
                  <p className="text-sm text-slate-700 leading-relaxed">
                    {prediction.description}
                  </p>
                </div>

                {/* Agronomic Advisory Tabs */}
                <div className="space-y-3">
                  <div className="flex border-b border-slate-200">
                    <button
                      onClick={() => setActiveTab('symptoms')}
                      className={`pb-2.5 px-4 text-xs font-bold transition border-b-2 ${
                        activeTab === 'symptoms'
                          ? 'border-emerald-600 text-emerald-700'
                          : 'border-transparent text-slate-500 hover:text-slate-800'
                      }`}
                    >
                      Key Symptoms
                    </button>
                    <button
                      onClick={() => setActiveTab('prevention')}
                      className={`pb-2.5 px-4 text-xs font-bold transition border-b-2 ${
                        activeTab === 'prevention'
                          ? 'border-emerald-600 text-emerald-700'
                          : 'border-transparent text-slate-500 hover:text-slate-800'
                      }`}
                    >
                      Cultural Prevention
                    </button>
                    <button
                      onClick={() => setActiveTab('management')}
                      className={`pb-2.5 px-4 text-xs font-bold transition border-b-2 ${
                        activeTab === 'management'
                          ? 'border-emerald-600 text-emerald-700'
                          : 'border-transparent text-slate-500 hover:text-slate-800'
                      }`}
                    >
                      IPM Management
                    </button>
                  </div>

                  <div className="bg-slate-50/80 p-4 rounded-2xl border border-slate-200/60 min-h-[120px]">
                    {activeTab === 'symptoms' && (
                      <ul className="space-y-2">
                        {prediction.symptoms?.map((s, i) => (
                          <li key={i} className="flex items-start gap-2 text-xs text-slate-700">
                            <span className="w-1.5 h-1.5 rounded-full bg-emerald-600 mt-1.5 shrink-0" />
                            <span>{s}</span>
                          </li>
                        ))}
                      </ul>
                    )}

                    {activeTab === 'prevention' && (
                      <ul className="space-y-2">
                        {prediction.prevention?.map((p, i) => (
                          <li key={i} className="flex items-start gap-2 text-xs text-slate-700">
                            <span className="w-1.5 h-1.5 rounded-full bg-emerald-600 mt-1.5 shrink-0" />
                            <span>{p}</span>
                          </li>
                        ))}
                      </ul>
                    )}

                    {activeTab === 'management' && (
                      <ul className="space-y-2">
                        {prediction.management?.map((m, i) => (
                          <li key={i} className="flex items-start gap-2 text-xs text-slate-700">
                            <span className="w-1.5 h-1.5 rounded-full bg-emerald-600 mt-1.5 shrink-0" />
                            <span>{m}</span>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Diagnostic Loading State */}
          {loading && (
            <div className="bg-white rounded-3xl p-10 border border-emerald-200/60 shadow-sm text-center flex flex-col items-center justify-center min-h-[380px] space-y-5">
              <div className="relative">
                <div className="w-20 h-20 rounded-3xl bg-emerald-100 flex items-center justify-center text-emerald-600 shadow-inner">
                  <RefreshCw className="w-10 h-10 animate-spin" />
                </div>
              </div>
              <div className="space-y-2 max-w-md">
                <h3 className="font-extrabold text-slate-900 text-lg">
                  Analyzing Plant Foliage...
                </h3>
                <p className="text-xs text-slate-500 leading-relaxed">
                  Executing ResNet-50 residual convolutional blocks across 37 crop disease categories.
                  Evaluating leaf texture, necrotic margins, and agronomic management protocols.
                </p>
                <div className="pt-2">
                  <span className="inline-flex items-center gap-1.5 px-3 py-1 bg-emerald-50 text-emerald-700 rounded-full text-[11px] font-semibold border border-emerald-200">
                    <Activity className="w-3.5 h-3.5 animate-pulse" /> ResNet-50 Neural Inference in Progress
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Initial State / Placeholder */}
          {!prediction && !loading && !errorMsg && (
            <div className="bg-white rounded-3xl p-8 border border-slate-200 shadow-sm text-center flex flex-col items-center justify-center min-h-[360px] space-y-4">
              <div className="w-16 h-16 rounded-2xl bg-slate-100 text-slate-400 flex items-center justify-center">
                <Search className="w-8 h-8" />
              </div>
              <div className="space-y-1 max-w-sm">
                <h3 className="font-bold text-slate-800 text-base">Awaiting Leaf Upload</h3>
                <p className="text-xs text-slate-500">
                  Select an image file or use the live camera to view ResNet-50 predictions,
                  confidence rankings, and agricultural guidance.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Camera Capture Modal */}
      <CameraModal
        isOpen={isCameraOpen}
        onClose={() => setIsCameraOpen(false)}
        onCapture={handleCameraCapture}
      />
    </div>
  );
};

export default DiseaseDetection;
