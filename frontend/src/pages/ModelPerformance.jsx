import React, { useState, useEffect } from 'react';
import {
  BarChart3,
  TrendingUp,
  Award,
  Layers,
  Activity,
  CheckCircle2,
  AlertCircle,
  FileText,
  Image as ImageIcon
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import { getPerformanceDashboard, getChartUrl } from '../services/api';

const ModelPerformance = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeChartTab, setActiveChartTab] = useState('recharts'); // recharts or static_charts

  useEffect(() => {
    getPerformanceDashboard()
      .then((res) => {
        setData(res);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Error loading performance data:", err);
        setLoading(false);
      });
  }, []);

  const metrics = data?.metrics || {};
  const modelInfo = data?.model_info || {};
  const history = data?.training_history || [];
  const classReport = data?.classification_report || [];
  const diagnostics = metrics?.diagnostics || {};

  return (
    <div className="space-y-8 py-4 pb-16 max-w-6xl mx-auto">
      {/* Header */}
      <div className="text-center space-y-2">
        <span className="text-xs font-bold text-emerald-600 uppercase tracking-widest">
          Empirical Validation
        </span>
        <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">
          Model Performance & Evaluation Dashboard
        </h1>
        <p className="text-sm text-slate-600 max-w-2xl mx-auto">
          Actual test set evaluation metrics, two-stage training history curves, and class-wise
          precision/recall breakdowns generated from independent test partitions.
        </p>
      </div>

      {loading ? (
        <div className="text-center py-16 text-slate-500 text-sm">
          Loading actual evaluation metrics and charts from backend...
        </div>
      ) : (
        <>
          {/* Key Metric Highlight Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-white p-5 rounded-3xl border border-slate-200 shadow-sm space-y-1">
              <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">
                Test Accuracy
              </span>
              <p className="text-3xl font-black text-emerald-600">
                {metrics.test_accuracy != null
                  ? `${(metrics.test_accuracy * 100).toFixed(2)}%`
                  : 'N/A'}
              </p>
              <p className="text-[11px] text-slate-400">
                Independent Test Partition
              </p>
            </div>

            <div className="bg-white p-5 rounded-3xl border border-slate-200 shadow-sm space-y-1">
              <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">
                Macro F1-Score
              </span>
              <p className="text-3xl font-black text-emerald-600">
                {metrics.macro_f1 != null
                  ? `${(metrics.macro_f1 * 100).toFixed(2)}%`
                  : 'N/A'}
              </p>
              <p className="text-[11px] text-slate-400">
                Balanced Class Average
              </p>
            </div>

            <div className="bg-white p-5 rounded-3xl border border-slate-200 shadow-sm space-y-1">
              <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">
                Macro Precision
              </span>
              <p className="text-3xl font-black text-teal-600">
                {metrics.macro_precision != null
                  ? `${(metrics.macro_precision * 100).toFixed(2)}%`
                  : 'N/A'}
              </p>
              <p className="text-[11px] text-slate-400">
                Weighted: {metrics.weighted_precision ? `${(metrics.weighted_precision * 100).toFixed(1)}%` : 'N/A'}
              </p>
            </div>

            <div className="bg-white p-5 rounded-3xl border border-slate-200 shadow-sm space-y-1">
              <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">
                Macro Recall
              </span>
              <p className="text-3xl font-black text-teal-600">
                {metrics.macro_recall != null
                  ? `${(metrics.macro_recall * 100).toFixed(2)}%`
                  : 'N/A'}
              </p>
              <p className="text-[11px] text-slate-400">
                Evaluated across {metrics.total_test_samples || 0} images
              </p>
            </div>
          </div>

          {/* Model & Architecture Summary Card */}
          <div className="bg-white rounded-3xl p-6 border border-slate-200 shadow-sm space-y-4">
            <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wider flex items-center gap-2">
              <Layers className="w-4 h-4 text-emerald-600" />
              Trained Model Configuration & Provenance
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
              <div className="p-3 bg-slate-50 rounded-xl border border-slate-200/70">
                <span className="text-slate-400 block font-medium">Architecture</span>
                <span className="font-bold text-slate-800 text-sm mt-0.5 block">
                  {modelInfo.architecture || 'ResNet-50'}
                </span>
              </div>
              <div className="p-3 bg-slate-50 rounded-xl border border-slate-200/70">
                <span className="text-slate-400 block font-medium">Pretrained Weights</span>
                <span className="font-bold text-slate-800 text-sm mt-0.5 block">
                  ImageNet (include_top=False)
                </span>
              </div>
              <div className="p-3 bg-slate-50 rounded-xl border border-slate-200/70">
                <span className="text-slate-400 block font-medium">Input Resolution</span>
                <span className="font-bold text-slate-800 text-sm mt-0.5 block">
                  224 × 224 × 3 RGB
                </span>
              </div>
              <div className="p-3 bg-slate-50 rounded-xl border border-slate-200/70">
                <span className="text-slate-400 block font-medium">Total Evaluated Classes</span>
                <span className="font-bold text-emerald-700 text-sm mt-0.5 block">
                  {metrics.total_classes_evaluated || modelInfo.supported_classes || 'N/A'} Classes
                </span>
              </div>
            </div>
          </div>

          {/* Training Curves Section (Interactive Recharts + Static Figures) */}
          <div className="bg-white rounded-3xl p-6 border border-slate-200 shadow-sm space-y-5">
            <div className="flex flex-wrap justify-between items-center gap-3 border-b border-slate-100 pb-4">
              <div>
                <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-emerald-600" />
                  Two-Stage Training History Curves
                </h3>
                <p className="text-xs text-slate-500 mt-0.5">
                  Epoch-by-epoch tracking across Stage 1 (Backbone Frozen) and Stage 2 (Fine-Tuning)
                </p>
              </div>

              <div className="flex rounded-xl bg-slate-100 p-1 border border-slate-200 text-xs font-bold">
                <button
                  onClick={() => setActiveChartTab('recharts')}
                  className={`px-3 py-1.5 rounded-lg transition ${
                    activeChartTab === 'recharts' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600'
                  }`}
                >
                  Interactive Curves
                </button>
                <button
                  onClick={() => setActiveChartTab('static_charts')}
                  className={`px-3 py-1.5 rounded-lg transition ${
                    activeChartTab === 'static_charts' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600'
                  }`}
                >
                  Matplotlib High-Res
                </button>
              </div>
            </div>

            {activeChartTab === 'recharts' && history.length > 0 ? (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Accuracy Chart */}
                <div className="space-y-2">
                  <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider text-center">
                    Accuracy Curve (Train vs Validation)
                  </h4>
                  <div className="h-64 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={history} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                        <XAxis dataKey="epoch" tick={{ fontSize: 11 }} label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
                        <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                        <Tooltip />
                        <Legend wrapperStyle={{ fontSize: '12px' }} />
                        <Line type="monotone" dataKey="accuracy" name="Train Accuracy" stroke="#2563eb" strokeWidth={2.5} dot={{ r: 3 }} />
                        <Line type="monotone" dataKey="val_accuracy" name="Val Accuracy" stroke="#16a34a" strokeWidth={2.5} dot={{ r: 3 }} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Loss Chart */}
                <div className="space-y-2">
                  <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider text-center">
                    Loss Curve (Categorical Crossentropy)
                  </h4>
                  <div className="h-64 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={history} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                        <XAxis dataKey="epoch" tick={{ fontSize: 11 }} label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
                        <YAxis tick={{ fontSize: 11 }} />
                        <Tooltip />
                        <Legend wrapperStyle={{ fontSize: '12px' }} />
                        <Line type="monotone" dataKey="loss" name="Train Loss" stroke="#dc2626" strokeWidth={2.5} dot={{ r: 3 }} />
                        <Line type="monotone" dataKey="val_loss" name="Val Loss" stroke="#9333ea" strokeWidth={2.5} dot={{ r: 3 }} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="border border-slate-200 rounded-2xl overflow-hidden bg-slate-50 p-2">
                  <img
                    src={getChartUrl('accuracy_curve.png')}
                    alt="Accuracy Curve"
                    className="w-full h-auto rounded-xl"
                    onError={(e) => { e.target.style.display = 'none'; }}
                  />
                  <p className="text-xs text-center text-slate-500 mt-2 font-medium">
                    Publication-grade Matplotlib Accuracy Plot (results/accuracy_curve.png)
                  </p>
                </div>
                <div className="border border-slate-200 rounded-2xl overflow-hidden bg-slate-50 p-2">
                  <img
                    src={getChartUrl('loss_curve.png')}
                    alt="Loss Curve"
                    className="w-full h-auto rounded-xl"
                    onError={(e) => { e.target.style.display = 'none'; }}
                  />
                  <p className="text-xs text-center text-slate-500 mt-2 font-medium">
                    Publication-grade Matplotlib Loss Plot (results/loss_curve.png)
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Confusion Matrix and Class Distribution Section */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Confusion Matrix Heatmap */}
            <div className="bg-white rounded-3xl p-6 border border-slate-200 shadow-sm space-y-3">
              <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-emerald-600" />
                Confusion Matrix Heatmap
              </h3>
              <p className="text-xs text-slate-500">
                Predicted vs Ground Truth class distribution across independent test dataset.
              </p>
              <div className="border border-slate-200 rounded-2xl overflow-hidden bg-slate-50 p-2">
                <img
                  src={getChartUrl('confusion_matrix.png')}
                  alt="Confusion Matrix"
                  className="w-full h-auto rounded-xl shadow-xs"
                  onError={(e) => { e.target.style.display = 'none'; }}
                />
              </div>
            </div>

            {/* Class Distribution Chart */}
            <div className="bg-white rounded-3xl p-6 border border-slate-200 shadow-sm space-y-3">
              <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                <ImageIcon className="w-4 h-4 text-emerald-600" />
                Dataset Class Distribution
              </h3>
              <p className="text-xs text-slate-500">
                Number of verified images per category and balance ratio across crops.
              </p>
              <div className="border border-slate-200 rounded-2xl overflow-hidden bg-slate-50 p-2">
                <img
                  src={getChartUrl('class_distribution.png')}
                  alt="Class Distribution"
                  className="w-full h-auto rounded-xl shadow-xs"
                  onError={(e) => { e.target.style.display = 'none'; }}
                />
              </div>
            </div>
          </div>

          {/* Diagnostic & Error Analysis Section */}
          <div className="bg-white rounded-3xl p-6 border border-slate-200 shadow-sm space-y-4">
            <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-emerald-600" />
              Machine Learning Diagnostics & Discussion
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
              <div className="p-4 bg-slate-50 rounded-2xl border border-slate-200 space-y-1.5">
                <h4 className="font-bold text-slate-800">Overfitting Analysis</h4>
                <p className="text-slate-600 leading-relaxed">
                  {diagnostics.overfitting_analysis ||
                    "Dropout (0.5), small data augmentation, and early stopping prevent training divergence and maintain generalizability."}
                </p>
              </div>

              <div className="p-4 bg-slate-50 rounded-2xl border border-slate-200 space-y-1.5">
                <h4 className="font-bold text-slate-800">Underfitting Analysis</h4>
                <p className="text-slate-600 leading-relaxed">
                  {diagnostics.underfitting_analysis ||
                    "Deep residual connections enable 50 layers to capture complex textural cues without gradient degradation."}
                </p>
              </div>

              <div className="p-4 bg-slate-50 rounded-2xl border border-slate-200 space-y-1.5">
                <h4 className="font-bold text-slate-800">Model Limitations</h4>
                <p className="text-slate-600 leading-relaxed">
                  {diagnostics.limitations ||
                    "Similar necrotic spots across related Solanaceae species (Tomato vs Potato Early Blight) demand close foliar examination."}
                </p>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default ModelPerformance;
