import React from 'react';
import { Cpu, Layers, GitBranch, ShieldCheck, Database, ArrowRight } from 'lucide-react';
import { Link } from 'react-router-dom';

const AboutModel = () => {
  return (
    <div className="space-y-12 py-4 pb-16 max-w-4xl mx-auto">
      {/* Header */}
      <div className="text-center space-y-2">
        <span className="text-xs font-bold text-emerald-600 uppercase tracking-widest">
          Academic Documentation & Methodology
        </span>
        <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">
          About the ResNet-50 Architecture & Methodology
        </h1>
        <p className="text-sm text-slate-600 max-w-2xl mx-auto">
          Technical explanation of deep residual learning, transfer learning mechanics,
          two-stage optimization, and dataset provenance prepared for IA-3 evaluation.
        </p>
      </div>

      {/* 1. ResNet-50 Architecture */}
      <section className="bg-white rounded-3xl p-6 md:p-8 border border-slate-200 shadow-sm space-y-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-emerald-100 text-emerald-700 flex items-center justify-center font-bold">
            <Cpu className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-slate-900">
              1. Deep Residual Networks (ResNet-50)
            </h2>
            <p className="text-xs text-slate-500">
              Mitigating vanishing gradients via identity skip connections
            </p>
          </div>
        </div>

        <p className="text-sm text-slate-700 leading-relaxed">
          Traditional deep convolutional neural networks suffer from the <em>degradation problem</em>: as network depth increases, accuracy saturates and degrades rapidly due to vanishing/exploding gradients. <strong>ResNet-50</strong> solves this by introducing <em>residual shortcut connections</em> that perform identity mappings:
        </p>

        <div className="bg-slate-50 p-4 rounded-2xl border border-slate-200 font-mono text-xs text-slate-800 text-center">
          <span className="text-emerald-700 font-bold">F(x) = H(x) - x</span> &nbsp;→&nbsp; <span className="text-blue-700 font-bold">H(x) = F(x) + x</span>
        </div>

        <p className="text-sm text-slate-700 leading-relaxed">
          Instead of forcing stacked layers to fit the underlying mapping <code className="bg-slate-100 px-1 py-0.5 rounded text-xs">H(x)</code>, the network learns a residual mapping <code className="bg-slate-100 px-1 py-0.5 rounded text-xs">F(x)</code>. This allows gradients to flow directly back through the identity shortcuts, enabling effective training of 50 deep convolutional layers.
        </p>
      </section>

      {/* 2. Transfer Learning Rationale */}
      <section className="bg-white rounded-3xl p-6 md:p-8 border border-slate-200 shadow-sm space-y-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-blue-100 text-blue-700 flex items-center justify-center font-bold">
            <Layers className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-slate-900">
              2. Transfer Learning & Classification Head
            </h2>
            <p className="text-xs text-slate-500">
              Repurposing ImageNet visual features for agricultural pathology
            </p>
          </div>
        </div>

        <p className="text-sm text-slate-700 leading-relaxed">
          Training a 50-layer network from scratch requires millions of labeled images and massive compute. Transfer learning leverages weights pre-trained on 1.4 million images from ImageNet (1,000 classes), which have already learned generic low-level and mid-level visual features like edges, corners, textures, and gradient patterns.
        </p>

        <div className="space-y-2 bg-slate-50 p-4 rounded-2xl border border-slate-200 text-xs">
          <p className="font-bold text-slate-800 uppercase tracking-wider text-[11px]">
            Custom Network Head Architecture:
          </p>
          <ul className="space-y-1.5 text-slate-700 list-disc list-inside">
            <li><strong>Input Image:</strong> 224 × 224 × 3 RGB tensor</li>
            <li><strong>ResNet-50 Backbone:</strong> Pretrained weights, <code className="font-mono text-[11px]">include_top=False</code></li>
            <li><strong>Global Average Pooling 2D:</strong> Collapses spatial 7 × 7 feature maps into a 2048-dimensional feature vector without flattening parameter explosion</li>
            <li><strong>Dense Layer (256, ReLU):</strong> Learns task-specific disease feature combinations</li>
            <li><strong>Dropout (0.5):</strong> Deactivates 50% of neurons during training to prevent co-adaptation</li>
            <li><strong>Final Dense Layer (Softmax):</strong> Dynamically dimensioned to the number of supported disease classes</li>
          </ul>
        </div>
      </section>

      {/* 3. Two-Stage Training Methodology */}
      <section className="bg-white rounded-3xl p-6 md:p-8 border border-slate-200 shadow-sm space-y-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-teal-100 text-teal-700 flex items-center justify-center font-bold">
            <GitBranch className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-slate-900">
              3. Two-Stage Training Methodology
            </h2>
            <p className="text-xs text-slate-500">
              Feature extraction followed by deep fine-tuning
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-4 rounded-2xl bg-slate-50 border border-slate-200 space-y-2">
            <span className="text-xs font-bold text-emerald-700 uppercase tracking-wider block">
              Stage 1: Feature Extraction
            </span>
            <p className="text-xs text-slate-600 leading-relaxed">
              The entire ResNet-50 backbone is frozen. Only the custom dense classification layers are trained with an Adam optimizer learning rate of <code className="font-mono text-xs">1e-4</code>. This prevents randomly initialized weights from destroying pretrained ImageNet weights.
            </p>
          </div>

          <div className="p-4 rounded-2xl bg-slate-50 border border-slate-200 space-y-2">
            <span className="text-xs font-bold text-teal-700 uppercase tracking-wider block">
              Stage 2: Fine-Tuning
            </span>
            <p className="text-xs text-slate-600 leading-relaxed">
              The top convolutional blocks of ResNet-50 are unfrozen while keeping BatchNormalization in inference mode. The model is recompiled with a 10x smaller learning rate (<code className="font-mono text-xs">1e-5</code>) to carefully adapt high-level visual representations to plant pathology nuances.
            </p>
          </div>
        </div>
      </section>

      {/* 4. Dataset Sources & Preprocessing */}
      <section className="bg-white rounded-3xl p-6 md:p-8 border border-slate-200 shadow-sm space-y-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-amber-100 text-amber-700 flex items-center justify-center font-bold">
            <Database className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-slate-900">
              4. Dataset Provenance & Preprocessing Pipeline
            </h2>
            <p className="text-xs text-slate-500">
              Verified multi-crop datasets with stratified partitions
            </p>
          </div>
        </div>

        <p className="text-sm text-slate-700 leading-relaxed">
          The unified dataset integrates authentic imagery from public agricultural repositories:
        </p>

        <ul className="text-xs text-slate-700 space-y-1.5 list-disc list-inside bg-slate-50 p-4 rounded-2xl border border-slate-200">
          <li><strong>PlantVillage:</strong> Verified laboratory and field imagery of Apple, Blueberry, Cherry, Corn, Grape, Peach, Pepper, Potato, Strawberry, and Tomato.</li>
          <li><strong>Rice Leaf Disease Dataset:</strong> Verified images of Bacterial Leaf Blight, Brown Spot, and Leaf Smut in rice paddies.</li>
          <li><strong>Stratified Split:</strong> 70% Train, 15% Validation, and 15% Test partitions, with split manifest preserved to eliminate data leakage.</li>
          <li><strong>Preprocessing:</strong> Resized to 224 × 224, RGB standardization, and official ImageNet zero-centering via <code className="font-mono text-xs">preprocess_input</code>.</li>
        </ul>
      </section>

      {/* 5. Leaf Guard and Out-of-Distribution Handling */}
      <section className="bg-white rounded-3xl p-6 md:p-8 border border-slate-200 shadow-sm space-y-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-purple-100 text-purple-700 flex items-center justify-center font-bold">
            <ShieldCheck className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-slate-900">
              5. Out-of-Distribution & Leaf Guard
            </h2>
            <p className="text-xs text-slate-500">
              Mandatory protection against non-plant false positives
            </p>
          </div>
        </div>

        <p className="text-sm text-slate-700 leading-relaxed">
          Standard softmax classifiers assign 100% total probability across known classes, forcing non-plant images (e.g. cars, faces, electronics) into one of the disease classes with misleading confidence. AgriVision AI incorporates a dual protection mechanism:
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          <div className="p-4 bg-slate-50 rounded-2xl border border-slate-200 space-y-1.5">
            <h4 className="font-bold text-slate-800">1. Semantic Flora Verification</h4>
            <p className="text-slate-600 leading-relaxed">
              Checks ImageNet broad category predictions and excess green index (ExG) to reject non-plant objects upfront with a clear warning: <em>"Invalid Image: Please upload a valid plant or leaf image."</em>
            </p>
          </div>

          <div className="p-4 bg-slate-50 rounded-2xl border border-slate-200 space-y-1.5">
            <h4 className="font-bold text-slate-800">2. Entropy & Margin Guard</h4>
            <p className="text-slate-600 leading-relaxed">
              If predictive distribution has excessive Shannon entropy or negligible top-1 vs top-2 confidence margin, the system flags the image as ambiguous rather than giving a false confident diagnosis.
            </p>
          </div>
        </div>
      </section>

      {/* 6. Limitations */}
      <section className="bg-white rounded-3xl p-6 md:p-8 border border-slate-200 shadow-sm space-y-3">
        <h2 className="text-base font-bold text-slate-900">
          6. Model Limitations & Real-World Considerations
        </h2>
        <p className="text-xs text-slate-600 leading-relaxed">
          While ResNet-50 achieves high accuracy on controlled leaf imagery, real-world deployment faces challenges including variable field illumination, overlapping leaves, multiple concurrent infections, and nutritional deficiencies mimicking fungal chlorosis. The system is designed to provide diagnostic support alongside field agronomic verification.
        </p>
      </section>
    </div>
  );
};

export default AboutModel;
