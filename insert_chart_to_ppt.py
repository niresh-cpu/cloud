"""
Helper script to insert the dataset distribution chart into a PowerPoint presentation.
Supports:
1. Updating an existing PowerPoint file by replacing any image or shape containing 'dataset class distribution' or on Slide 2.
2. Generating a fresh IA-3 PowerPoint presentation with Slide 1 (Project Overview) and Slide 2 (Dataset Distribution).
"""

import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

BASE_DIR = Path(__file__).resolve().parent
CHART_PATH = BASE_DIR / "results" / "dataset_class_distribution_green.png"
OUTPUT_PPTX = BASE_DIR / "IA-3_Plant_Disease_Detection_Presentation.pptx"

def create_or_update_presentation(existing_pptx_path=None):
    if not CHART_PATH.exists():
        raise FileNotFoundError(f"Chart image not found at {CHART_PATH}")

    if existing_pptx_path and Path(existing_pptx_path).exists():
        prs = Presentation(existing_pptx_path)
        print(f"Loaded existing presentation: {existing_pptx_path} (Slides: {len(prs.slides)})")
        
        # Target Slide 2 (index 1)
        if len(prs.slides) >= 2:
            slide = prs.slides[1]
        else:
            slide = prs.slides.add_slide(prs.slide_layouts[5] if len(prs.slide_layouts) > 5 else prs.slide_layouts[0])
            
        # Check for placeholder or shape with 'distribution' or 'placeholder'
        placeholder_shape = None
        for shape in list(slide.shapes):
            if shape.has_text_frame:
                txt = shape.text_frame.text.lower()
                if any(k in txt for k in ["dataset class distribution", "distribution graph", "placeholder"]):
                    placeholder_shape = shape
                    break
        
        # Position and size
        if placeholder_shape is not None:
            left = placeholder_shape.left
            top = placeholder_shape.top
            width = placeholder_shape.width
            height = placeholder_shape.height
            # Remove placeholder text shape
            sp = placeholder_shape._element
            sp.getparent().remove(sp)
            # Add picture in its place
            slide.shapes.add_picture(str(CHART_PATH), left, top, width=width, height=height)
            print("Replaced placeholder shape with distribution chart.")
        else:
            # Place cleanly on the right half or center of Slide 2 without overlapping title
            left = Inches(4.5)
            top = Inches(1.5)
            width = Inches(8.5)
            slide.shapes.add_picture(str(CHART_PATH), left, top, width=width)
            print("Inserted chart into Slide 2.")
            
        save_path = existing_pptx_path
        prs.save(save_path)
        print(f"Saved updated presentation to: {save_path}")
        return save_path
    else:
        # Create professional IA-3 Presentation with 16:9 widescreen
        prs = Presentation()
        prs.slide_width = Inches(13.333)
        prs.slide_height = Inches(7.5)
        blank_layout = prs.slide_layouts[6]
        
        # --- Slide 1: Title & IA-3 Cover ---
        slide1 = prs.slides.add_slide(blank_layout)
        
        # Header shape
        header = slide1.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(11.333), Inches(4.0))
        tf = header.text_frame
        tf.word_wrap = True
        
        p0 = tf.paragraphs[0]
        p0.text = "IA-3 SKILL-BASED EVALUATION (25 MARKS)"
        p0.font.size = Pt(14)
        p0.font.bold = True
        p0.font.color.rgb = RGBColor(22, 163, 74)
        
        p1 = tf.add_paragraph()
        p1.text = "Real-Time Multi-Crop Plant Leaf Disease Detection"
        p1.font.size = Pt(32)
        p1.font.bold = True
        p1.font.color.rgb = RGBColor(15, 23, 42)
        p1.space_before = Pt(12)
        
        p2 = tf.add_paragraph()
        p2.text = "Deep Residual Learning (ResNet-50) Transfer Learning with Automated Foliage Guard"
        p2.font.size = Pt(16)
        p2.font.color.rgb = RGBColor(71, 85, 105)
        p2.space_before = Pt(8)

        p3 = tf.add_paragraph()
        p3.text = "Course: Deep Learning (23DC2040)  |  B.Tech. AI & Data Science  |  Semester V"
        p3.font.size = Pt(13)
        p3.font.color.rgb = RGBColor(100, 116, 139)
        p3.space_before = Pt(24)
        
        # --- Slide 2: Dataset Class Distribution ---
        slide2 = prs.slides.add_slide(blank_layout)
        
        # Slide Title
        title_box = slide2.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.8))
        tf2 = title_box.text_frame
        p_t = tf2.paragraphs[0]
        p_t.text = "Problem Identification & Multi-Crop Dataset Distribution"
        p_t.font.size = Pt(22)
        p_t.font.bold = True
        p_t.font.color.rgb = RGBColor(15, 23, 42)
        
        # Left Info Card
        card = slide2.shapes.add_textbox(Inches(0.8), Inches(1.3), Inches(4.8), Inches(5.6))
        c_tf = card.text_frame
        c_tf.word_wrap = True
        
        def add_bullet(tf_obj, heading, text):
            p = tf_obj.add_paragraph()
            p.text = f"{heading}: "
            p.font.bold = True
            p.font.size = Pt(12)
            p.font.color.rgb = RGBColor(22, 163, 74)
            p.space_before = Pt(10)
            
            # add plain text run
            run = p.add_run()
            run.text = text
            run.font.bold = False
            run.font.color.rgb = RGBColor(51, 65, 85)
            
        c_p = c_tf.paragraphs[0]
        c_p.text = "Dataset Synthesis & Coverage"
        c_p.font.bold = True
        c_p.font.size = Pt(15)
        c_p.font.color.rgb = RGBColor(15, 23, 42)
        
        add_bullet(c_tf, "Total Classes", "37 Verified Agricultural Pathologies & Healthy Categories")
        add_bullet(c_tf, "Crops Covered (11)", "Apple, Blueberry, Cherry, Corn, Grape, Peach, Pepper, Potato, Rice, Strawberry, Tomato")
        add_bullet(c_tf, "Ground-Truth Sources", "PlantVillage & Rice Leaf Disease benchmark repositories")
        add_bullet(c_tf, "Stratified Split", "70% Training (1,217 imgs) / 15% Validation (250 imgs) / 15% Test (272 imgs)")
        add_bullet(c_tf, "Pre-processing", "Standardized RGB 224x224 with ResNet-50 zero-centering & TTA")
        
        # Insert Chart Image on the right half of Slide 2 without overlapping
        chart_left = Inches(5.8)
        chart_top = Inches(1.1)
        chart_width = Inches(7.0)
        chart_height = Inches(6.0)
        slide2.shapes.add_picture(str(CHART_PATH), chart_left, chart_top, width=chart_width, height=chart_height)
        
        prs.save(str(OUTPUT_PPTX))
        print(f"Generated complete IA-3 presentation with Slide 2 chart: {OUTPUT_PPTX}")
        return str(OUTPUT_PPTX)

if __name__ == "__main__":
    path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    create_or_update_presentation(path_arg)
