#!/usr/bin/env python
"""BoneMedVQA — clinical research demo UI (responsive web layout)."""

from __future__ import annotations

import html
import os
import sys
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bonemedvqa.inference.confidence import WARNING
from bonemedvqa.inference.predictor import Predictor
from bonemedvqa.utils.io import load_yaml

CFG_PATH = ROOT / "configs" / "maxillofacial.yaml"
if not CFG_PATH.exists():
    CFG_PATH = ROOT / "configs" / "baseline.yaml"
if not CFG_PATH.exists():
    CFG_PATH = ROOT / "configs" / "lightweight.yaml"
ckpt_candidates = [
    ROOT / "outputs" / "maxillofacial" / "checkpoints" / "best.pt",
    ROOT / "outputs" / "checkpoints" / "best.pt",
]
ckpt_path = next((p for p in ckpt_candidates if p.exists()), None)
PREDICTOR = Predictor(
    load_yaml(CFG_PATH),
    checkpoint=ckpt_path,
    backend=os.getenv("BONEMEDVQA_BACKEND", "auto"),
)
BACKEND = getattr(PREDICTOR, "backend_name", "unknown").upper()
KG_ON = getattr(PREDICTOR, "use_kg", False)

SAMPLE_DIRS = [
    ROOT / "demo" / "assets" / "samples",
    ROOT / "data" / "images" / "maxillofacial",
    ROOT / "data" / "images",
]
SAMPLE_IMAGES: list[str] = []
for d in SAMPLE_DIRS:
    if d.exists():
        SAMPLE_IMAGES.extend(
            str(p.resolve())
            for p in sorted(d.glob("*"))
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
SAMPLE_IMAGES = list(dict.fromkeys(SAMPLE_IMAGES))[:12]

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── App shell ── */
html, body { overflow-x: hidden !important; }
.gradio-container {
  width: 100% !important;
  max-width: 100% !important;
  margin: 0 auto !important;
  padding: 12px 16px 24px !important;
  font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
  box-sizing: border-box !important;
  overflow-x: hidden !important;
}
.main, .wrap, .gradio-container, .contain, .app {
  background: #eef1f6 !important;
  max-width: 100vw !important;
  overflow-x: hidden !important;
}
footer { display: none !important; }
.block, .form, .column, .row {
  box-sizing: border-box !important;
  min-width: 0 !important;
  max-width: 100% !important;
}

/* ── Header ── */
.bm-header {
  background: linear-gradient(135deg, #0c4a6e 0%, #0f766e 55%, #134e4a 100%);
  border-radius: 16px;
  padding: 20px 24px;
  margin-bottom: 12px;
  color: #fff;
  box-shadow: 0 4px 24px rgba(15, 118, 110, 0.22);
}
.bm-header-inner {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
}
.bm-brand { display: flex; align-items: center; gap: 14px; min-width: 0; }
.bm-brand-icon {
  flex-shrink: 0;
  width: 48px; height: 48px;
  border-radius: 12px;
  background: rgba(255,255,255,0.15);
  border: 1px solid rgba(255,255,255,0.25);
  display: flex; align-items: center; justify-content: center;
  font-weight: 700; font-size: 15px; letter-spacing: -0.02em;
}
.bm-brand h1 {
  margin: 0;
  font-size: clamp(1.25rem, 2.5vw, 1.65rem);
  font-weight: 700;
  line-height: 1.2;
  letter-spacing: -0.02em;
}
.bm-brand p {
  margin: 4px 0 0;
  font-size: clamp(0.75rem, 1.5vw, 0.875rem);
  opacity: 0.88;
  font-weight: 400;
}
.bm-pills { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; align-items: center; }
.bm-pill {
  padding: 5px 12px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  white-space: nowrap;
}
.bm-pill-light { background: rgba(255,255,255,0.18); border: 1px solid rgba(255,255,255,0.3); }
.bm-pill-warn { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }

.bm-disclaimer {
  padding: 10px 16px;
  margin-bottom: 16px;
  background: #fffbeb;
  border: 1px solid #fcd34d;
  border-radius: 10px;
  font-size: 13px;
  color: #92400e;
  line-height: 1.5;
}

/* ── Layout grid ── */
.bm-main-row {
  display: flex !important;
  flex-wrap: nowrap !important;
  gap: 14px !important;
  width: 100% !important;
  max-width: 100% !important;
}
.bm-main-row > .column {
  flex: 1 1 0 !important;
  min-width: 0 !important;
  max-width: 50% !important;
  overflow: hidden !important;
}
.bm-results-row {
  display: flex !important;
  flex-wrap: nowrap !important;
  gap: 14px !important;
  width: 100% !important;
  margin-top: 14px !important;
}
.bm-results-row > .column {
  flex: 1 1 0 !important;
  min-width: 0 !important;
  max-width: 50% !important;
}
.bm-col-left, .bm-col-right, .bm-col-result, .bm-col-overlay {
  min-width: 0 !important;
  width: 100% !important;
}

/* ── Cards ── */
.bm-panel {
  background: #ffffff !important;
  border: 1px solid #dde3ea !important;
  border-radius: 14px !important;
  padding: 16px 18px !important;
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06) !important;
  height: 100%;
}
.bm-panel-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
  padding-bottom: 10px;
  border-bottom: 1px solid #eef2f6;
}
.bm-panel-title {
  font-size: 14px;
  font-weight: 700;
  color: #0f172a;
  margin: 0;
  letter-spacing: -0.01em;
}
.bm-panel-sub {
  font-size: 12px;
  color: #64748b;
  font-weight: 500;
}

/* ── Images ── */
.bm-xray .image-container,
.bm-overlay .image-container {
  border-radius: 10px !important;
  border: 1px solid #334155 !important;
  background: #0f172a !important;
  min-height: 220px !important;
  max-height: 340px !important;
  height: 340px !important;
  overflow: hidden !important;
  width: 100% !important;
}
.bm-xray img, .bm-overlay img,
.bm-xray .image-container img, .bm-overlay .image-container img {
  object-fit: contain !important;
  width: 100% !important;
  height: 100% !important;
  max-height: 340px !important;
  margin: 0 auto !important;
}
.bm-xray .empty, .bm-overlay .empty {
  min-height: 220px !important;
  background: #1e293b !important;
}

/* ── Button ── */
#analyze-btn {
  width: 100% !important;
  min-height: 44px !important;
  border-radius: 10px !important;
  background: linear-gradient(135deg, #0f766e, #0d9488) !important;
  border: none !important;
  color: #fff !important;
  font-weight: 600 !important;
  font-size: 14px !important;
  box-shadow: 0 2px 8px rgba(15, 118, 110, 0.35) !important;
  transition: transform 0.12s, box-shadow 0.12s !important;
}
#analyze-btn:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 14px rgba(15, 118, 110, 0.4) !important;
}

/* ── Results HTML ── */
.bm-empty {
  padding: 48px 20px;
  text-align: center;
  color: #94a3b8;
  font-size: 14px;
  border: 2px dashed #cbd5e1;
  border-radius: 12px;
  background: #f8fafc;
}
.bm-empty-icon { font-size: 2.5rem; margin-bottom: 8px; opacity: 0.6; }
.bm-result {
  border-radius: 12px;
  border: 1px solid #e2e8f0;
  overflow: hidden;
  background: #fff;
}
.bm-result-head {
  padding: 12px 16px;
  background: #f8fafc;
  border-bottom: 1px solid #e2e8f0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 8px;
}
.bm-status {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  padding: 4px 10px;
  border-radius: 6px;
}
.bm-status-ok { background: #d1fae5; color: #065f46; }
.bm-status-warn { background: #fef3c7; color: #92400e; }
.bm-status-low { background: #fee2e2; color: #991b1b; }
.bm-result-body { padding: 16px; }
.bm-answer {
  font-size: clamp(15px, 2vw, 17px);
  font-weight: 600;
  color: #0f172a;
  line-height: 1.55;
  white-space: pre-wrap;
  word-break: break-word;
}
.bm-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 8px;
  margin-top: 14px;
}
.bm-stat {
  padding: 10px 12px;
  border-radius: 8px;
  background: #f1f5f9;
  border: 1px solid #e2e8f0;
}
.bm-stat-label {
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #64748b;
  margin-bottom: 2px;
}
.bm-stat-value {
  font-size: 13px;
  font-weight: 600;
  color: #0f172a;
  word-break: break-word;
}
.bm-kg {
  margin-top: 14px;
  padding: 14px;
  border-radius: 10px;
  background: #f0fdfa;
  border: 1px solid #99f6e4;
}
.bm-kg-title {
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #0f766e;
  margin-bottom: 8px;
}
.bm-kg ul { margin: 0; padding-left: 18px; color: #134e4a; font-size: 13px; line-height: 1.6; }
.bm-kg-note { font-size: 11px; color: #64748b; margin-top: 8px; }

/* ── Tabs & chat ── */
.bm-tabs .tab-nav { border-bottom: 1px solid #e2e8f0 !important; }
.bm-tabs button.selected {
  border-color: #0f766e !important;
  color: #0f766e !important;
}
.bm-chat { min-height: 320px !important; max-height: min(45vh, 420px) !important; }

/* ── Examples ── */
.bm-examples { margin-top: 8px !important; max-width: 100% !important; overflow: hidden !important; }
.bm-examples .gallery {
  gap: 6px !important;
  max-width: 100% !important;
  flex-wrap: wrap !important;
}
.bm-examples .thumbnail-item {
  border-radius: 8px !important;
  border: 2px solid transparent !important;
  transition: border-color 0.15s !important;
  max-width: 72px !important;
  max-height: 72px !important;
}
.bm-examples .thumbnail-item:hover { border-color: #0f766e !important; }
.bm-examples .grid-wrap, .bm-examples table { max-width: 100% !important; }

/* ── Form controls ── */
.bm-col-right textarea, .bm-col-right input {
  font-size: 14px !important;
}
label.block { font-size: 13px !important; font-weight: 600 !important; color: #334155 !important; }

/* ── Responsive ── */
@media (max-width: 960px) {
  .bm-main-row, .bm-results-row {
    flex-wrap: wrap !important;
  }
  .bm-main-row > .column,
  .bm-results-row > .column {
    flex: 1 1 100% !important;
    max-width: 100% !important;
  }
}
@media (max-width: 900px) {
  .gradio-container { padding: 10px 12px 20px !important; }
  .bm-header { padding: 16px; border-radius: 12px; }
  .bm-header-inner { flex-direction: column; }
  .bm-pills { justify-content: flex-start; }
  .bm-xray .image-container, .bm-overlay .image-container {
    height: 280px !important;
    max-height: 280px !important;
  }
  .bm-stats { grid-template-columns: 1fr 1fr; }
}
@media (max-width: 560px) {
  .bm-stats { grid-template-columns: 1fr; }
  .bm-brand-icon { width: 40px; height: 40px; font-size: 13px; }
}
"""

EMPTY_RESULT_HTML = """
<div class="bm-empty">
  <div class="bm-empty-icon">🩻</div>
  <div>Tải ảnh X-quang và nhấn <strong>Phân tích</strong> để xem kết quả.</div>
</div>
"""


def _status_badge(abstained: bool, conf: float) -> tuple[str, str]:
    if abstained:
        return "Từ chối trả lời", "bm-status-warn"
    if conf >= 0.55:
        return "Đã trả lời", "bm-status-ok"
    return "Độ tin cậy thấp", "bm-status-low"


def _result_html(result: dict | None) -> str:
    if not result:
        return EMPTY_RESULT_HTML

    abstained = result.get("abstained", False)
    conf = float(result.get("confidence", 0))
    status_text, status_cls = _status_badge(abstained, conf)
    answer = html.escape(str(result.get("answer", "")))
    backend = html.escape(str(result.get("backend", "—")))
    anatomy = html.escape(str(result.get("anatomy") or "—"))
    label = html.escape(str(result.get("raw_label") or "—"))
    prompts = result.get("activated_prompts") or {}
    prompt_txt = " · ".join(
        k.upper() for k, v in [("V", prompts.get("visual")), ("T", prompts.get("textual")), ("L", prompts.get("latent"))] if v
    ) or "—"

    kg = result.get("knowledge_graph") or {}
    kg_block = ""
    if kg.get("enabled") and kg.get("treatments"):
        tx_lines = "".join(
            f"<li>{html.escape(t.get('label_vi', t.get('label_en', '')))}</li>"
            for t in kg.get("treatments", [])[:3]
        )
        kg_block = f"""
        <div class="bm-kg">
          <div class="bm-kg-title">Gợi ý điều trị · Knowledge Graph</div>
          <ul>{tx_lines}</ul>
          <div class="bm-kg-note">{html.escape(kg.get('disclaimer', ''))}</div>
        </div>
        """

    return f"""
    <div class="bm-result">
      <div class="bm-result-head">
        <span class="bm-status {status_cls}">{status_text}</span>
        <span style="font-size:12px;color:#64748b;font-weight:600;">Confidence {conf:.0%}</span>
      </div>
      <div class="bm-result-body">
        <div class="bm-answer">{answer}</div>
        <div class="bm-stats">
          <div class="bm-stat"><div class="bm-stat-label">Backend</div><div class="bm-stat-value">{backend}</div></div>
          <div class="bm-stat"><div class="bm-stat-label">Vùng giải phẫu</div><div class="bm-stat-value">{anatomy}</div></div>
          <div class="bm-stat"><div class="bm-stat-label">Nhãn</div><div class="bm-stat-value">{label}</div></div>
          <div class="bm-stat"><div class="bm-stat-label">Prompts</div><div class="bm-stat-value">{html.escape(prompt_txt)}</div></div>
        </div>
        {kg_block}
      </div>
    </div>
    """


def run_vqa(
    image,
    question,
    question_type,
    prompt_mode,
    box_x1,
    box_y1,
    box_x2,
    box_y2,
    point_x,
    point_y,
    history,
):
    history = history or []
    if image is None:
        return None, EMPTY_RESULT_HTML, history
    if not question or not str(question).strip():
        return image, EMPTY_RESULT_HTML, history

    visual_prompt = None
    if prompt_mode == "box":
        visual_prompt = {"type": "box", "coordinates": [box_x1, box_y1, box_x2, box_y2]}
    elif prompt_mode == "point":
        visual_prompt = {"type": "point", "coordinates": [point_x, point_y]}
    elif prompt_mode == "automatic":
        visual_prompt = {"type": "automatic", "coordinates": None}

    result = PREDICTOR.predict(
        image=Image.fromarray(image.astype(np.uint8)) if isinstance(image, np.ndarray) else image,
        question=question,
        question_type=question_type,
        visual_prompt=visual_prompt,
        return_explanation=True,
        session_id="gradio_session",
    )
    overlay_path = result.get("heatmap_url")
    overlay_img = Image.open(overlay_path) if overlay_path and Path(overlay_path).exists() else image
    answer_text = f"{result['answer']} (conf {result['confidence']:.0%})"
    history = list(history) + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer_text},
    ]
    return overlay_img, _result_html(result), history


def build_ui() -> tuple[gr.Blocks, gr.themes.Base]:
    examples = [str(p) for p in SAMPLE_IMAGES]
    kg_badge = "KG · Bật" if KG_ON else "KG · Tắt"

    theme = gr.themes.Base(
        primary_hue="teal",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        body_background_fill="#eef1f6",
        block_background_fill="#ffffff",
        block_border_color="#dde3ea",
        block_radius="12px",
        block_label_text_weight="600",
        input_background_fill="#f8fafc",
        button_primary_background_fill="#0f766e",
        button_primary_text_color="#ffffff",
        block_title_text_weight="600",
    )

    with gr.Blocks(title="BoneMedVQA", fill_width=True) as demo:
        gr.HTML(
            f"""
            <div class="bm-header">
              <div class="bm-header-inner">
                <div class="bm-brand">
                  <div class="bm-brand-icon">BM</div>
                  <div>
                    <h1>BoneMedVQA</h1>
                    <p>Hệ thống VQA X-quang hàm mặt · Visual + Textual + Latent Prompt</p>
                  </div>
                </div>
                <div class="bm-pills">
                  <span class="bm-pill bm-pill-light">Backend · {BACKEND}</span>
                  <span class="bm-pill bm-pill-light">{kg_badge}</span>
                  <span class="bm-pill bm-pill-warn">Nghiên cứu only</span>
                </div>
              </div>
            </div>
            <div class="bm-disclaimer">{html.escape(WARNING)}</div>
            """
        )

        # Hàng 1: Ảnh | Câu hỏi (50/50, không tràn)
        with gr.Row(equal_height=False, elem_classes=["bm-main-row"]):
            with gr.Column(scale=1, min_width=280, elem_classes=["bm-col-left"]):
                with gr.Group(elem_classes=["bm-panel"]):
                    gr.HTML(
                        '<div class="bm-panel-head">'
                        '<p class="bm-panel-title">Ảnh X-quang</p>'
                        '<span class="bm-panel-sub">Tải lên hoặc chọn mẫu</span>'
                        '</div>'
                    )
                    image = gr.Image(
                        label="X-ray",
                        type="numpy",
                        show_label=False,
                        elem_classes=["bm-xray"],
                        sources=["upload", "clipboard"],
                        height=340,
                    )
                    if examples:
                        gr.Examples(
                            examples=examples,
                            inputs=image,
                            label="Ảnh mẫu",
                            examples_per_page=3,
                        )

                with gr.Accordion("Visual Prompt (tùy chọn)", open=False):
                    prompt_mode = gr.Radio(
                        choices=["automatic", "none", "point", "box"],
                        value="automatic",
                        label="Chế độ",
                    )
                    with gr.Row():
                        point_x = gr.Number(value=128, label="Point X", precision=0)
                        point_y = gr.Number(value=128, label="Point Y", precision=0)
                    with gr.Row():
                        box_x1 = gr.Number(value=60, label="X1", precision=0)
                        box_y1 = gr.Number(value=60, label="Y1", precision=0)
                        box_x2 = gr.Number(value=200, label="X2", precision=0)
                        box_y2 = gr.Number(value=200, label="Y2", precision=0)

            with gr.Column(scale=1, min_width=280, elem_classes=["bm-col-right"]):
                with gr.Group(elem_classes=["bm-panel"]):
                    gr.HTML(
                        '<div class="bm-panel-head">'
                        '<p class="bm-panel-title">Câu hỏi</p>'
                        '<span class="bm-panel-sub">VQA đóng hoặc mở</span>'
                        '</div>'
                    )
                    question = gr.Textbox(
                        label="Câu hỏi",
                        value="Có dấu hiệu gãy xương hàm dưới không?",
                        lines=2,
                        placeholder="VD: Có dấu hiệu gãy xương hàm dưới không?",
                        show_label=False,
                    )
                    with gr.Row():
                        question_type = gr.Radio(
                            choices=["auto", "closed", "open"],
                            value="auto",
                            label="Loại câu hỏi",
                            scale=3,
                        )
                        btn = gr.Button("Phân tích", elem_id="analyze-btn", scale=2)

                    gr.Examples(
                        examples=[
                            ["Có dấu hiệu gãy xương hàm dưới không?"],
                            ["Vùng giải phẫu chính trên ảnh là gì?"],
                            ["Phương án điều trị phù hợp nhất là gì?"],
                            ["Giải thích lý do lâm sàng cho hướng xử trí."],
                        ],
                        inputs=question,
                        label="Câu hỏi nhanh",
                    )

        # Hàng 2: Kết quả | Overlay (full width, 50/50)
        with gr.Row(equal_height=False, elem_classes=["bm-results-row"]):
            with gr.Column(scale=1, min_width=280, elem_classes=["bm-col-result"]):
                with gr.Group(elem_classes=["bm-panel"]):
                    gr.HTML(
                        '<div class="bm-panel-head">'
                        '<p class="bm-panel-title">Kết quả phân tích</p>'
                        '<span class="bm-panel-sub">Trả lời + Knowledge Graph</span>'
                        '</div>'
                    )
                    result_html = gr.HTML(value=EMPTY_RESULT_HTML)
                    with gr.Tabs(elem_classes=["bm-tabs"]):
                        with gr.Tab("Hội thoại"):
                            chatbot = gr.Chatbot(
                                label="Phiên hỏi đáp",
                                height=240,
                                elem_classes=["bm-chat"],
                            )

            with gr.Column(scale=1, min_width=280, elem_classes=["bm-col-overlay"]):
                with gr.Group(elem_classes=["bm-panel"]):
                    gr.HTML(
                        '<div class="bm-panel-head">'
                        '<p class="bm-panel-title">Overlay giải thích</p>'
                        '<span class="bm-panel-sub">Mask / heatmap</span>'
                        '</div>'
                    )
                    overlay = gr.Image(
                        label="Overlay",
                        show_label=False,
                        interactive=False,
                        elem_classes=["bm-overlay"],
                        height=340,
                    )

        btn.click(
            run_vqa,
            inputs=[
                image,
                question,
                question_type,
                prompt_mode,
                box_x1,
                box_y1,
                box_x2,
                box_y2,
                point_x,
                point_y,
                chatbot,
            ],
            outputs=[overlay, result_html, chatbot],
        )

    return demo, theme


if __name__ == "__main__":
    ui, app_theme = build_ui()
    ui.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        theme=app_theme,
        css=CSS,
    )
