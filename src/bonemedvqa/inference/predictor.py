"""High-level prediction API used by scripts / FastAPI / Gradio."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from PIL import Image

from bonemedvqa.data.collators import BoneMedVQACollator
from bonemedvqa.data.transforms import build_transforms, load_image
from bonemedvqa.explainability.attention_visualizer import attention_to_heatmap
from bonemedvqa.explainability.result_renderer import render_prediction_overlay
from bonemedvqa.inference.confidence import WARNING, apply_abstention
from bonemedvqa.inference.conversation import ConversationState
from bonemedvqa.inference.vlm_backend import build_vlm_backend
from bonemedvqa.knowledge_graph.treatment_advisor import TreatmentAdvisor
from bonemedvqa.models.model_factory import build_model
from bonemedvqa.prompting.textual_prompt import TextualPromptBuilder
from bonemedvqa.prompting.visual_prompt import VisualPromptGenerator, build_segmentor
from bonemedvqa.utils.device import get_device
from bonemedvqa.utils.io import ensure_dir, load_yaml
from bonemedvqa.utils.logger import get_logger


class Predictor:
    """Singleton-friendly predictor that loads the model once.

    Inference priority (configurable via ``inference.backend`` / env ``BONEMEDVQA_BACKEND``):
      1. OpenAI Vision (if ``OPENAI_API_KEY`` set)
      2. Hugging Face BLIP-VQA pretrained
      3. Local BoneMedVQA checkpoint
    """

    def __init__(
        self,
        cfg: dict[str, Any] | str | Path,
        checkpoint: str | Path | None = None,
        device: torch.device | None = None,
        backend: str | None = None,
    ):
        if isinstance(cfg, (str, Path)):
            cfg = load_yaml(cfg)
        self.cfg = cfg
        self.device = device or get_device(cfg.get("device", {}).get("prefer_cuda", True))
        self.logger = get_logger("bonemedvqa.predictor")
        self.prompt_builder = TextualPromptBuilder(
            **{
                k: v
                for k, v in (cfg.get("model", {}).get("textual_prompt", {}) or {}).items()
                if k
                in {
                    "include_task",
                    "include_anatomy",
                    "include_abnormality",
                    "include_output_format",
                    "require_abstention_instruction",
                    "language",
                }
            }
        )
        self.visual_prompt_gen = VisualPromptGenerator(
            segmentor=build_segmentor(
                str(cfg.get("model", {}).get("visual_prompt", {}).get("segmentor", "heuristic"))
            )
        )
        self.transform = build_transforms(
            image_size=int(cfg.get("model", {}).get("visual_encoder", {}).get("image_size", 224)),
            train=False,
        )
        self.collator = BoneMedVQACollator(
            max_length=int(cfg.get("model", {}).get("text_encoder", {}).get("max_length", 64))
        )
        self.id_to_label = {0: "no", 1: "yes"}
        self.threshold = float(cfg.get("calibration", {}).get("confidence_threshold", 0.55))
        self.sessions: dict[str, ConversationState] = {}
        out_dir = cfg.get("output", {}).get("dir", "outputs")
        # Prefer absolute project-relative outputs when possible
        self.output_dir = ensure_dir(out_dir)

        kg_cfg = cfg.get("knowledge_graph", {}) or {}
        self.use_kg = bool(
            kg_cfg.get("enabled", False) or cfg.get("inference", {}).get("use_knowledge_graph", False)
        )
        self.treatment_advisor = (
            TreatmentAdvisor(language=str(kg_cfg.get("language", "vi"))) if self.use_kg else None
        )

        backend_name = (
            backend
            or os.getenv("BONEMEDVQA_BACKEND")
            or str(cfg.get("inference", {}).get("backend", "auto"))
        ).lower()
        self.vlm = None
        if backend_name != "local":
            try:
                self.vlm = build_vlm_backend(prefer=backend_name, device=str(self.device))
            except Exception as exc:
                self.logger.warning("VLM backend unavailable: %s", exc)
                self.vlm = None
        self.backend_name = getattr(self.vlm, "model", None) or (
            "blip" if self.vlm is not None and self.vlm.__class__.__name__.startswith("Blip") else "local"
        )
        if self.vlm is not None:
            self.backend_name = "openai" if self.vlm.__class__.__name__.startswith("OpenAI") else "blip"
            self.logger.info("Using VLM backend: %s", self.backend_name)
        else:
            self.backend_name = "local"
            self.logger.info("Using local BoneMedVQA checkpoint backend")

        payload = None
        if checkpoint and Path(checkpoint).exists():
            payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
            extra = payload.get("extra") or {}
            if extra.get("id_to_label"):
                self.id_to_label = {int(k): str(v) for k, v in extra["id_to_label"].items()}
            if isinstance(extra.get("cfg"), dict):
                cal = cfg.get("calibration", {})
                cfg = {**extra["cfg"], "calibration": {**extra["cfg"].get("calibration", {}), **cal}}
                self.cfg = cfg
            self.logger.info(
                "Preparing local model for checkpoint %s (%s classes)",
                checkpoint,
                len(self.id_to_label),
            )

        self.label_to_id = {v: k for k, v in self.id_to_label.items()}
        self.model = build_model(cfg, num_classes=len(self.id_to_label), id_to_label=self.id_to_label)
        self.model.to(self.device)
        self.model.eval()

        if payload is not None:
            missing, unexpected = self.model.load_state_dict(payload["model_state"], strict=False)
            if missing:
                self.logger.warning("Missing keys when loading checkpoint: %s", list(missing)[:8])
            if unexpected:
                self.logger.warning("Unexpected keys when loading checkpoint: %s", list(unexpected)[:8])
            self.logger.info("Loaded checkpoint %s", checkpoint)
        else:
            self.logger.warning("No local checkpoint loaded.")

    def _attach_kg_suggestions(self, result: dict[str, Any], question: str) -> dict[str, Any]:
        if self.treatment_advisor is None:
            result["knowledge_graph"] = {"enabled": False}
            return result
        suggestion = self.treatment_advisor.suggest(
            anatomy=result.get("anatomy"),
            abnormality=result.get("abnormality"),
            answer=result.get("raw_label") or result.get("answer"),
            question=question,
            confidence=float(result.get("confidence", 0.0)),
        )
        result["knowledge_graph"] = suggestion
        result["treatment_suggestions"] = suggestion.get("treatments", [])
        result["treatment_summary"] = suggestion.get("summary_vi") or suggestion.get("summary_en")
        return result

    def _crop_for_prompt(self, pil: Image.Image, bundle) -> Image.Image:
        if bundle is None or not bundle.box or bundle.box[2] <= bundle.box[0]:
            return pil
        x1, y1, x2, y2 = [int(v) for v in bundle.box]
        w, h = pil.size
        # pad ROI a bit
        pad = 8
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            return pil
        return pil.crop((x1, y1, x2, y2))

    @torch.no_grad()
    def predict(
        self,
        image: str | Path | Image.Image,
        question: str,
        question_type: str = "auto",
        visual_prompt: dict[str, Any] | None = None,
        return_explanation: bool = True,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        request_id = str(uuid4())
        if isinstance(image, (str, Path)):
            pil = load_image(image)
            image_id = Path(image).stem
        else:
            pil = image.convert("RGB")
            image_id = "upload"

        arr = np.array(pil)
        bundle = None
        prompt_type = "none"
        if visual_prompt:
            ptype = str(visual_prompt.get("type", "box")).lower()
            prompt_type = ptype
            if ptype == "point":
                bundle = self.visual_prompt_gen.generate_from_point(arr, visual_prompt["coordinates"])
            elif ptype == "box":
                bundle = self.visual_prompt_gen.generate_from_box(arr, visual_prompt["coordinates"])
            elif ptype == "mask" and visual_prompt.get("mask") is not None:
                bundle = self.visual_prompt_gen.generate_from_mask(arr, visual_prompt["mask"])
            elif ptype == "none":
                bundle = None
                prompt_type = "none"
            else:
                bundle = self.visual_prompt_gen.generate_automatic(arr)
                prompt_type = "automatic"
        else:
            # Default automatic ROI helps VLM focus
            bundle = self.visual_prompt_gen.generate_automatic(arr)
            prompt_type = "automatic"

        qtype = self.prompt_builder.classify_question_type(
            question, None if question_type == "auto" else question_type
        )
        anatomy = self.prompt_builder.extract_anatomy(question)

        # -------- VLM path (OpenAI / BLIP) --------
        if self.vlm is not None:
            focus = self._crop_for_prompt(pil, bundle) if prompt_type != "none" else pil
            note = None
            if bundle is not None:
                note = f"type={prompt_type}, box={bundle.box}, mask_area={int(bundle.mask.sum())}"
            vlm_out = self.vlm.predict(
                image=focus,
                question=question,
                question_type=qtype,
                visual_prompt_note=note,
            )
            if anatomy and not vlm_out.get("anatomy"):
                vlm_out["anatomy"] = anatomy

            mask_url = None
            heatmap_url = None
            if return_explanation:
                exp_dir = ensure_dir(Path(self.output_dir) / "figures" / "explanations")
                heat = bundle.mask.astype(np.float32) if bundle is not None else None
                render_prediction_overlay(
                    pil,
                    mask=bundle.mask if bundle else None,
                    box=bundle.box if bundle else None,
                    heatmap=heat,
                    out_path=exp_dir / f"{request_id}_overlay.png",
                )
                heatmap_url = str(exp_dir / f"{request_id}_overlay.png")
                if bundle is not None:
                    mask_path = exp_dir / f"{request_id}_mask.png"
                    Image.fromarray((bundle.mask * 255).astype(np.uint8)).save(mask_path)
                    mask_url = str(mask_path)

            result = {
                "request_id": request_id,
                "image_id": image_id,
                "answer": vlm_out["answer"],
                "answer_type": vlm_out.get("answer_type", qtype),
                "raw_label": vlm_out.get("raw_label"),
                "confidence": float(vlm_out.get("confidence", 0.0)),
                "anatomy": vlm_out.get("anatomy"),
                "prompt_type": prompt_type,
                "mask_url": mask_url,
                "heatmap_url": heatmap_url,
                "abstained": bool(vlm_out.get("abstained", False)),
                "open_answer": vlm_out.get("open_answer"),
                "warning": WARNING,
                "backend": vlm_out.get("backend", self.backend_name),
                "activated_prompts": {
                    "visual": prompt_type != "none",
                    "textual": True,
                    "latent": False,
                },
            }
            if session_id is not None:
                state = self.sessions.setdefault(session_id, ConversationState(session_id=session_id))
                state.add(
                    question,
                    result["answer"],
                    result["confidence"],
                    result["abstained"],
                    anatomy=result.get("anatomy"),
                    raw_label=result.get("raw_label"),
                    backend=result.get("backend"),
                )
                result["session_id"] = state.session_id
                result["history"] = state.to_dict()["turns"]
            return self._attach_kg_suggestions(result, question)

        # -------- Local checkpoint path --------
        prompt_text = self.prompt_builder.build(question=question, question_type=qtype, anatomy=anatomy)
        tfm = self.transform(pil)
        sample = {
            "sample_id": image_id,
            "patient_id": "",
            "pixel_values": tfm["pixel_values"],
            "question": question,
            "prompt_text": prompt_text,
            "answer_text": "",
            "label": torch.tensor(-100),
            "question_type": qtype,
            "anatomy": anatomy or "",
            "abnormality": "",
            "bbox": bundle.box if bundle else None,
            "mask_path": None,
            "resize_meta": tfm["resize_meta"],
            "image_path": image_id,
        }
        batch = self.collator([sample])
        if bundle is not None and self.cfg.get("model", {}).get("use_visual_prompt", False):
            prompt_img = Image.fromarray(bundle.masked_image)
            prompt_tfm = self.transform(prompt_img)
            batch["prompt_pixel_values"] = prompt_tfm["pixel_values"].unsqueeze(0)
        batch["anatomies"] = [anatomy or ""]
        batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        outputs = self.model(batch)
        pred_id = int(outputs["pred"][0].item())
        conf = float(outputs["confidence"][0].item())
        label = self.id_to_label.get(pred_id, str(pred_id))
        # For fracture yes/no questions, map non yes/no labels to abstain rather than nonsense anatomy labels
        is_fracture_q = "fracture" in question.lower() or "abnormal" in question.lower()
        if is_fracture_q and label not in {"yes", "no"}:
            abstain_pack = apply_abstention(label, min(conf, 0.4), threshold=self.threshold)
        else:
            abstain_pack = apply_abstention(label, conf, threshold=self.threshold)
        answer = abstain_pack["answer"]
        abstained = abstain_pack["abstained"]

        open_text = None
        if outputs.get("open") is not None:
            open_text = outputs["open"]["texts"][0]
            if qtype == "open":
                answer = open_text if not abstained else abstain_pack["answer"]

        mask_url = None
        heatmap_url = None
        if return_explanation:
            exp_dir = ensure_dir(Path(self.output_dir) / "figures" / "explanations")
            heat = None
            if outputs.get("attention_weights") is not None:
                heat = attention_to_heatmap(outputs["attention_weights"][0])
            elif bundle is not None:
                heat = bundle.mask.astype(np.float32)
            render_prediction_overlay(
                pil,
                mask=bundle.mask if bundle else None,
                box=bundle.box if bundle else None,
                heatmap=heat,
                out_path=exp_dir / f"{request_id}_overlay.png",
            )
            heatmap_url = str(exp_dir / f"{request_id}_overlay.png")
            if bundle is not None:
                mask_path = exp_dir / f"{request_id}_mask.png"
                Image.fromarray((bundle.mask * 255).astype(np.uint8)).save(mask_path)
                mask_url = str(mask_path)

        result = {
            "request_id": request_id,
            "image_id": image_id,
            "answer": answer,
            "answer_type": qtype,
            "raw_label": label,
            "confidence": conf,
            "anatomy": anatomy,
            "prompt_type": prompt_type,
            "mask_url": mask_url,
            "heatmap_url": heatmap_url,
            "abstained": abstained,
            "open_answer": open_text,
            "warning": WARNING,
            "backend": "local",
            "activated_prompts": {
                "visual": bool(self.cfg.get("model", {}).get("use_visual_prompt", False)),
                "textual": bool(self.cfg.get("model", {}).get("use_textual_prompt", True)),
                "latent": bool(self.cfg.get("model", {}).get("use_latent_prompt", False)),
            },
        }

        if session_id is not None:
            state = self.sessions.setdefault(session_id, ConversationState(session_id=session_id))
            state.add(question, answer, conf, abstained, anatomy=anatomy, raw_label=label, backend="local")
            result["session_id"] = state.session_id
            result["history"] = state.to_dict()["turns"]

        return self._attach_kg_suggestions(result, question)
