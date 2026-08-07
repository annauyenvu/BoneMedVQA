"""In-memory Knowledge Graph store for maxillofacial VQA."""

from __future__ import annotations

from bonemedvqa.knowledge_graph.ontology import KGEdge, KGNode, TreatmentPathway


class MaxillofacialKnowledgeGraph:
    """Clinical pathway graph for jaw/skull X-ray findings and treatments."""

    def __init__(self) -> None:
        self.nodes: dict[str, KGNode] = {}
        self.edges: list[KGEdge] = []
        self._build_default_ontology()

    def add_node(self, node: KGNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: KGEdge) -> None:
        self.edges.append(edge)

    def neighbors(self, node_id: str, relation: str | None = None) -> list[KGEdge]:
        out = [e for e in self.edges if e.source_id == node_id]
        if relation:
            out = [e for e in out if e.relation == relation]
        return out

    def get_node(self, node_id: str) -> KGNode | None:
        return self.nodes.get(node_id)

    def _build_default_ontology(self) -> None:
        """Populate maxillofacial ontology (research reference pathways)."""
        nodes = [
            # Anatomy
            KGNode("anat_mandible", "anatomy", "Mandible", "Xương hàm dưới"),
            KGNode("anat_maxilla", "anatomy", "Maxilla", "Xương hàm trên"),
            KGNode("anat_tmj", "anatomy", "Temporomandibular joint", "Khớp thái dương hàm"),
            KGNode("anat_zygoma", "anatomy", "Zygomatic bone", "Xương gò má"),
            KGNode("anat_dental", "anatomy", "Dental arch", "Cung răng"),
            # Findings
            KGNode("find_mandible_fracture", "finding", "Mandible fracture", "Gãy xương hàm dưới"),
            KGNode("find_maxilla_fracture", "finding", "Maxilla fracture", "Gãy xương hàm trên"),
            KGNode("find_tmj_dislocation", "finding", "TMJ dislocation", "Trật khớp thái dương hàm"),
            KGNode("find_zygoma_fracture", "finding", "Zygomatic fracture", "Gãy xương gò má"),
            KGNode("find_impacted_tooth", "finding", "Impacted tooth", "Răng mọc lệch/ngầm"),
            KGNode("find_periapical_lesion", "finding", "Periapical lesion", "Tổn thương quanh chóp răng"),
            KGNode("find_normal", "finding", "No acute abnormality", "Không bất thường cấp tính"),
            # Conditions
            KGNode("cond_mandible_fracture", "condition", "Mandibular fracture", "Gãy hàm dưới"),
            KGNode("cond_lefort", "condition", "Midface / Le Fort fracture", "Gãy giữa mặt / Le Fort"),
            KGNode("cond_tmj_dislocation", "condition", "TMJ dislocation", "Trật khớp TMJ"),
            KGNode("cond_zygoma_fracture", "condition", "Zygomatic complex fracture", "Gãy phức hợp gò má"),
            KGNode("cond_impacted_tooth", "condition", "Impacted third molar", "Răng khôn mọc ngầm"),
            KGNode("cond_periapical_abscess", "condition", "Periapical abscess", "Áp xe quanh chóp răng"),
            KGNode("cond_normal", "condition", "Normal study", "Hình ảnh bình thường"),
            # Treatments
            KGNode(
                "tx_closed_reduction",
                "treatment",
                "Closed reduction + IMF",
                "Nắn kín + cố định hàm (IMF)",
                "Intermaxillary fixation for stable non-displaced fractures.",
            ),
            KGNode(
                "tx_open_reduction",
                "treatment",
                "Open reduction internal fixation (ORIF)",
                "Mổ mở nắn xương + cố định nội tại (ORIF)",
                "For displaced/unstable mandible or midface fractures.",
            ),
            KGNode(
                "tx_tmj_reduction",
                "treatment",
                "Manual TMJ reduction",
                "Nắn trật khớp TMJ",
                "Acute anterior dislocation reduction under sedation if needed.",
            ),
            KGNode(
                "tx_soft_diet",
                "treatment",
                "Soft diet + analgesia",
                "Ăn mềm + giảm đau",
                "Conservative management for minor/stable findings.",
            ),
            KGNode(
                "tx_surgical_extraction",
                "treatment",
                "Surgical extraction",
                "Nhổ răng phẫu thuật",
                "For symptomatic impacted teeth or odontogenic infection source.",
            ),
            KGNode(
                "tx_endodontic",
                "treatment",
                "Endodontic therapy / drainage",
                "Điều trị tủy / dẫn lưu",
                "For periapical abscess with pulpal source.",
            ),
            KGNode(
                "tx_refer_omfs",
                "treatment",
                "Refer to oral & maxillofacial surgery",
                "Chuyển phẫu thuật hàm mặt",
                "Specialist referral for complex fractures or surgical planning.",
            ),
            KGNode(
                "tx_observation",
                "treatment",
                "Observation",
                "Theo dõi",
                "No acute intervention; routine follow-up if symptomatic.",
            ),
            # Follow-up
            KGNode("fu_1w", "followup", "Review in 1 week", "Tái khám 1 tuần"),
            KGNode("fu_4w", "followup", "Review in 4 weeks", "Tái khám 4 tuần"),
            KGNode("fu_imaging", "followup", "Repeat imaging if symptoms persist", "Chụp lại nếu triệu chứng kéo dài"),
            # Contraindications
            KGNode(
                "ci_self_reduction",
                "contraindication",
                "Avoid unsupervised self-reduction",
                "Không tự nắn trật khớp tại nhà",
            ),
        ]
        for n in nodes:
            self.add_node(n)

        edges = [
            # Finding → Condition
            KGEdge("find_mandible_fracture", "cond_mandible_fracture", "INDICATES"),
            KGEdge("find_maxilla_fracture", "cond_lefort", "INDICATES"),
            KGEdge("find_tmj_dislocation", "cond_tmj_dislocation", "INDICATES"),
            KGEdge("find_zygoma_fracture", "cond_zygoma_fracture", "INDICATES"),
            KGEdge("find_impacted_tooth", "cond_impacted_tooth", "INDICATES"),
            KGEdge("find_periapical_lesion", "cond_periapical_abscess", "INDICATES"),
            KGEdge("find_normal", "cond_normal", "INDICATES"),
            # Condition → Treatment
            KGEdge("cond_mandible_fracture", "tx_closed_reduction", "RECOMMENDS", 0.7, "Stable/non-displaced"),
            KGEdge("cond_mandible_fracture", "tx_open_reduction", "RECOMMENDS", 0.9, "Displaced/unstable"),
            KGEdge("cond_mandible_fracture", "tx_refer_omfs", "RECOMMENDS", 0.95, "Complex or comminuted"),
            KGEdge("cond_lefort", "tx_open_reduction", "RECOMMENDS", 0.95),
            KGEdge("cond_lefort", "tx_refer_omfs", "RECOMMENDS", 1.0),
            KGEdge("cond_tmj_dislocation", "tx_tmj_reduction", "RECOMMENDS", 0.95),
            KGEdge("cond_tmj_dislocation", "tx_soft_diet", "RECOMMENDS", 0.6),
            KGEdge("cond_zygoma_fracture", "tx_open_reduction", "RECOMMENDS", 0.85),
            KGEdge("cond_zygoma_fracture", "tx_refer_omfs", "RECOMMENDS", 0.9),
            KGEdge("cond_impacted_tooth", "tx_surgical_extraction", "RECOMMENDS", 0.8),
            KGEdge("cond_impacted_tooth", "tx_observation", "RECOMMENDS", 0.5, "Asymptomatic"),
            KGEdge("cond_periapical_abscess", "tx_endodontic", "RECOMMENDS", 0.9),
            KGEdge("cond_periapical_abscess", "tx_surgical_extraction", "RECOMMENDS", 0.7, "Non-restorable"),
            KGEdge("cond_normal", "tx_observation", "RECOMMENDS", 1.0),
            # Treatment → Follow-up
            KGEdge("tx_closed_reduction", "fu_1w", "REQUIRES_FOLLOWUP"),
            KGEdge("tx_closed_reduction", "fu_4w", "REQUIRES_FOLLOWUP"),
            KGEdge("tx_open_reduction", "fu_1w", "REQUIRES_FOLLOWUP"),
            KGEdge("tx_open_reduction", "fu_4w", "REQUIRES_FOLLOWUP"),
            KGEdge("tx_tmj_reduction", "fu_1w", "REQUIRES_FOLLOWUP"),
            KGEdge("tx_endodontic", "fu_1w", "REQUIRES_FOLLOWUP"),
            KGEdge("tx_observation", "fu_imaging", "REQUIRES_FOLLOWUP"),
            # Contraindications
            KGEdge("cond_tmj_dislocation", "ci_self_reduction", "CONTRAINDICATES"),
        ]
        for e in edges:
            self.add_edge(e)

    def query_pathway(
        self,
        finding_id: str,
        severity: str = "moderate",
    ) -> TreatmentPathway | None:
        """Traverse graph from finding to treatments and follow-ups."""
        finding = self.get_node(finding_id)
        if not finding:
            return None

        cond_edges = self.neighbors(finding_id, "INDICATES")
        if not cond_edges:
            return None
        cond_edge = cond_edges[0]
        condition = self.get_node(cond_edge.target_id)
        if not condition:
            return None

        tx_edges = sorted(
            self.neighbors(condition.node_id, "RECOMMENDS"),
            key=lambda e: e.weight,
            reverse=True,
        )
        treatments = []
        for te in tx_edges[:3]:
            tx = self.get_node(te.target_id)
            if tx:
                treatments.append(
                    {
                        "id": tx.node_id,
                        "label_en": tx.label_en,
                        "label_vi": tx.label_vi,
                        "rationale": te.evidence or tx.description,
                        "priority": f"{te.weight:.2f}",
                    }
                )

        followups = []
        seen_fu: set[str] = set()
        for te in tx_edges[:2]:
            for fe in self.neighbors(te.target_id, "REQUIRES_FOLLOWUP"):
                fu = self.get_node(fe.target_id)
                if fu and fu.node_id not in seen_fu:
                    followups.append(fu.label_vi)
                    seen_fu.add(fu.node_id)

        contraindications = []
        for ce in self.neighbors(condition.node_id, "CONTRAINDICATES"):
            ci = self.get_node(ce.target_id)
            if ci:
                contraindications.append(ci.label_vi)

        sev_boost = {"mild": 0.05, "moderate": 0.0, "severe": 0.1}.get(severity, 0.0)
        base_conf = tx_edges[0].weight if tx_edges else 0.5
        confidence = min(1.0, base_conf + sev_boost)

        reasoning = (
            f"Finding '{finding.label_vi}' → condition '{condition.label_vi}' "
            f"→ {len(treatments)} treatment option(s) from KG."
        )
        return TreatmentPathway(
            condition_id=condition.node_id,
            condition_label=condition.label_vi,
            treatments=treatments,
            followups=followups,
            contraindications=contraindications,
            confidence=confidence,
            reasoning=reasoning,
            references=["BoneMedVQA maxillofacial KG v1.0 (research ontology)"],
        )
