import type { CasePrediction, Explanation } from "../../lib/types";

export const STUDY_ARMS = [
  "data_only",
  "score_only",
  "generic_explanation",
  "reliability_gated_case_context",
] as const;

export type StudyArm = (typeof STUDY_ARMS)[number];

export interface StudyArmPresentation {
  label: string;
  description: string;
  showScore: boolean;
  showExplanation: boolean;
  applyReliabilityGate: boolean;
}

const PRESENTATIONS: Record<StudyArm, StudyArmPresentation> = {
  data_only: {
    label: "Transaction evidence",
    description: "Review the transaction record and account context.",
    showScore: false,
    showExplanation: false,
    applyReliabilityGate: false,
  },
  score_only: {
    label: "Risk score",
    description: "Review transaction evidence with the model risk signals.",
    showScore: true,
    showExplanation: false,
    applyReliabilityGate: false,
  },
  generic_explanation: {
    label: "Risk score and key factors",
    description: "Review risk signals with the model's leading feature contributions.",
    showScore: true,
    showExplanation: true,
    applyReliabilityGate: false,
  },
  reliability_gated_case_context: {
    label: "Reliability-aware case context",
    description: "Review model context only when its reliability checks pass.",
    showScore: true,
    showExplanation: true,
    applyReliabilityGate: true,
  },
};

export function isStudyArm(value: string): value is StudyArm {
  return STUDY_ARMS.includes(value as StudyArm);
}

export function getStudyArmPresentation(value: string): StudyArmPresentation {
  if (isStudyArm(value)) {
    return PRESENTATIONS[value];
  }

  // Unknown server-assigned arms get the most conservative presentation.
  return PRESENTATIONS.data_only;
}

export function getPrimaryPrediction(
  predictions: CasePrediction[] | undefined,
): CasePrediction | undefined {
  return predictions?.[0];
}

export function getPredictionExplanation(
  prediction: CasePrediction | undefined,
): Explanation | undefined {
  if (!prediction) return undefined;

  const reliable = prediction.reliability_status === "reliable";
  if (prediction.explanation) {
    return {
      ...prediction.explanation,
      reliable: prediction.explanation.reliable ?? reliable,
      reliability_reasons: prediction.explanation.reliability_reasons ??
        (prediction.explanation_gate_reason ? [prediction.explanation_gate_reason] : []),
    };
  }

  if (!reliable || prediction.explanation_gate_reason) {
    return {
      top_features: [],
      narrative: null,
      reliable: false,
      reliability_reasons: prediction.explanation_gate_reason
        ? [prediction.explanation_gate_reason]
        : [],
      is_withheld: true,
      gate_reason: prediction.explanation_gate_reason,
    };
  }

  return undefined;
}
