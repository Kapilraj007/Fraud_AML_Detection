import { describe, expect, it } from "vitest";

import { getStudyArmPresentation, isStudyArm, STUDY_ARMS } from "./studyArms";

describe("case study-arm presentation", () => {
  it("recognizes every fixed server assignment", () => {
    expect(STUDY_ARMS.every(isStudyArm)).toBe(true);
    expect(isStudyArm("analyst_selected")).toBe(false);
  });

  it("reveals only the evidence allowed for each assignment", () => {
    expect(getStudyArmPresentation("data_only")).toMatchObject({
      showScore: false,
      showExplanation: false,
      applyReliabilityGate: false,
    });
    expect(getStudyArmPresentation("score_only")).toMatchObject({
      showScore: true,
      showExplanation: false,
    });
    expect(getStudyArmPresentation("generic_explanation")).toMatchObject({
      showScore: true,
      showExplanation: true,
      applyReliabilityGate: false,
    });
    expect(getStudyArmPresentation("reliability_gated_case_context")).toMatchObject({
      showScore: true,
      showExplanation: true,
      applyReliabilityGate: true,
    });
  });

  it("fails closed for an unknown assignment", () => {
    expect(getStudyArmPresentation("future_arm")).toMatchObject({
      showScore: false,
      showExplanation: false,
    });
  });
});

