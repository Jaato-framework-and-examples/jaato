import { describe, expect, it } from "vitest";
import { normalizeClarificationQuestion } from "./clarification";

describe("normalizeClarificationQuestion", () => {
  it("reads the ClarificationBatchEvent shape the daemon actually sends (question_payload)", () => {
    const q = normalizeClarificationQuestion({
      index: 1,
      text: "What's your current experience with Python?",
      question_type: "single_choice",
      required: true,
      choices: [{ text: "None" }, { text: "Some", default: true }, { text: "A lot", expects_attachment: true }],
    });
    expect(q.question_text).toBe("What's your current experience with Python?");
    expect(q.options).toEqual(["None", "Some", "A lot"]);
    expect(q.default).toBe(2);
    expect(q.optional).toBe(false);
    expect(q.expects_attachment).toEqual([false, false, true]);
  });

  it("reads a free-text optional question", () => {
    const q = normalizeClarificationQuestion({ index: 2, text: "Anything else?", question_type: "free_text", required: false });
    expect(q).toMatchObject({ question_text: "Anything else?", question_type: "free_text", options: [], optional: true, default: null });
  });

  it("still reads the per-question ClarificationQuestionEvent vocabulary", () => {
    expect(normalizeClarificationQuestion({ question_text: "Which?", options: [{ text: "a" }, { text: "b" }] }).options).toEqual(["a", "b"]);
    expect(normalizeClarificationQuestion({ question_text: "Which?", options: ["a", "b"] })).toMatchObject({ question_text: "Which?", options: ["a", "b"] });
  });

  it("never invents a question: an unknown shape yields no text rather than a placeholder", () => {
    expect(normalizeClarificationQuestion({ index: 1 }).question_text).toBeUndefined();
    expect(normalizeClarificationQuestion(null).options).toEqual([]);
  });
});
