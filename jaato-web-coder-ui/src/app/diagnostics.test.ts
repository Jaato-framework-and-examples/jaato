/**
 * The Diagnostics rail's data module (#1294).
 *
 * Pinned here, each a way the section could mislead:
 *   • a failed check is REPORTED and keeps the last answer it had -- a
 *     blanked panel reads as "not confined";
 *   • an answer for a session this client has left is dropped;
 *   • a daemon below 1.25 is "unsupported", not empty;
 *   • the live ``probe`` and the cached record fields are kept SEPARATE --
 *     a probe that could not determine an answer never turns into a
 *     guessed ``enforced`` / ``confined`` in the summary;
 *   • an owner-gate refusal is shown in words and nothing else changes.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useJaato } from "@/store/store";

const getDiagnostics = vi.fn();
let protocol = "1.25";
let connected = true;

vi.mock("@/sdk/connection", () => ({
  isConnected: () => connected,
  getClient: () => ({
    get serverProtocolVersion() { return protocol; },
    getDiagnostics,
  }),
}));

const diag = await import("@/app/diagnostics");

beforeEach(() => {
  connected = false;
  useJaato.getState().resetSessionState();
  useJaato.setState({ sessionId: "s1" });
  getDiagnostics.mockReset();
  protocol = "1.25";
  connected = true;
});

describe("the check", () => {
  it("loads the record and the live probe, kept apart", async () => {
    getDiagnostics.mockResolvedValue({
      ok: true,
      runner_identity: { runner_pid: 4242, pool_served: true },
      confinement_id: "jaato-ws-mine-abc",
      sandbox_mode: "apparmor",
      consumption: { totals: { usd: 0.42 } },
      notebook_boundary_kind: "apparmor",
      protocol_version: "1.25",
      server_version: "0.30.0",
      probe: {
        ok: true, error: "", expected_profile: "jaato-ws-mine-abc",
        current_profile: "jaato-ws-mine-abc", current_mode: "enforce",
        enforced: true, confined: true,
        scan: { scanned: 5, matched: 5, divergent: 0, unreadable: 0, gone: 0, uniform: true, route: "task_dir", divergent_threads: [], unreadable_threads: [] },
      },
    });
    await diag.refreshDiagnostics();
    const d = useJaato.getState().diagnostics;
    expect(d.status).toBe("loaded");
    expect(d.confinementId).toBe("jaato-ws-mine-abc");
    expect(d.sandboxMode).toBe("apparmor");
    expect(d.probe?.enforced).toBe(true);
    expect(d.checkedAt).not.toBeNull();
    expect(diag.diagnosticsSummary(d)).toBe("enforced");
  });

  it("a cached record can disagree with a fresh probe -- both survive, distinctly (#1253)", async () => {
    getDiagnostics.mockResolvedValue({
      ok: true,
      runner_identity: null,
      confinement_id: "jaato-ws-mine-abc",
      // The record CLAIMS confined -- exactly the #1253 shape.
      sandbox_mode: "apparmor",
      protocol_version: "1.25",
      server_version: "0.30.0",
      probe: {
        ok: true, error: "", expected_profile: "jaato-ws-mine-abc",
        current_profile: "unconfined", current_mode: null,
        enforced: false, confined: false, scan: null,
      },
    });
    await diag.refreshDiagnostics();
    const d = useJaato.getState().diagnostics;
    // The cached field is untouched by the live answer...
    expect(d.sandboxMode).toBe("apparmor");
    // ...and the live probe says the opposite, without one overwriting the other.
    expect(d.probe?.enforced).toBe(false);
    expect(diag.diagnosticsSummary(d)).toBe("not confined");
  });

  it("a failed probe reports itself, never a guessed true or false", async () => {
    getDiagnostics.mockResolvedValue({
      ok: true, confinement_id: "", sandbox_mode: null, protocol_version: "1.25", server_version: "0.30.0",
      probe: { ok: false, error: "thread scan failed: OSError", expected_profile: "x", current_profile: "", current_mode: null, enforced: false, confined: false, scan: null },
    });
    await diag.refreshDiagnostics();
    const d = useJaato.getState().diagnostics;
    expect(d.probe?.ok).toBe(false);
    expect(diag.diagnosticsSummary(d)).toBe("could not check");
  });

  it("no runner to probe: the record still loads, probe is null", async () => {
    getDiagnostics.mockResolvedValue({
      ok: true, confinement_id: "", sandbox_mode: null, protocol_version: "1.25", server_version: "0.30.0", probe: null,
    });
    await diag.refreshDiagnostics();
    const d = useJaato.getState().diagnostics;
    expect(d.status).toBe("loaded");
    expect(d.probe).toBeNull();
    expect(d.checkedAt).not.toBeNull();
  });

  it("an owner-gate refusal is shown in words and nothing else changes", async () => {
    getDiagnostics.mockResolvedValueOnce({
      ok: true, confinement_id: "x", sandbox_mode: "apparmor", protocol_version: "1.25", server_version: "0.30.0",
      probe: { ok: true, error: "", expected_profile: "x", current_profile: "x", current_mode: "enforce", enforced: true, confined: true, scan: null },
    });
    await diag.refreshDiagnostics();
    getDiagnostics.mockResolvedValueOnce({ ok: false, category: "not_owner", error: "no" });
    await diag.refreshDiagnostics();
    const d = useJaato.getState().diagnostics;
    expect(d.status).toBe("error");
    expect(d.error).toContain("Only the owner");
    expect(d.category).toBe("not_owner");
    // The prior loaded answer is NOT wiped by the refusal.
    expect(d.confinementId).toBe("x");
  });

  it("an answer for a session this client has left is dropped", async () => {
    let resolve!: (v: unknown) => void;
    getDiagnostics.mockReturnValue(new Promise((r) => { resolve = r; }));
    const asked = diag.refreshDiagnostics();
    connected = false;
    useJaato.getState().resetSessionState();
    useJaato.setState({ sessionId: "s2" });
    connected = true;
    resolve({ ok: true, confinement_id: "stale", sandbox_mode: "apparmor", protocol_version: "1.25", server_version: "0.30.0", probe: null });
    await asked;
    expect(useJaato.getState().diagnostics.confinementId).toBe("");
    expect(useJaato.getState().diagnostics.status).toBe("idle");
  });

  it("a daemon below 1.25 is unsupported, and is never asked", async () => {
    protocol = "1.24";
    await diag.refreshDiagnostics();
    expect(useJaato.getState().diagnostics.status).toBe("unsupported");
    expect(getDiagnostics).not.toHaveBeenCalled();
    expect(diag.diagnosticsSummary(useJaato.getState().diagnostics)).toBeNull();
  });

  it("nothing has been checked yet: summary is null, not a guessed verdict", () => {
    expect(diag.diagnosticsSummary(useJaato.getState().diagnostics)).toBeNull();
  });
});
