import os
import time
from typing import List, Dict, Any

import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from shared import genai, TokenLedger, active_cert_bundle

# === Prompt (can be overridden with --prompt) ===
DEFAULT_PROMPT = (
	"Overview of your training corpus dedicated to IBM COBOL Enterprise + DB2 + CICS + JCL technological stack"
)

# Verbosity toggle (set VERBOSE=0 for quiet mode)
VERBOSE = os.environ.get("VERBOSE", "1") not in ("0", "false", "False")

ledger = TokenLedger()


def main():
	import argparse
	parser = argparse.ArgumentParser(description="Simple Vertex AI connectivity + token accounting")
	parser.add_argument("--env-file", default=".env", help="Path to mandatory .env file (must exist)")
	parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt text to send")
	args = parser.parse_args()

	# Load env file
	load_dotenv(args.env_file)

	# Resolve LEDGER_PATH relative to this script directory if provided as a relative path.
	script_dir = pathlib.Path(__file__).resolve().parent
	ledger_env = os.environ.get("LEDGER_PATH")
	if ledger_env:
		# If not absolute, make it absolute anchored to script directory
		if not os.path.isabs(ledger_env):
			resolved = script_dir / ledger_env
			os.environ["LEDGER_PATH"] = str(resolved)
			if VERBOSE:
				print(f"[connectivity] Resolved LEDGER_PATH -> {resolved}")
	else:
		# Provide a default inside the script directory if none specified.
		default_ledger = script_dir / "token_events_ledger.jsonl"
		os.environ["LEDGER_PATH"] = str(default_ledger)
		if VERBOSE:
			print(f"[connectivity] Defaulting LEDGER_PATH -> {default_ledger}")
	active_bundle = active_cert_bundle(verbose=VERBOSE)
	if active_bundle and VERBOSE:
		print(f"[connectivity] Using custom CA bundle: {active_bundle}")

	# Mandatory environment variables (no defaults). Fail fast if missing.
	required_vars = [
		"GOOGLE_APPLICATION_CREDENTIALS",
		"PROJECT_ID",
		"LOCATION",
		"MODEL_NAME",
	]
	missing = [v for v in required_vars if not os.environ.get(v)]
	if missing:
		raise RuntimeError(f"Missing mandatory env vars in {args.env_file}: {', '.join(missing)}")
	service_account_key = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
	project_id = os.environ["PROJECT_ID"]
	location = os.environ["LOCATION"]
	model_name = os.environ["MODEL_NAME"]

	if not os.path.isfile(service_account_key):
		raise FileNotFoundError(f"Service account key not found: {service_account_key}")
	os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_key

	# Initialize google-genai client for Vertex AI
	client = genai.Client(vertexai=True, project=project_id, location=location)

	response = ledger.generate_with_accounting(client, model_name, args.prompt)
	if VERBOSE:
		print("Response:\n", response.text)

	# === Extended diagnostics: candidate count & raw usage metadata ===
	if VERBOSE:
		try:
			candidates = getattr(response, "candidates", []) or []
			print(f"\nCandidate count: {len(candidates)}")
			for idx, cand in enumerate(candidates):
				# Safely extract text parts
				parts = []
				content = getattr(cand, "content", None)
				if content and hasattr(content, "parts"):
					for p in content.parts:
						if hasattr(p, "text") and p.text:
							parts.append(p.text)
				joined = "\n".join(parts)
				preview = joined[:500].replace("\n", " ")
				print(f"--- Candidate {idx} preview (first 500 chars) ---\n{preview}\n")
		except Exception as exc:
			print("(Candidate inspection error)", exc)

	usage = getattr(response, "usage_metadata", None)
	if VERBOSE:
		print("\nRaw usage metadata dump:")
		try:
			if usage is None:
				print("<None>")
			elif hasattr(usage, "__dict__") and usage.__dict__:
				print(usage.__dict__)
			else:
				# Fallback: collect public attributes
				attrs = {k: getattr(usage, k) for k in dir(usage) if not k.startswith("_") and not callable(getattr(usage, k))}
				print(attrs)
		except Exception as exc:
			print("(Usage metadata dump error)", exc)

	summary = ledger.summarize()
	if VERBOSE:
		print("\n=== Token Usage Summary ===")
		print(f"Calls: {summary['calls']}")
		print(
			f"Total Prompt Tokens: {summary['total_prompt_tokens']} | Total Output Tokens: {summary['total_output_tokens']} | Total Tokens: {summary['total_tokens']}"
		)
		print("Detailed Events:")
		for e in summary["events"]:
			print(e)

	# Write ledger after printing summary
	ledger.write_ledger()
	if VERBOSE:
		print("\nLedger appended: token_events_ledger.jsonl (override with LEDGER_PATH env var)")
	else:
		print("Ledger written.")


if __name__ == "__main__":
	main()

