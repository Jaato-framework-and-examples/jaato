# Implementation Plan: SCLM Training Data Extraction Agent for LoRA Fine-Tuning

## Executive Summary

Build an autonomous agentic tool that connects to z/OS mainframe via zOSMF API, extracts COBOL source code with SCLM modification history, intelligently parses change logs and code markers, and automatically generates structured training datasets optimized for LoRA (Low-Rank Adaptation) fine-tuning on Vertex AI.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Agent Orchestrator                       │
│  (LangGraph/CrewAI/Custom with Vertex AI Gemini)           │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ├─────────────────────────────────────────┐
                   │                                         │
         ┌─────────▼─────────┐                   ┌──────────▼──────────┐
         │  Data Collection  │                   │  Data Processing    │
         │      Agent        │                   │      Agent          │
         └─────────┬─────────┘                   └──────────┬──────────┘
                   │                                        │
         ┌─────────▼─────────┐                   ┌──────────▼──────────┐
         │  zOSMF Connector  │                   │  SCLM Parser        │
         │  - List datasets  │                   │  - Change history   │
         │  - Read members   │                   │  - Change markers   │
         │  - Handle errors  │                   │  - Code extraction  │
         └───────────────────┘                   └──────────┬──────────┘
                                                            │
                                                  ┌─────────▼──────────┐
                                                  │ Training Data Gen  │
                                                  │ - Bug fix pairs    │
                                                  │ - Code completion  │
                                                  │ - Pattern examples │
                                                  └─────────┬──────────┘
                                                            │
                                                  ┌─────────▼──────────┐
                                                  │  GCS Uploader      │
                                                  │  - Format JSONL    │
                                                  │  - Train/val split │
                                                  │  - Upload to GCS   │
                                                  └────────────────────┘
```

---

## Component 1: Agent Orchestrator

### Purpose
Coordinate the entire workflow using an LLM-powered agent that can make decisions, handle errors, and adapt to varying data quality.

### Technology Stack
- **LLM**: Vertex AI Gemini 1.5 Pro (for reasoning and decision-making)
- **Framework**: LangGraph or Custom agent loop
- **Language**: Python 3.11+

### Agent Capabilities
1. **Planning**: Analyze SCLM structure and determine extraction strategy
2. **Error Handling**: Retry with different approaches when data quality is poor
3. **Quality Assessment**: Evaluate extracted data and decide if re-processing is needed
4. **Adaptive Parsing**: Adjust parsing strategies based on code patterns discovered

### Implementation

```python
# agent_orchestrator.py
from typing import List, Dict, Any
from vertexai.generative_models import GenerativeModel
from langgraph.graph import StateGraph, END
import json

class ExtractionState:
    """State object passed between agent nodes"""
    def __init__(self):
        self.datasets_to_process: List[str] = []
        self.current_dataset: str = None
        self.members_processed: int = 0
        self.training_examples: List[Dict] = []
        self.errors: List[str] = []
        self.quality_metrics: Dict[str, float] = {}
        self.should_retry: bool = False
        self.extraction_strategy: str = "standard"

class SCLMExtractionAgent:
    """Orchestrates the entire SCLM data extraction process"""
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        self.model = GenerativeModel("gemini-1.5-pro")
        self.project_id = project_id
        self.location = location
        
        # Build agent workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow for extraction"""
        workflow = StateGraph(ExtractionState)
        
        # Define nodes
        workflow.add_node("discover_datasets", self.discover_datasets_node)
        workflow.add_node("analyze_structure", self.analyze_structure_node)
        workflow.add_node("extract_data", self.extract_data_node)
        workflow.add_node("parse_modifications", self.parse_modifications_node)
        workflow.add_node("generate_training_data", self.generate_training_data_node)
        workflow.add_node("quality_check", self.quality_check_node)
        workflow.add_node("upload_to_gcs", self.upload_to_gcs_node)
        
        # Define edges
        workflow.set_entry_point("discover_datasets")
        workflow.add_edge("discover_datasets", "analyze_structure")
        workflow.add_edge("analyze_structure", "extract_data")
        workflow.add_edge("extract_data", "parse_modifications")
        workflow.add_edge("parse_modifications", "generate_training_data")
        workflow.add_edge("generate_training_data", "quality_check")
        
        # Conditional edge based on quality check
        workflow.add_conditional_edges(
            "quality_check",
            self.should_retry_or_finish,
            {
                "retry": "extract_data",
                "finish": "upload_to_gcs"
            }
        )
        workflow.add_edge("upload_to_gcs", END)
        
        return workflow.compile()
    
    def discover_datasets_node(self, state: ExtractionState) -> ExtractionState:
        """Use agent to discover relevant SCLM datasets"""
        prompt = f"""
        You are analyzing a mainframe SCLM repository structure.
        
        Task: Determine which datasets contain COBOL source code with modification history.
        
        Available dataset patterns typically include:
        - *.COBOL.SOURCE - Main COBOL programs
        - *.COPY.SOURCE - Copybooks
        - *.JCL.SOURCE - JCL jobs
        
        Return a JSON list of dataset patterns to search.
        """
        
        response = self.model.generate_content(prompt)
        datasets = json.loads(response.text)
        state.datasets_to_process = datasets
        return state
    
    def analyze_structure_node(self, state: ExtractionState) -> ExtractionState:
        """Analyze COBOL structure to determine parsing strategy"""
        # Sample a few programs to understand structure
        sample_code = self._get_sample_programs(state.datasets_to_process[:3])
        
        prompt = f"""
        Analyze these COBOL program samples and determine the modification log structure.
        
        Sample programs:
        {sample_code}
        
        Identify:
        1. Change history comment format (date, programmer, description format)
        2. Change marker patterns (e.g., *FIXBUG-###-I/E, *CHG###, etc.)
        3. Comment column conventions (typically column 7)
        4. Any non-standard patterns
        
        Return JSON with:
        {{
            "history_format": "description of format",
            "marker_pattern": "regex pattern",
            "extraction_strategy": "standard|custom|hybrid",
            "special_notes": ["any unusual patterns found"]
        }}
        """
        
        response = self.model.generate_content(prompt)
        analysis = json.loads(response.text)
        state.extraction_strategy = analysis["extraction_strategy"]
        return state
    
    def extract_data_node(self, state: ExtractionState) -> ExtractionState:
        """Extract source code from mainframe"""
        # Implementation in Component 2
        pass
    
    def parse_modifications_node(self, state: ExtractionState) -> ExtractionState:
        """Parse modification history using agent"""
        # Implementation in Component 3
        pass
    
    def generate_training_data_node(self, state: ExtractionState) -> ExtractionState:
        """Generate training examples using agent"""
        # Implementation in Component 4
        pass
    
    def quality_check_node(self, state: ExtractionState) -> ExtractionState:
        """Use agent to assess data quality"""
        prompt = f"""
        Evaluate the quality of extracted training data.
        
        Metrics:
        - Total examples: {len(state.training_examples)}
        - Errors encountered: {len(state.errors)}
        - Examples by type: {self._count_by_type(state.training_examples)}
        
        Sample examples:
        {json.dumps(state.training_examples[:5], indent=2)}
        
        Assess:
        1. Is the data quality sufficient for training?
        2. Are there systematic parsing errors?
        3. Should we retry with a different strategy?
        4. What percentage of examples are high quality?
        
        Return JSON:
        {{
            "quality_score": 0.0-1.0,
            "should_retry": true|false,
            "recommended_strategy": "if retry needed",
            "issues": ["list of issues found"]
        }}
        """
        
        response = self.model.generate_content(prompt)
        quality = json.loads(response.text)
        
        state.quality_metrics = quality
        state.should_retry = quality["should_retry"]
        return state
    
    def should_retry_or_finish(self, state: ExtractionState) -> str:
        """Decide whether to retry or finish"""
        if state.should_retry and state.errors:
            return "retry"
        return "finish"
    
    def upload_to_gcs_node(self, state: ExtractionState) -> ExtractionState:
        """Upload training data to GCS"""
        # Implementation in Component 5
        pass
    
    def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent workflow"""
        initial_state = ExtractionState()
        final_state = self.workflow.invoke(initial_state, config)
        
        return {
            "examples_generated": len(final_state.training_examples),
            "quality_metrics": final_state.quality_metrics,
            "errors": final_state.errors
        }
```

---

## Component 2: zOSMF Data Collection Agent

### Purpose
Intelligently connect to mainframe, discover COBOL programs, and handle connection/authentication issues autonomously.

### Implementation

```python
# zosmf_collector.py
import requests
import base64
from typing import List, Dict, Optional
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class ZOSMFCollectionAgent:
    """Agent for collecting data from z/OS via zOSMF API"""
    
    def __init__(self, host: str, username: str, password: str, 
                 gemini_model: GenerativeModel):
        self.base_url = f"https://{host}/zosmf"
        self.gemini = gemini_model
        
        # Setup session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        
        # Authentication
        auth_string = f"{username}:{password}"
        auth_bytes = auth_string.encode('ascii')
        auth_base64 = base64.b64encode(auth_bytes).decode('ascii')
        
        self.session.headers.update({
            'Authorization': f'Basic {auth_base64}',
            'Content-Type': 'application/json',
            'X-CSRF-ZOSMF-HEADER': 'true'
        })
    
    def discover_datasets_with_agent(self, hlq_pattern: str) -> List[Dict]:
        """Use agent to intelligently discover relevant datasets"""
        
        # First, get raw dataset list
        raw_datasets = self._list_datasets_raw(hlq_pattern)
        
        # Use agent to filter and prioritize
        prompt = f"""
        You are analyzing mainframe datasets to find COBOL source code with 
        modification history.
        
        Available datasets:
        {json.dumps(raw_datasets, indent=2)}
        
        Filter and prioritize datasets that:
        1. Contain COBOL source code (*.COBOL, *.SOURCE, *.CBL patterns)
        2. Are likely to have modification tracking (SCLM-managed)
        3. Exclude test, backup, or temporary datasets
        
        Return JSON array of dataset names to process, ordered by priority:
        {{
            "datasets": [
                {{"name": "PROD.COBOL.SOURCE", "priority": "high", "reason": "..."}},
                ...
            ]
        }}
        """
        
        response = self.gemini.generate_content(prompt)
        filtered = json.loads(response.text)
        
        return filtered["datasets"]
    
    def _list_datasets_raw(self, hlq: str) -> List[Dict]:
        """Raw dataset listing via zOSMF API"""
        url = f"{self.base_url}/restfiles/ds"
        params = {'dslevel': hlq}
        
        try:
            response = self.session.get(url, params=params, verify=False, timeout=30)
            response.raise_for_status()
            return response.json().get('items', [])
        except requests.exceptions.RequestException as e:
            print(f"Error listing datasets: {e}")
            return []
    
    def get_dataset_members_with_sampling(self, dataset: str) -> List[str]:
        """Get members with intelligent sampling for large datasets"""
        url = f"{self.base_url}/restfiles/ds/{dataset}/member"
        
        try:
            response = self.session.get(url, verify=False, timeout=30)
            response.raise_for_status()
            
            members = response.json().get('items', [])
            member_names = [m['member'] for m in members]
            
            # If dataset is very large, use agent to sample intelligently
            if len(member_names) > 1000:
                return self._intelligent_sampling(dataset, member_names)
            
            return member_names
            
        except requests.exceptions.RequestException as e:
            print(f"Error getting members from {dataset}: {e}")
            return []
    
    def _intelligent_sampling(self, dataset: str, all_members: List[str]) -> List[str]:
        """Use agent to select representative sample of members"""
        prompt = f"""
        Dataset {dataset} has {len(all_members)} members.
        
        Sample member names:
        {json.dumps(all_members[:50], indent=2)}
        
        Create a sampling strategy to select ~500 representative members that:
        1. Cover different naming patterns
        2. Include a mix of old and potentially new programs
        3. Avoid obvious test/temporary programs
        
        Return JSON:
        {{
            "strategy": "description",
            "sample_size": 500,
            "selection_criteria": ["list of criteria"],
            "sampled_members": ["member1", "member2", ...]
        }}
        
        If you can't determine selection from names alone, return a random sample.
        """
        
        response = self.gemini.generate_content(prompt)
        sampling = json.loads(response.text)
        
        return sampling["sampled_members"]
    
    def read_member_with_retry(self, dataset: str, member: str) -> Optional[str]:
        """Read member source with intelligent retry logic"""
        url = f"{self.base_url}/restfiles/ds/{dataset}({member})"
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, verify=False, timeout=60)
                response.raise_for_status()
                return response.text
                
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Timeout reading {member}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Failed to read {member} after {max_retries} attempts")
                    return None
                    
            except requests.exceptions.RequestException as e:
                print(f"Error reading {dataset}({member}): {e}")
                return None
        
        return None
    
    def batch_read_members(self, dataset: str, members: List[str], 
                          batch_size: int = 10) -> Dict[str, str]:
        """Read multiple members with rate limiting"""
        results = {}
        
        for i in range(0, len(members), batch_size):
            batch = members[i:i+batch_size]
            
            for member in batch:
                content = self.read_member_with_retry(dataset, member)
                if content:
                    results[member] = content
            
            # Rate limiting
            time.sleep(0.5)
        
        return results
```

---

## Component 3: SCLM Modification Parser Agent

### Purpose
Intelligently parse various SCLM modification log formats, handle non-standard patterns, and extract structured change information.

### Implementation

```python
# sclm_parser.py
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from vertexai.generative_models import GenerativeModel

@dataclass
class ChangeEntry:
    """Structured change log entry"""
    date: datetime
    programmer: str
    description: str
    change_type: str  # bug_fix, enhancement, performance, security
    change_id: Optional[str] = None

@dataclass
class MarkedChange:
    """Code section marked with change identifiers"""
    change_id: str
    code: str
    start_line: int
    end_line: int
    change_type: str

class SCLMParserAgent:
    """Agent for intelligently parsing SCLM modification logs"""
    
    def __init__(self, gemini_model: GenerativeModel):
        self.gemini = gemini_model
        self.learned_patterns = {}  # Cache learned patterns
    
    def parse_modification_history(self, source_code: str, 
                                   program_name: str) -> List[ChangeEntry]:
        """Parse modification history from COBOL comments"""
        
        # Try standard patterns first
        entries = self._try_standard_patterns(source_code)
        
        # If standard patterns fail or return poor results, use agent
        if not entries or len(entries) < 2:
            entries = self._agent_parse_history(source_code, program_name)
        
        return entries
    
    def _try_standard_patterns(self, source_code: str) -> List[ChangeEntry]:
        """Try common SCLM modification log patterns"""
        entries = []
        
        # Pattern 1: * DATE       PROGRAMMER    DESCRIPTION
        pattern1 = r'^\s*\*\s+(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})\s+(\w+)\s+(.+?)$'
        
        # Pattern 2: *  MM/DD/YY  INITIALS  DESCRIPTION
        pattern2 = r'^\s*\*\s+(\d{2}/\d{2}/\d{2})\s+([A-Z]{2,4})\s+(.+?)$'
        
        # Pattern 3: * YYYYMMDD PROGRAMMER - DESCRIPTION
        pattern3 = r'^\s*\*\s+(\d{8})\s+(\w+)\s+-\s+(.+?)$'
        
        for line in source_code.split('\n'):
            # Try each pattern
            for pattern in [pattern1, pattern2, pattern3]:
                match = re.match(pattern, line)
                if match:
                    date_str, programmer, description = match.groups()
                    
                    # Parse date
                    try:
                        date = self._parse_date_flexible(date_str)
                        change_type = self._classify_change_type(description)
                        
                        entries.append(ChangeEntry(
                            date=date,
                            programmer=programmer,
                            description=description.strip(),
                            change_type=change_type
                        ))
                        break
                    except:
                        continue
        
        return entries
    
    def _agent_parse_history(self, source_code: str, 
                            program_name: str) -> List[ChangeEntry]:
        """Use agent to parse non-standard modification logs"""
        
        # Extract likely comment section
        comment_section = self._extract_comment_section(source_code)
        
        prompt = f"""
        You are parsing a COBOL program's modification history from comments.
        
        Program: {program_name}
        
        Comment section:
        ```
        {comment_section}
        ```
        
        Extract modification history entries. Each entry typically includes:
        - Date (various formats possible)
        - Programmer name or initials
        - Description of change
        
        Return JSON array:
        [
            {{
                "date": "YYYY-MM-DD",
                "programmer": "name or initials",
                "description": "what changed",
                "confidence": 0.0-1.0
            }},
            ...
        ]
        
        Only include entries you're confident about (confidence > 0.7).
        """
        
        response = self.gemini.generate_content(prompt)
        parsed_entries = json.loads(response.text)
        
        # Convert to ChangeEntry objects
        entries = []
        for entry in parsed_entries:
            if entry["confidence"] > 0.7:
                try:
                    date = datetime.strptime(entry["date"], "%Y-%m-%d")
                    change_type = self._classify_change_type(entry["description"])
                    
                    entries.append(ChangeEntry(
                        date=date,
                        programmer=entry["programmer"],
                        description=entry["description"],
                        change_type=change_type
                    ))
                except:
                    continue
        
        return entries
    
    def extract_marked_changes(self, source_code: str) -> List[MarkedChange]:
        """Extract code sections marked with change identifiers"""
        
        # Try standard marker patterns first
        changes = self._try_standard_markers(source_code)
        
        # If few/no markers found, use agent to discover custom patterns
        if len(changes) < 3:
            changes.extend(self._agent_find_markers(source_code))
        
        return changes
    
    def _try_standard_markers(self, source_code: str) -> List[MarkedChange]:
        """Try common change marker patterns"""
        changes = []
        
        # Pattern: *FIXBUG-001-I ... *FIXBUG-001-E
        # Also: *CHG-001-I ... *CHG-001-E
        # Also: *ENHANCEMENT-042-I ... *ENHANCEMENT-042-E
        marker_pattern = r'^\s*\*([A-Z]+-\d+)-I\s*$\n(.*?)^\s*\*\1-E\s*$'
        
        matches = re.finditer(marker_pattern, source_code, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            change_id, code = match.groups()
            
            # Determine change type from prefix
            change_type = self._infer_type_from_id(change_id)
            
            start_line = source_code[:match.start()].count('\n')
            end_line = source_code[:match.end()].count('\n')
            
            changes.append(MarkedChange(
                change_id=change_id,
                code=code.strip(),
                start_line=start_line,
                end_line=end_line,
                change_type=change_type
            ))
        
        return changes
    
    def _agent_find_markers(self, source_code: str) -> List[MarkedChange]:
        """Use agent to find non-standard change markers"""
        
        # Sample the code to find potential marker patterns
        lines = source_code.split('\n')
        sample_lines = '\n'.join(lines[:200])  # First 200 lines
        
        prompt = f"""
        Analyze this COBOL source to identify change marker patterns.
        
        Sample code:
        ```
        {sample_lines}
        ```
        
        Look for comment patterns that mark modified code sections, such as:
        - Start/end markers (e.g., *BEGIN-CHANGE-001 ... *END-CHANGE-001)
        - Change tags (e.g., *MOD: description)
        - Any other systematic way developers marked changes
        
        Return JSON:
        {{
            "marker_pattern_found": true|false,
            "pattern_type": "description",
            "start_marker_regex": "regex pattern",
            "end_marker_regex": "regex pattern",
            "examples": ["example markers from code"]
        }}
        """
        
        response = self.gemini.generate_content(prompt)
        pattern_info = json.loads(response.text)
        
        if not pattern_info["marker_pattern_found"]:
            return []
        
        # Try to extract using discovered pattern
        # This would need custom logic based on pattern type
        return []  # Simplified for brevity
    
    def _classify_change_type(self, description: str) -> str:
        """Classify change type from description"""
        desc_lower = description.lower()
        
        # Bug fixes
        if any(word in desc_lower for word in [
            'fix', 'bug', 'defect', 'error', 'abend', 'sqlcode', 
            's0c7', 's0c4', 'aica', 'correct', 'resolve'
        ]):
            return 'bug_fix'
        
        # Performance
        if any(word in desc_lower for word in [
            'performance', 'optimize', 'tuning', 'slow', 'faster', 
            'efficiency', 'speed'
        ]):
            return 'performance'
        
        # Security
        if any(word in desc_lower for word in [
            'security', 'pci', 'audit', 'vulnerability', 'breach', 
            'mask', 'encrypt', 'authentication'
        ]):
            return 'security'
        
        # Enhancement
        if any(word in desc_lower for word in [
            'add', 'new', 'implement', 'enhance', 'feature', 
            'requirement', 'capability'
        ]):
            return 'enhancement'
        
        return 'other'
    
    def _infer_type_from_id(self, change_id: str) -> str:
        """Infer change type from change ID prefix"""
        change_id_upper = change_id.upper()
        
        type_map = {
            'FIXBUG': 'bug_fix',
            'FIX': 'bug_fix',
            'BUG': 'bug_fix',
            'PERF': 'performance',
            'PERFORMANCE': 'performance',
            'SEC': 'security',
            'SECURITY': 'security',
            'ENH': 'enhancement',
            'ENHANCEMENT': 'enhancement',
            'FEATURE': 'enhancement'
        }
        
        for prefix, change_type in type_map.items():
            if change_id_upper.startswith(prefix):
                return change_type
        
        return 'other'
    
    def _extract_comment_section(self, source_code: str) -> str:
        """Extract the likely modification history comment section"""
        lines = source_code.split('\n')
        
        # Look for dense comment blocks in first 20% of file
        max_lines = min(len(lines), int(len(lines) * 0.2))
        comment_lines = []
        
        in_comment_block = False
        for i, line in enumerate(lines[:max_lines]):
            # COBOL comments start with * in column 7
            if line.startswith('      *') or line.startswith('     *'):
                in_comment_block = True
                comment_lines.append(line)
            elif in_comment_block and not line.strip():
                # Empty line might end comment block
                if len(comment_lines) > 5:  # Substantial block
                    break
                comment_lines = []
                in_comment_block = False
        
        return '\n'.join(comment_lines)
    
    def _parse_date_flexible(self, date_str: str) -> datetime:
        """Parse date from various formats"""
        formats = [
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%m/%d/%y',
            '%Y%m%d',
            '%d-%b-%Y',
            '%d-%b-%y'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except:
                continue
        
        raise ValueError(f"Could not parse date: {date_str}")
    
    def match_changes_to_history(self, marked_changes: List[MarkedChange],
                                 history: List[ChangeEntry]) -> List[Tuple[MarkedChange, ChangeEntry]]:
        """Use agent to intelligently match marked code changes to history entries"""
        
        if not marked_changes or not history:
            return []
        
        prompt = f"""
        Match code change markers to modification history entries.
        
        Marked changes:
        {json.dumps([{"id": c.change_id, "type": c.change_type, "lines": f"{c.start_line}-{c.end_line}"} 
                     for c in marked_changes], indent=2)}
        
        History entries:
        {json.dumps([{"date": e.date.isoformat(), "programmer": e.programmer, 
                      "desc": e.description[:100], "type": e.change_type} 
                     for e in history], indent=2)}
        
        Match based on:
        1. Change type similarity
        2. Temporal proximity (changes near history entry dates)
        3. Description keywords matching change ID
        
        Return JSON array of matches:
        [
            {{
                "change_id": "FIXBUG-001",
                "history_index": 0,
                "confidence": 0.0-1.0,
                "reasoning": "why matched"
            }},
            ...
        ]
        """
        
        response = self.gemini.generate_content(prompt)
        matches = json.loads(response.text)
        
        # Convert to tuple pairs
        result = []
        for match in matches:
            if match["confidence"] > 0.6:
                change = next((c for c in marked_changes if c.change_id == match["change_id"]), None)
                hist_entry = history[match["history_index"]] if match["history_index"] < len(history) else None
                
                if change and hist_entry:
                    result.append((change, hist_entry))
        
        return result
```

---

## Component 4: Training Data Generation Agent

### Purpose
Generate diverse, high-quality training examples optimized for LoRA fine-tuning, using the agent to create variations and ensure quality.

### Implementation

```python
# training_data_generator.py
from typing import List, Dict, Optional
from dataclasses import dataclass
import json
from vertexai.generative_models import GenerativeModel

@dataclass
class TrainingExample:
    """Single training example for LoRA fine-tuning"""
    task_type: str  # bug_fix, code_completion, code_review, etc.
    instruction: str  # User prompt
    response: str  # Expected model response
    metadata: Dict  # Additional context
    quality_score: float  # 0.0-1.0

class TrainingDataGenerator:
    """Agent for generating high-quality training examples"""
    
    def __init__(self, gemini_model: GenerativeModel):
        self.gemini = gemini_model
    
    def generate_from_matched_changes(self, 
                                      matched_changes: List[Tuple[MarkedChange, ChangeEntry]],
                                      source_context: str,
                                      program_name: str) -> List[TrainingExample]:
        """Generate training examples from matched code changes and history"""
        
        examples = []
        
        for change, history in matched_changes:
            # Generate bug fix example
            bug_fix_example = self._generate_bug_fix_example(
                change, history, source_context, program_name
            )
            if bug_fix_example:
                examples.append(bug_fix_example)
            
            # Generate code explanation example
            explanation_example = self._generate_explanation_example(
                change, history, source_context, program_name
            )
            if explanation_example:
                examples.append(explanation_example)
        
        return examples
    
    def _generate_bug_fix_example(self, change: MarkedChange, 
                                  history: ChangeEntry,
                                  context: str,
                                  program_name: str) -> Optional[TrainingExample]:
        """Generate a bug fix training example using agent"""
        
        # Use agent to create a realistic "before" state
        prompt = f"""
        Generate a training example for teaching an AI to fix bugs in COBOL code.
        
        Context:
        - Program: {program_name}
        - Bug description: {history.description}
        - Fixed code:
        ```cobol
        {change.code}
        ```
        
        Create a training pair:
        1. User instruction: Describe the problem and ask for a fix
        2. Assistant response: Provide the fixed code with explanation
        
        Make it realistic - the user might describe symptoms, error codes (SQLCODE, S0C7, etc.),
        or unexpected behavior.
        
        Return JSON:
        {{
            "instruction": "User's problem description and code context",
            "response": "Fixed code with clear explanation",
            "quality_indicators": {{
                "has_error_code": true|false,
                "has_context": true|false,
                "fix_is_clear": true|false
            }}
        }}
        """
        
        try:
            response = self.gemini.generate_content(prompt)
            generated = json.loads(response.text)
            
            # Calculate quality score
            quality = sum(generated["quality_indicators"].values()) / 3.0
            
            return TrainingExample(
                task_type="bug_fix",
                instruction=generated["instruction"],
                response=generated["response"],
                metadata={
                    "program": program_name,
                    "change_id": change.change_id,
                    "date": history.date.isoformat(),
                    "programmer": history.programmer,
                    "original_description": history.description
                },
                quality_score=quality
            )
        except Exception as e:
            print(f"Error generating bug fix example: {e}")
            return None
    
    def _generate_explanation_example(self, change: MarkedChange,
                                      history: ChangeEntry,
                                      context: str,
                                      program_name: str) -> Optional[TrainingExample]:
        """Generate code explanation training example"""
        
        prompt = f"""
        Generate a training example for teaching an AI to explain COBOL code changes.
        
        Code change:
        ```cobol
        {change.code}
        ```
        
        Historical context: {history.description}
        
        Create a training pair where:
        1. User asks: "Explain this code" or "What does this change do?"
        2. Assistant explains: the purpose, how it works, and why it was needed
        
        Return JSON:
        {{
            "instruction": "User asking for explanation",
            "response": "Clear, technical explanation",
            "quality_score": 0.0-1.0
        }}
        """
        
        try:
            response = self.gemini.generate_content(prompt)
            generated = json.loads(response.text)
            
            return TrainingExample(
                task_type="code_explanation",
                instruction=generated["instruction"],
                response=generated["response"],
                metadata={
                    "program": program_name,
                    "change_id": change.change_id,
                    "change_type": change.change_type
                },
                quality_score=generated["quality_score"]
            )
        except Exception as e:
            print(f"Error generating explanation example: {e}")
            return None
    
    def generate_code_completion_examples(self, source_code: str,
                                         program_name: str,
                                         num_examples: int = 10) -> List[TrainingExample]:
        """Generate code completion examples by splitting procedures"""
        
        examples = []
        paragraphs = self._extract_paragraphs(source_code)
        
        for paragraph_name, paragraph_code in paragraphs[:num_examples]:
            if len(paragraph_code.strip()) < 100:  # Skip tiny paragraphs
                continue
            
            # Generate multiple splits at different points
            for split_ratio in [0.3, 0.5, 0.7]:
                example = self._create_completion_example(
                    paragraph_name, paragraph_code, split_ratio, program_name
                )
                if example:
                    examples.append(example)
        
        return examples
    
    def _create_completion_example(self, para_name: str, para_code: str,
                                  split_ratio: float,
                                  program_name: str) -> Optional[TrainingExample]:
        """Create a single code completion example"""
        
        lines = para_code.split('\n')
        split_idx = int(len(lines) * split_ratio)
        
        if split_idx < 2 or split_idx >= len(lines) - 2:
            return None
        
        prompt_code = '\n'.join(lines[:split_idx])
        completion_code = '\n'.join(lines[split_idx:])
        
        instruction = f"Complete this COBOL paragraph:\n\n{prompt_code}"
        response = completion_code
        
        # Use agent to assess quality
        quality = self._assess_completion_quality(prompt_code, completion_code)
        
        return TrainingExample(
            task_type="code_completion",
            instruction=instruction,
            response=response,
            metadata={
                "program": program_name,
                "paragraph": para_name,
                "split_ratio": split_ratio
            },
            quality_score=quality
        )
    
    def _assess_completion_quality(self, prompt: str, completion: str) -> float:
        """Use agent to assess quality of completion example"""
        
        assessment_prompt = f"""
        Assess the quality of this code completion training example.
        
        Prompt:
        ```cobol
        {prompt}
        ```
        
        Completion:
        ```cobol
        {completion}
        ```
        
        Rate quality (0.0-1.0) based on:
        - Does the completion make sense given the prompt?
        - Is the split point natural (not mid-statement)?
        - Does it demonstrate useful COBOL patterns?
        
        Return JSON: {{"quality_score": 0.0-1.0, "reasoning": "..."}}
        """
        
        try:
            response = self.gemini.generate_content(assessment_prompt)
            result = json.loads(response.text)
            return result["quality_score"]
        except:
            return 0.5  # Default moderate quality
    
    def generate_code_review_examples(self, source_code: str,
                                      program_name: str,
                                      coding_standards: Dict[str, str]) -> List[TrainingExample]:
        """Generate code review examples using agent to find violations"""
        
        # Sample code sections
        sections = self._extract_code_sections(source_code, num_sections=10)
        
        examples = []
        for section in sections:
            example = self._generate_review_example(section, program_name, coding_standards)
            if example:
                examples.append(example)
        
        return examples
    
    def _generate_review_example(self, code_section: str,
                                 program_name: str,
                                 standards: Dict[str, str]) -> Optional[TrainingExample]:
        """Generate a code review training example"""
        
        prompt = f"""
        You are reviewing COBOL code against organizational standards.
        
        Code section:
        ```cobol
        {code_section}
        ```
        
        Coding standards:
        {json.dumps(standards, indent=2)}
        
        Identify violations and suggest improvements. Create a training example:
        
        Return JSON:
        {{
            "instruction": "Review this code against our standards: [code]",
            "response": "Issues found: [...] Recommendations: [...]",
            "has_violations": true|false,
            "quality_score": 0.0-1.0
        }}
        """
        
        try:
            response = self.gemini.generate_content(prompt)
            generated = json.loads(response.text)
            
            if not generated["has_violations"]:
                return None  # Skip examples with no issues
            
            return TrainingExample(
                task_type="code_review",
                instruction=generated["instruction"],
                response=generated["response"],
                metadata={
                    "program": program_name,
                    "standards_checked": list(standards.keys())
                },
                quality_score=generated["quality_score"]
            )
        except Exception as e:
            print(f"Error generating review example: {e}")
            return None
    
    def _extract_paragraphs(self, source_code: str) -> List[Tuple[str, str]]:
        """Extract COBOL paragraphs from PROCEDURE DIVISION"""
        
        paragraphs = []
        
        # Find PROCEDURE DIVISION
        proc_div_match = re.search(r'PROCEDURE\s+DIVISION', source_code, re.IGNORECASE)
        if not proc_div_match:
            return paragraphs
        
        proc_section = source_code[proc_div_match.end():]
        
        # Pattern for paragraph names
        para_pattern = r'^[\s]{7,11}([A-Z0-9][A-Z0-9-]*)\.'
        
        lines = proc_section.split('\n')
        current_para = None
        current_code = []
        
        for line in lines:
            match = re.match(para_pattern, line)
            
            if match:
                # New paragraph found
                if current_para and current_code:
                    paragraphs.append((current_para, '\n'.join(current_code)))
                
                current_para = match.group(1)
                current_code = [line]
            elif current_para:
                current_code.append(line)
        
        # Add last paragraph
        if current_para and current_code:
            paragraphs.append((current_para, '\n'.join(current_code)))
        
        return paragraphs
    
    def _extract_code_sections(self, source_code: str, num_sections: int = 10) -> List[str]:
        """Extract representative code sections for review"""
        
        lines = source_code.split('\n')
        section_size = 20  # 20 lines per section
        
        # Sample sections from different parts of the code
        sections = []
        step = max(1, len(lines) // num_sections)
        
        for i in range(0, len(lines) - section_size, step):
            section = '\n'.join(lines[i:i+section_size])
            sections.append(section)
            
            if len(sections) >= num_sections:
                break
        
        return sections
    
    def filter_quality_examples(self, examples: List[TrainingExample],
                                min_quality: float = 0.6) -> List[TrainingExample]:
        """Filter examples by quality score"""
        return [ex for ex in examples if ex.quality_score >= min_quality]
    
    def balance_dataset(self, examples: List[TrainingExample]) -> List[TrainingExample]:
        """Balance dataset across different task types"""
        
        # Group by task type
        by_type = {}
        for ex in examples:
            if ex.task_type not in by_type:
                by_type[ex.task_type] = []
            by_type[ex.task_type].append(ex)
        
        # Find target size (smallest group size * number of types)
        min_size = min(len(exs) for exs in by_type.values())
        
        # Sample evenly from each type
        balanced = []
        for task_type, exs in by_type.items():
            # Sort by quality and take top min_size examples
            exs_sorted = sorted(exs, key=lambda x: x.quality_score, reverse=True)
            balanced.extend(exs_sorted[:min_size])
        
        return balanced
```

---

## Component 5: Data Formatting and GCS Upload

### Purpose
Format training examples as JSONL for LoRA fine-tuning, create train/validation splits, and upload to Google Cloud Storage.

### Implementation

```python
# gcs_uploader.py
from google.cloud import storage
from typing import List, Dict, Tuple
import json
import random
from datetime import datetime

class GCSTrainingDataUploader:
    """Upload formatted training data to Google Cloud Storage"""
    
    def __init__(self, project_id: str, bucket_name: str):
        self.client = storage.Client(project=project_id)
        self.bucket = self.client.bucket(bucket_name)
    
    def format_for_vertex_ai(self, examples: List[TrainingExample]) -> List[Dict]:
        """Format examples for Vertex AI LoRA fine-tuning"""
        
        formatted = []
        
        for example in examples:
            # Vertex AI format for instruction fine-tuning
            formatted_example = {
                "messages": [
                    {
                        "role": "user",
                        "content": example.instruction
                    },
                    {
                        "role": "assistant",
                        "content": example.response
                    }
                ]
            }
            
            formatted.append(formatted_example)
        
        return formatted
    
    def create_train_val_split(self, examples: List[TrainingExample],
                               val_ratio: float = 0.1) -> Tuple[List[TrainingExample], List[TrainingExample]]:
        """Split examples into training and validation sets"""
        
        # Shuffle examples
        shuffled = examples.copy()
        random.shuffle(shuffled)
        
        # Calculate split point
        split_idx = int(len(shuffled) * (1 - val_ratio))
        
        train_examples = shuffled[:split_idx]
        val_examples = shuffled[split_idx:]
        
        return train_examples, val_examples
    
    def upload_training_data(self, train_examples: List[TrainingExample],
                            val_examples: List[TrainingExample],
                            dataset_name: str = "mainframe-lora") -> Dict[str, str]:
        """Upload training data to GCS in JSONL format"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Format examples
        train_formatted = self.format_for_vertex_ai(train_examples)
        val_formatted = self.format_for_vertex_ai(val_examples)
        
        # Create JSONL content
        train_jsonl = '\n'.join(json.dumps(ex) for ex in train_formatted)
        val_jsonl = '\n'.join(json.dumps(ex) for ex in val_formatted)
        
        # Upload to GCS
        train_path = f"training_data/{dataset_name}/{timestamp}/train.jsonl"
        val_path = f"training_data/{dataset_name}/{timestamp}/validation.jsonl"
        
        train_blob = self.bucket.blob(train_path)
        train_blob.upload_from_string(train_jsonl, content_type='application/jsonl')
        
        val_blob = self.bucket.blob(val_path)
        val_blob.upload_from_string(val_jsonl, content_type='application/jsonl')
        
        # Upload metadata
        metadata = {
            "total_examples": len(train_examples) + len(val_examples),
            "train_examples": len(train_examples),
            "val_examples": len(val_examples),
            "task_distribution": self._get_task_distribution(train_examples + val_examples),
            "timestamp": timestamp,
            "dataset_name": dataset_name
        }
        
        metadata_path = f"training_data/{dataset_name}/{timestamp}/metadata.json"
        metadata_blob = self.bucket.blob(metadata_path)
        metadata_blob.upload_from_string(json.dumps(metadata, indent=2))
        
        return {
            "train_uri": f"gs://{self.bucket.name}/{train_path}",
            "val_uri": f"gs://{self.bucket.name}/{val_path}",
            "metadata_uri": f"gs://{self.bucket.name}/{metadata_path}"
        }
    
    def _get_task_distribution(self, examples: List[TrainingExample]) -> Dict[str, int]:
        """Get distribution of examples by task type"""
        distribution = {}
        for ex in examples:
            distribution[ex.task_type] = distribution.get(ex.task_type, 0) + 1
        return distribution
```

---

## Component 6: Vertex AI LoRA Fine-Tuning Integration

### Purpose
Trigger LoRA fine-tuning jobs on Vertex AI using the uploaded training data.

### Implementation

```python
# vertex_lora_trainer.py
from google.cloud import aiplatform
from typing import Dict, Optional
import time

class VertexAILoRATrainer:
    """Manage LoRA fine-tuning jobs on Vertex AI"""
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        aiplatform.init(project=project_id, location=location)
        self.project_id = project_id
        self.location = location
    
    def start_lora_tuning(self, train_data_uri: str,
                         val_data_uri: str,
                         base_model: str = "gemini-1.5-pro-002",
                         tuning_config: Optional[Dict] = None) -> str:
        """Start a LoRA fine-tuning job on Vertex AI"""
        
        # Default tuning configuration optimized for COBOL
        if tuning_config is None:
            tuning_config = {
                "epochs": 3,
                "learning_rate": 0.0001,
                "batch_size": 8,
                "adapter_size": 8,  # LoRA rank
            }
        
        print(f"Starting LoRA fine-tuning job...")
        print(f"Base model: {base_model}")
        print(f"Training data: {train_data_uri}")
        print(f"Validation data: {val_data_uri}")
        
        # Create tuning job
        tuning_job = aiplatform.TuningJob.create(
            source_model=base_model,
            train_dataset=train_data_uri,
            validation_dataset=val_data_uri,
            tuned_model_display_name=f"mainframe-cobol-lora-{int(time.time())}",
            tuning_parameters=tuning_config
        )
        
        print(f"Tuning job created: {tuning_job.resource_name}")
        
        return tuning_job.resource_name
    
    def monitor_tuning_job(self, job_name: str) -> Dict:
        """Monitor a tuning job until completion"""
        
        job = aiplatform.TuningJob(job_name)
        
        print("Monitoring tuning job...")
        while True:
            job.refresh()
            state = job.state.name
            
            print(f"Current state: {state}")
            
            if state == "JOB_STATE_SUCCEEDED":
                print("✓ Tuning job completed successfully!")
                break
            elif state in ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED"]:
                print(f"✗ Tuning job {state}")
                break
            
            time.sleep(60)  # Check every minute
        
        return {
            "state": job.state.name,
            "tuned_model": job.tuned_model_name if hasattr(job, 'tuned_model_name') else None,
            "metrics": job.tuning_metrics if hasattr(job, 'tuning_metrics') else None
        }
    
    def deploy_tuned_model(self, tuned_model_name: str,
                          endpoint_name: str = "mainframe-cobol-endpoint") -> str:
        """Deploy the tuned model to an endpoint for inference"""
        
        model = aiplatform.Model(tuned_model_name)
        
        # Create or get endpoint
        endpoints = aiplatform.Endpoint.list(
            filter=f'display_name="{endpoint_name}"'
        )
        
        if endpoints:
            endpoint = endpoints[0]
            print(f"Using existing endpoint: {endpoint.resource_name}")
        else:
            endpoint = aiplatform.Endpoint.create(
                display_name=endpoint_name
            )
            print(f"Created new endpoint: {endpoint.resource_name}")
        
        # Deploy model
        print("Deploying model to endpoint...")
        endpoint.deploy(
            model=model,
            machine_type="n1-standard-4",
            min_replica_count=1,
            max_replica_count=3,
        )
        
        print(f"✓ Model deployed to endpoint: {endpoint.resource_name}")
        
        return endpoint.resource_name
```

---

## Main Orchestration Script

### Purpose
Tie all components together in a single executable script.

### Implementation

```python
# main.py
import argparse
import os
from typing import Dict, Any
from vertexai.generative_models import GenerativeModel

# Import all components
from agent_orchestrator import SCLMExtractionAgent, ExtractionState
from zosmf_collector import ZOSMFCollectionAgent
from sclm_parser import SCLMParserAgent
from training_data_generator import TrainingDataGenerator, TrainingExample
from gcs_uploader import GCSTrainingDataUploader
from vertex_lora_trainer import VertexAILoRATrainer

def main(config: Dict[str, Any]):
    """Main execution flow"""
    
    print("=" * 80)
    print("SCLM Training Data Extraction Agent for LoRA Fine-Tuning")
    print("=" * 80)
    
    # Initialize Gemini model for agent reasoning
    gemini = GenerativeModel("gemini-1.5-pro")
    
    # Step 1: Initialize components
    print("\n[1/7] Initializing components...")
    
    zosmf_agent = ZOSMFCollectionAgent(
        host=config["zosmf_host"],
        username=config["zosmf_user"],
        password=config["zosmf_pass"],
        gemini_model=gemini
    )
    
    parser_agent = SCLMParserAgent(gemini)
    generator = TrainingDataGenerator(gemini)
    uploader = GCSTrainingDataUploader(
        project_id=config["gcp_project"],
        bucket_name=config["gcs_bucket"]
    )
    trainer = VertexAILoRATrainer(
        project_id=config["gcp_project"],
        location=config["gcp_region"]
    )
    
    # Step 2: Discover and collect source code
    print("\n[2/7] Discovering SCLM datasets...")
    datasets = zosmf_agent.discover_datasets_with_agent(config["dataset_hlq"])
    print(f"Found {len(datasets)} relevant datasets")
    
    all_examples = []
    
    # Step 3: Process each dataset
    print("\n[3/7] Extracting and parsing source code...")
    for i, dataset_info in enumerate(datasets):
        dataset_name = dataset_info["name"]
        print(f"\nProcessing dataset {i+1}/{len(datasets)}: {dataset_name}")
        
        # Get members
        members = zosmf_agent.get_dataset_members_with_sampling(dataset_name)
        print(f"  Found {len(members)} members")
        
        # Read source code in batches
        source_code_batch = zosmf_agent.batch_read_members(dataset_name, members)
        
        # Parse each program
        for member_name, source_code in source_code_batch.items():
            try:
                # Parse modification history
                history = parser_agent.parse_modification_history(source_code, member_name)
                
                # Extract marked changes
                marked_changes = parser_agent.extract_marked_changes(source_code)
                
                # Match changes to history
                matched = parser_agent.match_changes_to_history(marked_changes, history)
                
                # Generate training examples
                examples = generator.generate_from_matched_changes(
                    matched, source_code, member_name
                )
                
                # Also generate code completion examples
                completion_examples = generator.generate_code_completion_examples(
                    source_code, member_name
                )
                
                all_examples.extend(examples)
                all_examples.extend(completion_examples)
                
                print(f"    {member_name}: Generated {len(examples) + len(completion_examples)} examples")
                
            except Exception as e:
                print(f"    Error processing {member_name}: {e}")
                continue
    
    # Step 4: Filter and balance dataset
    print(f"\n[4/7] Processing {len(all_examples)} raw examples...")
    
    # Filter by quality
    quality_examples = generator.filter_quality_examples(all_examples, min_quality=0.6)
    print(f"  Filtered to {len(quality_examples)} high-quality examples")
    
    # Balance dataset
    balanced_examples = generator.balance_dataset(quality_examples)
    print(f"  Balanced to {len(balanced_examples)} examples across task types")
    
    # Step 5: Create train/val split and upload
    print("\n[5/7] Uploading to Google Cloud Storage...")
    
    train_examples, val_examples = uploader.create_train_val_split(
        balanced_examples, val_ratio=0.1
    )
    
    uris = uploader.upload_training_data(train_examples, val_examples)
    
    print(f"  Training examples: {len(train_examples)}")
    print(f"  Validation examples: {len(val_examples)}")
    print(f"  Training data URI: {uris['train_uri']}")
    print(f"  Validation data URI: {uris['val_uri']}")
    
    # Step 6: Start LoRA fine-tuning
    if config.get("start_training", False):
        print("\n[6/7] Starting LoRA fine-tuning on Vertex AI...")
        
        job_name = trainer.start_lora_tuning(
            train_data_uri=uris['train_uri'],
            val_data_uri=uris['val_uri'],
            base_model=config.get("base_model", "gemini-1.5-pro-002")
        )
        
        # Monitor training
        result = trainer.monitor_tuning_job(job_name)
        
        if result["state"] == "JOB_STATE_SUCCEEDED":
            print(f"\n[7/7] Training completed successfully!")
            print(f"  Tuned model: {result['tuned_model']}")
            
            # Optionally deploy
            if config.get("deploy_model", False):
                endpoint = trainer.deploy_tuned_model(result['tuned_model'])
                print(f"  Deployed to endpoint: {endpoint}")
        else:
            print(f"\n✗ Training failed with state: {result['state']}")
    else:
        print("\n[6/7] Skipping training (start_training=False)")
        print("[7/7] Done!")
    
    print("\n" + "=" * 80)
    print("Extraction and preparation complete!")
    print("=" * 80)
    
    return {
        "examples_generated": len(balanced_examples),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "data_uris": uris
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SCLM Training Data Extraction Agent")
    
    # zOSMF configuration
    parser.add_argument("--zosmf-host", required=True, help="zOSMF host:port")
    parser.add_argument("--zosmf-user", required=True, help="zOSMF username")
    parser.add_argument("--zosmf-pass", required=True, help="zOSMF password")
    parser.add_argument("--dataset-hlq", required=True, help="Dataset high-level qualifier")
    
    # GCP configuration
    parser.add_argument("--gcp-project", required=True, help="GCP project ID")
    parser.add_argument("--gcp-region", default="us-central1", help="GCP region")
    parser.add_argument("--gcs-bucket", required=True, help="GCS bucket name")
    
    # Training configuration
    parser.add_argument("--base-model", default="gemini-1.5-pro-002", help="Base model for fine-tuning")
    parser.add_argument("--start-training", action="store_true", help="Start training after data prep")
    parser.add_argument("--deploy-model", action="store_true", help="Deploy model after training")
    
    args = parser.parse_args()
    
    config = vars(args)
    
    result = main(config)
    
    print(f"\nFinal statistics:")
    print(f"  Total examples: {result['examples_generated']}")
    print(f"  Training set: {result['train_examples']}")
    print(f"  Validation set: {result['val_examples']}")
```

---

## Configuration File

```yaml
# config.yaml
zosmf:
  host: "mainframe.company.com:443"
  username: "${ZOSMF_USER}"  # From environment variable
  password: "${ZOSMF_PASS}"  # From environment variable
  
datasets:
  hlq: "PROD.COBOL"
  patterns:
    - "*.COBOL.SOURCE"
    - "*.COPY.SOURCE"
  
gcp:
  project_id: "your-project-id"
  region: "us-central1"
  bucket: "mainframe-training-data"
  
training:
  base_model: "gemini-1.5-pro-002"
  lora_config:
    epochs: 3
    learning_rate: 0.0001
    batch_size: 8
    adapter_size: 8
  
  quality:
    min_quality_score: 0.6
    validation_ratio: 0.1
  
agent:
  model: "gemini-1.5-pro"
  temperature: 0.1  # Lower for more consistent parsing
  max_retries: 3
```

---

## Deployment Instructions

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install google-cloud-aiplatform google-cloud-storage requests pyyaml

# Set environment variables
export ZOSMF_USER="your_username"
export ZOSMF_PASS="your_password"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"
```

### 2. Run the Agent

```bash
# Full pipeline with training
python main.py \
  --zosmf-host mainframe.company.com:443 \
  --zosmf-user $ZOSMF_USER \
  --zosmf-pass $ZOSMF_PASS \
  --dataset-hlq PROD.COBOL \
  --gcp-project your-project-id \
  --gcs-bucket mainframe-training-data \
  --start-training \
  --deploy-model

# Data preparation only (no training)
python main.py \
  --zosmf-host mainframe.company.com:443 \
  --zosmf-user $ZOSMF_USER \
  --zosmf-pass $ZOSMF_PASS \
  --dataset-hlq PROD.COBOL \
  --gcp-project your-project-id \
  --gcs-bucket mainframe-training-data
```

### 3. Deploy as Cloud Run Job

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py .
COPY config.yaml .

ENTRYPOINT ["python", "main.py"]
```

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/your-project/sclm-agent

gcloud run jobs create sclm-training-agent \
  --image gcr.io/your-project/sclm-agent \
  --region us-central1 \
  --set-env-vars GCP_PROJECT=your-project \
  --set-secrets ZOSMF_USER=zosmf-user:latest \
  --set-secrets ZOSMF_PASS=zosmf-pass:latest \
  --max-retries 1 \
  --task-timeout 7200  # 2 hours
```

---

## Dependencies (requirements.txt)

```txt
google-cloud-aiplatform>=1.38.0
google-cloud-storage>=2.10.0
requests>=2.31.0
pyyaml>=6.0.1
langgraph>=0.0.20
vertexai>=1.38.0
urllib3>=2.0.0
```

---

## Monitoring and Logging

```python
# logging_config.py
import logging
from google.cloud import logging as cloud_logging

def setup_logging(project_id: str):
    """Setup Cloud Logging"""
    
    client = cloud_logging.Client(project=project_id)
    client.setup_logging()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    return logging.getLogger(__name__)
```

---

## Success Metrics

Track these metrics to evaluate the agent:

### 1. Data Collection
- Programs processed: X/Y
- Success rate: Z%
- Average processing time per program

### 2. Parsing Quality
- Change history entries extracted: X
- Marked changes found: Y
- Match rate: Z%

### 3. Training Data Quality
- Total examples generated: X
- High-quality examples (>0.6): Y
- Task type distribution balance

### 4. LoRA Training
- Training time
- Validation loss
- Training loss convergence

### 5. Model Performance
- COBOL code completion accuracy
- Bug fix suggestion relevance
- Code review false positive rate

---

## Project Structure

```
sclm-lora-agent/
├── main.py
├── agent_orchestrator.py
├── zosmf_collector.py
├── sclm_parser.py
├── training_data_generator.py
├── gcs_uploader.py
├── vertex_lora_trainer.py
├── logging_config.py
├── config.yaml
├── requirements.txt
├── Dockerfile
├── .env.example
└── README.md
```

---

## Environment Variables (.env.example)

```bash
# zOSMF Configuration
ZOSMF_HOST=mainframe.company.com:443
ZOSMF_USER=your_username
ZOSMF_PASS=your_password

# GCP Configuration
GCP_PROJECT=your-project-id
GCP_REGION=us-central1
GCS_BUCKET=mainframe-training-data
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json

# Dataset Configuration
DATASET_HLQ=PROD.COBOL

# Training Configuration
BASE_MODEL=gemini-1.5-pro-002
START_TRAINING=false
DEPLOY_MODEL=false
```

---

## Troubleshooting Guide

### Issue: zOSMF Connection Timeout
**Solution**: Check network connectivity, verify VPN if required, increase timeout values

### Issue: Low Quality Training Examples
**Solution**: Adjust `min_quality_score` threshold, review parsing patterns, add more curated examples

### Issue: LoRA Training Fails
**Solution**: Check data format, verify GCS permissions, review Vertex AI quotas

### Issue: Memory Issues During Processing
**Solution**: Reduce batch size, process datasets sequentially, use streaming where possible

---

## Next Steps

1. **Test on Sample Dataset**: Start with small dataset (100-500 programs) to validate pipeline
2. **Evaluate Model Quality**: Test fine-tuned model on held-out COBOL programs
3. **Iterate on Parsing**: Refine agent prompts based on parsing errors
4. **Scale Up**: Gradually increase to full 38M line codebase
5. **Deploy to Production**: Set up automated monthly refresh pipeline

---

## License

Proprietary - Internal Use Only

---

## Contact

For questions or support, contact: [Your Team/Email]

---

**End of Implementation Plan**
