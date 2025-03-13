# Task: Develop a Targeted Static Analysis Tool for BonBon Classification System

## Overview
Create a Python-based static analysis tool that:
1. Accepts classification codes (e.g., "5.2.2.0-v1") as input
2. Identifies relevant files matching that classification
3. Performs targeted static analysis on those files
4. Outputs results in a structured JSON format compatible with our context booster pack system

## Classification System Integration
Use our BonBon classification manifest (format shown below) to identify relevant files:

```toml
[[file]]
path = "src/mcp/protocol_handler.py"
classification = "5.1.1.0-v3"
description = "MCP request handling and routing"
```

The tool should:
- Match exact classifications
- Match parent classifications (e.g., "5.2" matches "5.2.1.0")
- Include configurable "related" classifications

## Static Analysis Requirements
Implement these analysis types:
- AST-based structural analysis (imports, dependencies, function signatures)
- Control flow analysis (complexity, edge cases, error handling)
- Pattern detection (identify common patterns like singletons, factories)
- Security checks on authentication/validation code

Use libraries like `libcst`, `astroid`, or `jedi` for parsing and analysis.

## Performance Considerations
- Implement incremental analysis (only analyze changed files)
- Cache analysis results with classification-aware invalidation
- Support parallel analysis for larger codebases

## Output Format
JSON structured as:

```json
{
  "classification": {
    "code": "5.2.2.0-v1",
    "path": "Authentication > OAuth Integration",
    "version": "v1"
  },
  "files_analyzed": [
    {"path": "src/mcp/oauth_handler.py", "match_type": "exact", "lines": 250},
    {"path": "src/security/token_validation.py", "match_type": "related", "lines": 180}
  ],
  "entry_points": [
    {"file": "src/mcp/oauth_handler.py", "function": "validate_token", "line": 42, "relevance": 9, "confidence": 8}
  ],
  "dependencies": [
    {"primary": "src/mcp/oauth_handler.py", "depends_on": "src/security/token_validation.py", "type": "import", "confidence": 9}
  ],
  "patterns": [
    {"pattern": "Token validation", "implementation": "JWT verification with expiration checks", "location": "src/security/token_validation.py:55-78", "confidence": 7}
  ],
  "performance_concerns": [
    {"issue": "Multiple database calls in token validation path", "impact": "HIGH", "location": "src/mcp/oauth_handler.py:102-118", "confidence": 8}
  ],
  "security_findings": [
    {"finding": "Missing token expiration check", "severity": "MEDIUM", "location": "src/mcp/oauth_handler.py:152", "confidence": 9}
  ],
  "metadata": {
    "analysis_time": "2024-03-12T15:45:30Z",
    "tool_version": "1.0.0"
  }
}
```

This format should seamlessly integrate with the BonBon context generation pipeline.

## CLI Interface
The tool should provide:
```
python bonbon_analyzer.py --classification 5.2.2.0-v1 [--include-related] [--depth 2] [--output output.json]
```

Focus on making the analysis accurate and useful for the LLM context generation process. Analysis depth should be configurable based on task complexity.