# BonBon

## Intelligent Context Management for LLM Development

BonBon is a specialized tool designed to optimize the interaction between developers and Large Language Models by intelligently managing context windows. It addresses a critical inefficiency in the current LLM workflow: the indiscriminate transmission of project documentation that wastes tokens, increases costs, and dilutes relevance.

## Design Documentation Index

### System Overview
- [Software Architecture](software-architecture.md) - System design, components, and practical implementation
- [User Stories](user-stories.md) - Core use cases and user workflows driving the design

### Technical Architecture
- [AI Architecture](ai-architecture.md) - Advanced AI/ML implementation techniques and algorithms
- [Prompt Chain Design](prompt-chain.md) - Context orchestration through specialized agents

### Core Components
- [Data Model](data-model.md) - Core data entities and storage implementation
- [Context Pack Format](context-pack.txt) - Context pack structure specification
- [Static Analysis Tool](static-analysis-tool-prompt.md) - Static analysis classification design

### Supporting Documentation
- [Project Requirements](requirements.txt) - Dependencies and package requirements

## The Problem

When working with LLMs like Claude or GPT for development tasks, we typically send one of:

1. **Too little context** - resulting in the LLM lacking necessary information
2. **Too much context** - wasting tokens on irrelevant information and increasing costs
3. **The wrong context** - providing tangentially related but ultimately unhelpful documentation

Each of these approaches is suboptimal, particularly as projects scale in complexity. Manual selection of context is time-consuming and error-prone, often missing critical dependencies or related components.

## Core Concept

BonBon introduces the concept of **Context Booster Packs** - dynamically generated, task-specific context packages that contain precisely the information an LLM needs to complete a specific task. These packs:

- Include only task-relevant code files and documentation
- Maintain awareness of dependencies between components
- Intelligently manage token budgets to maximize utility
- Come with clear instructions for the LLM on what's available and how to ask for additional information

## Architecture

BonBon is built on a three-tier architecture:

1. **Core Analysis Engine**
   - Code dependency analysis
   - Semantic relevance scoring
   - Token optimization
   - Vector embedding for similarity search
   - MCP protocol integration

2. **API/Service Layer**
   - WebSocket server for LLM connections
   - RESTful API for context generation
   - Asynchronous processing pipeline
   - Model-specific formatting
   - MCP server connectivity

3. **macOS Native UI**
   - Project file browser
   - Task definition interface
   - Context visualization
   - LLM integration panel
   - UUID management for MCP

The system uses:
- LanceDB for vector storage and similarity search
- Python's AST module and Jedi for static code analysis
- Sentence Transformers for semantic embedding
- PyQt6 for the native macOS interface
- DuckDB for structured data storage

## Model Context Protocol (MCP)

BonBon implements the Model Context Protocol (MCP), which enables LLMs to dynamically retrieve context during operation. This innovation allows for:

- Contextual task lists with deferred context retrieval
- Token-efficient interactions that load context only when needed
- Secure sharing of context between applications
- Stateful conversations with evolving context needs

When a context pack is generated, BonBon:
1. Assigns it a unique UUID
2. Stores it securely with appropriate access controls
3. Allows LLMs to retrieve it via the MCP server
4. Tracks usage and maintains audit logs

## Key Capabilities

### Documentation Harvesting

BonBon adopts a pragmatic approach to documentation acquisition, focusing development effort on context optimization rather than rebuilding existing solutions:

- **DevDocs Integration**: Direct API access to DevDocs.io's extensive library of technical documentation
- **DocSet Support**: Import from Zeal/Dash docsets for offline, structured documentation
- **Flexible Web Extraction**: Multiple options for documentation not available through standard repositories:
  - **trafilatura**: Clean content extraction for most documentation sites
  - **crawl4ai**: Specialized extraction for complex documentation
- **Semantic Chunking**: Division of documentation into logical, overlapping semantic units
- **Hierarchical Organization**: Preservation of documentation structure and section relationships
- **Versioned Collections**: Tracking of documentation versions with automated update detection

This multi-source approach eliminates the overhead of building custom crawlers while maintaining high-quality documentation access:

```python
# Example configuration for documentation harvesting
doc_config = {
    # Primary sources - use established documentation repositories
    "sources": [
        {"type": "devdocs", "documentation_id": "python~3.11"},
        {"type": "docset", "path": "/Users/developer/Library/Application Support/Zeal/docsets/FastAPI.docset"}
    ],

    # Fallback for custom documentation
    "web_extraction": {
        "base_url": "https://company-internal-docs.example.com/api/",
        "extraction_library": "crawl4ai",  # Options: "trafilatura", "crawl4ai"
        "content_selectors": ["main", "article", ".content"],
        "code_block_selectors": ["pre", "code"]
    },

    # Processing configuration
    "chunking_strategy": "semantic",
    "chunk_size": 1500,
    "chunk_overlap": 150
}

# Start the harvesting process
doc_source_id = bonbon.harvest_documentation(
    project_id=project.id,
    name="Python and FastAPI Documentation",
    config=doc_config
)
```

The resulting documentation chunks are embedded and stored in LanceDB, making them available for context-aware retrieval during development tasks.

### Intelligent Context Selection

Unlike basic RAG approaches that only perform similarity searches, BonBon analyzes:
- Code dependencies and imports
- Module relationships and hierarchies
- Recent code changes in the repository
- Documentation relevance to specified tasks
- Online API documentation (OpenAPI, Swagger, etc.)

### Token Economy

BonBon optimizes for token efficiency by:
- Accurately counting tokens using model-specific tokenizers
- Prioritizing content based on relevance scores
- Truncating less-relevant content when necessary
- Balancing between code and documentation based on task requirements

### LLM Integration

The tool integrates with LLMs through multiple channels:
- Direct API integration with major providers
- MCP server for dynamic context retrieval
- WebSocket connections for streaming interactions
- Clipboard formatting for manual insertion
- Command-line interface for script automation

## Use Cases

BonBon excels in the following scenarios:

**Code Refactoring**
- Provides the LLM with all affected components and their dependencies
- Includes relevant architecture documentation
- Minimizes token usage by excluding unrelated system components

**Bug Fixing**
- Prioritizes the error-containing file and its dependencies
- Includes relevant error handling documentation
- Optimizes token usage based on error complexity

**Feature Development**
- Identifies similar existing implementations
- Includes architecture guidelines and related components
- Balances code and documentation based on feature complexity

**Technical Documentation**
- Provides code samples alongside existing documentation
- Focuses on components relevant to the documentation task
- Optimizes for knowledge transfer between code and docs

**API Integration**
- Harvests OpenAPI specifications from documentation URLs
- Maintains versioned collections of third-party API documentation
- Enriches context with relevant API endpoints and parameters

**Task Batching**
- Supports multi-step development workflows
- Enables LLMs to retrieve context on demand for specific tasks
- Maintains coherence across related tasks without context overload

## Requirements

- Python 3.9+
- macOS 11.0+ (for the native UI)
- Access to LLM APIs (OpenAI, Anthropic, etc.)
- Git repository for optimal code analysis

## The BonBon Advantage

Unlike general-purpose RAG implementations or basic code search tools, BonBon:

1. **Understands code structure** - not just text similarity
2. **Optimizes for token efficiency** - reducing costs and improving relevance
3. **Adapts to specific tasks** - providing precisely what's needed
4. **Integrates with developer workflow** - through native UI and CLI interfaces
5. **Provides transparency** - showing exactly what's included and why
6. **Harvests online documentation** - turning web-based API docs into structured context
7. **Supports dynamic context retrieval** - through MCP protocol integration

By serving as the intelligent intermediary between your codebase and LLMs, BonBon represents a significant advancement in programming with AI assistance, transforming how developers leverage LLMs for complex software tasks.
