# BonBon: Software Architecture

## System Overview

BonBon is an intelligent context management system that optimizes the interaction between developers and LLMs by generating task-specific "Context Booster Packs." The system analyzes code dependencies, performs semantic searches, and manages token budgets to provide just the right context.

```
┌─────────────────┐     ┌───────────────┐     ┌─────────────────┐
│                 │     │               │     │                 │
│  macOS UI App   │◄────┤   API Layer   │◄────┤  Core Engine    │
│                 │     │               │     │                 │
└─────────────────┘     └───────────────┘     └─────────────────┘
                                                       ▲
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │                 │
                                              │   MCP Server    │
                                              │                 │
                                              └─────────────────┘
                                                       ▲
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │                 │
                                              │      LLM        │
                                              │                 │
                                              └─────────────────┘
```

## Core Components

### 1. Core Engine

The heart of BonBon handling all analysis and intelligence.

```
┌────────────────────────────────────────────────────────┐
│                     Core Engine                        │
│                                                        │
│  ┌───────────────┐   ┌────────────────┐   ┌─────────┐  │
│  │ Code Analyzer │   │ Vector Storage │   │ Token   │  │
│  │ - AST parsing │   │ - LanceDB      │   │ Counter │  │
│  │ - Git history │   │ - Embeddings   │   │         │  │
│  └───────────────┘   └────────────────┘   └─────────┘  │
│                                                        │
│  ┌───────────────┐   ┌────────────────┐   ┌─────────┐  │
│  │ Doc Harvester │   │ Booster Pack   │   │ Cache   │  │
│  │ - API specs   │   │ Generator      │   │ Manager │  │
│  │ - Markdown    │   │                │   │         │  │
│  └───────────────┘   └────────────────┘   └─────────┘  │
│                                                        │
│  ┌───────────────────────────────────────────────────┐ │
│  │              MCP Protocol Handler                  │ │
│  │ - UUID generation                                 │ │
│  │ - Pack retrieval                                  │ │
│  │ - Request authentication                          │ │
│  └───────────────────────────────────────────────────┘ │
│                                                        │
└────────────────────────────────────────────────────────┘
```

#### Code Analyzer
- **Static Analysis**: Uses Python's AST and Jedi for code parsing
- **Dependency Tracking**: Builds a dependency graph of modules/imports
- **Git Integration**: Uses repository history to identify recent changes
- **Semantic Matching**: Maps task descriptions to code components

#### Data Storage
- **Structured Data**: Uses DuckDB for relational data and complex queries
- **Vector Database**: Uses LanceDB for embeddings and similarity search
- **Document Indexing**: Chunks and stores documentation with metadata
- **Query Optimization**: Leverages columnar storage for analytical workloads

#### Token Management
- **Token Counter**: Uses tiktoken to count tokens accurately by model
- **Budget Allocation**: Dynamically allocates tokens between code/docs
- **Content Prioritization**: Ranks content by relevance for inclusion
- **Truncation Strategy**: Intelligent truncation preserving context

#### Document Harvester
- **API Integration**: Direct integration with DevDocs API and documentation repositories
- **DocSet Support**: Import from Zeal/Dash docsets for offline documentation
- **URL Processing**: Fallback web crawling for unsupported documentation sites
- **Markdown Processor**: Converts HTML to markdown and parses documentation
- **API Spec Parser**: Extracts structured data from OpenAPI/Swagger specifications
- **Section Analysis**: Identifies logical documentation sections and hierarchies
- **Chunking Engine**: Divides content into semantic units with strategic overlap

```
┌─────────────────────────────────────────────────────────────────┐
│                Documentation Harvesting Pipeline                │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌────────────────────┐   │
│  │ Source      │───►│ Content     │───►│ Extraction Layer   │   │
│  │ Selection   │    │ Acquisition │    │ - DevDocs API      │   │
│  │             │    │             │    │ - DocSet Parser    │   │
│  └─────────────┘    └─────────────┘    │ - trafilatura      │   │
│                                        │ - crawl4ai         │   │
│                                        └────────────────────┘   │
│                                                 │               │
│                                                 ▼               │
│  ┌─────────────┐    ┌─────────────┐    ┌────────────────────┐   │
│  │ Vector      │◄───│ Semantic    │◄───│ Content Chunking   │   │
│  │ Storage     │    │ Embedding   │    │ - Semantic units   │   │
│  │ - LanceDB   │    │ - ST models │    │ - Overlap strategy │   │
│  └─────────────┘    └─────────────┘    └────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Implementation Note**: The harvester primarily leverages existing documentation repositories (DevDocs, Zeal/Dash docsets) where possible. For documentation sources not available through these channels, the extractor can use either trafilatura (for general content) or crawl4ai (for specialized documentation extraction) based on configuration. This tiered approach balances pragmatism with flexibility.

#### Booster Pack Generator
- **Task Analyzer**: Processes task descriptions into components
- **Content Selector**: Chooses relevant files and documentation
- **Instruction Generator**: Produces clear instructions for LLM use
- **Pack Optimizer**: Ensures maximum utility within token limits

#### MCP Protocol Handler
- **UUID Management**: Generates and tracks UUIDs for booster packs
- **Pack Storage**: Maintains a repository of generated packs for retrieval
- **Request Validation**: Authenticates incoming requests from LLMs
- **TTL Management**: Enforces time-to-live policies for cached packs

### 2. API Layer

The communication bridge between UI, core engine, and external systems.

```
┌────────────────────────────────────────────────────┐
│                    API Layer                       │
│                                                    │
│  ┌─────────────────┐   ┌────────────────────────┐  │
│  │ FastAPI Server  │   │      WebSocket         │  │
│  │ - RESTful API   │   │      Manager           │  │
│  │ - JSON schemas  │   │                        │  │
│  └─────────────────┘   └────────────────────────┘  │
│                                                    │
│  ┌─────────────────┐   ┌────────────────────────┐  │
│  │ Error Handler   │   │      Request           │  │
│  │ - Logging       │   │      Queue             │  │
│  │ - User feedback │   │                        │  │
│  └─────────────────┘   └────────────────────────┘  │
│                                                    │
│  ┌────────────────────────────────────────────┐    │
│  │             MCP Server Connector           │    │
│  │ - API integration                          │    │
│  │ - Request/response handling                │    │
│  │ - Authentication                           │    │
│  └────────────────────────────────────────────┘    │
│                                                    │
└────────────────────────────────────────────────────┘
```

- **FastAPI Server**: Provides RESTful endpoints
- **WebSocket Manager**: Handles real-time connections with UI
- **Request Queue**: Manages concurrent requests
- **Error Handler**: Standardizes error responses and logging
- **MCP Server Connector**: Mediates between BonBon and MCP Server

### 3. macOS UI Application

The native interface for developer interaction.

```
┌──────────────────────────────────────────────────┐
│                macOS UI App                      │
│                                                  │
│  ┌─────────────────┐   ┌────────────────────┐    │
│  │ Project Browser │   │  Task Definition   │    │
│  │ - File viewer   │   │  - Description     │    │
│  │ - Selection     │   │  - Component list  │    │
│  └─────────────────┘   └────────────────────┘    │
│                                                  │
│  ┌─────────────────┐   ┌────────────────────┐    │
│  │ Booster Viewer  │   │  LLM Integration   │    │
│  │ - Content view  │   │  - Provider config │    │
│  │ - Statistics    │   │  - Export options  │    │
│  └─────────────────┘   └────────────────────┘    │
│                                                  │
│  ┌────────────────────────────────────────┐      │
│  │          UUID Management               │      │
│  │ - Pack ID display                      │      │
│  │ - Sharing options                      │      │
│  │ - TTL indicators                       │      │
│  └────────────────────────────────────────┘      │
│                                                  │
└──────────────────────────────────────────────────┘
```

- **Project Browser**: Navigate and select project files
- **Task Definition**: Define specific development tasks
- **Booster Pack Viewer**: Examine generated context
- **LLM Integration**: Configure and connect to LLMs
- **UUID Management**: Manage and share pack identifiers

### 4. MCP Server

The Model Context Protocol server enables LLMs to retrieve context packs dynamically.

```
┌────────────────────────────────────────────┐
│               MCP Server                   │
│                                            │
│  ┌─────────────────┐   ┌────────────────┐  │
│  │ Request Handler │   │ Authentication │  │
│  │ - UUID lookup   │   │ - API keys     │  │
│  │ - Validation    │   │ - Rate limits  │  │
│  └─────────────────┘   └────────────────┘  │
│                                            │
│  ┌─────────────────┐   ┌────────────────┐  │
│  │ Pack Delivery   │   │ Metrics        │  │
│  │ - Formatting    │   │ - Usage stats  │  │
│  │ - Streaming     │   │ - Performance  │  │
│  └─────────────────┘   └────────────────┘  │
│                                            │
└────────────────────────────────────────────┘
```

- **Request Handler**: Processes incoming requests from LLMs
- **Authentication**: Validates access to booster packs
- **Pack Delivery**: Formats and delivers context packs
- **Metrics**: Tracks usage and performance statistics

## Key Workflows

### 1. Project Setup

```
┌───────────┐     ┌───────────┐     ┌───────────┐     ┌───────────┐
│           │     │           │     │           │     │           │
│  Select   │────►│  Index    │────►│  Parse    │────►│  Build    │
│  Project  │     │  Files    │     │  Code     │     │  Graph    │
│           │     │           │     │           │     │           │
└───────────┘     └───────────┘     └───────────┘     └───────────┘
```

### 2. Booster Pack Generation

```
┌───────────┐     ┌───────────┐     ┌───────────┐     ┌───────────┐
│           │     │           │     │           │     │           │
│  Define   │────►│  Analyze  │────►│ Generate  │────►│  Display  │
│  Task     │     │  Context  │     │   Pack    │     │   Pack    │
│           │     │           │     │           │     │           │
└───────────┘     └───────────┘     └───────────┘     └───────────┘
```

### 3. LLM Integration

```
┌───────────┐     ┌───────────┐     ┌───────────┐     ┌───────────┐
│           │     │           │     │           │     │           │
│  Review   │────►│ Generate  │────►│  Provide  │────►│ Integrate │
│  Pack     │     │   UUID    │     │  to LLM   │     │  Response │
│           │     │           │     │           │     │           │
└───────────┘     └───────────┘     └───────────┘     └───────────┘
```

### 4. MCP Protocol Flow

```
┌───────────┐     ┌───────────┐     ┌───────────┐     ┌───────────┐
│           │     │           │     │           │     │           │
│   LLM     │────►│   MCP     │────►│  BonBon   │────►│  Return   │
│  Request  │     │  Server   │     │  Lookup   │     │   Pack    │
│           │     │           │     │           │     │           │
└───────────┘     └───────────┘     └───────────┘     └───────────┘
```

## MCP Protocol Integration

The Model Context Protocol (MCP) is a key component that enables LLMs to dynamically retrieve context during their operation. This integration allows for "just-in-time" context delivery, significantly enhancing token efficiency by enabling:

1. **Contextual Task Lists**: Providing UUIDs instead of full context in initial prompts
2. **On-demand Retrieval**: Fetching only the context needed at the moment
3. **Stateful Conversations**: Maintaining context across multiple interactions
4. **Secure Sharing**: Ensuring context is accessible only to authorized LLMs

### MCP Protocol Flow Details

1. **UUID Generation**:
   - When a booster pack is created, BonBon assigns it a unique UUID
   - The pack is cached in BonBon's storage with appropriate TTL settings
   - The UUID is displayed to the user for sharing with LLMs

2. **LLM Request**:
   - The user provides the UUID to the LLM (e.g., Claude-Code)
   - The LLM identifies the UUID in the task description
   - LLM makes an API call to the MCP Server with the UUID

3. **MCP Server Processing**:
   - Validates the request authenticity
   - Locates the BonBon instance associated with the UUID
   - Forwards the request to the appropriate BonBon instance

4. **BonBon Response**:
   - Validates the request against security policies
   - Retrieves the cached booster pack
   - Formats the pack according to MCP specifications
   - Returns the pack to the MCP Server

5. **Context Delivery**:
   - MCP Server delivers the context to the requesting LLM
   - LLM incorporates the context into its processing
   - Task execution continues with the enriched context

### MCP API Specification

```
POST /api/v1/context/{uuid}
Authorization: Bearer {api_key}
Content-Type: application/json

Request Body (optional):
{
  "client_info": {
    "name": "Claude-Code",
    "version": "1.0.0",
    "user_id": "user-123"
  },
  "format_preferences": {
    "chunk_size": 4000,
    "include_metadata": true
  }
}

Response:
{
  "task": {
    "description": "Task description",
    "components": ["file1.py", "file2.py"]
  },
  "code_chunks": [
    {
      "source": "file1.py",
      "content": "...",
      "metadata": {
        "language": "python",
        "last_modified": "2023-08-15T14:32:40Z",
        "relevance_score": 0.95
      }
    }
  ],
  "doc_chunks": [
    {
      "source": "README.md",
      "content": "...",
      "metadata": {
        "format": "markdown",
        "relevance_score": 0.87
      }
    }
  ],
  "instructions": "...",
  "session": {
    "expires_at": "2023-08-16T23:59:59Z",
    "remaining_requests": 50
  }
}
```

### Tool Integration Adapters

The MCP Server includes specialized adapters for popular development tools:

```
┌────────────────────────────────────────────────────────┐
│                   MCP Server Adapters                  │
│                                                        │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────┐ │
│  │ Claude Desktop │  │  Claude-Code   │  │  VS Code  │ │
│  │  - URI schema  │  │ - CLI protocol │  │ - Plugin  │ │
│  │  - Embedding   │  │ - Streaming    │  │ - Panel   │ │
│  └────────────────┘  └────────────────┘  └───────────┘ │
│                                                        │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────┐ │
│  │  Cursor IDE    │  │    Windsurf    │  │  GitHub   │ │
│  │ - Annotations  │  │ - Integration  │  │  Copilot  │ │
│  │ - Context menu │  │ - Plugin API   │  │ - Comments│ │
│  └────────────────┘  └────────────────┘  └───────────┘ │
│                                                        │
└────────────────────────────────────────────────────────┘
```

#### Claude Desktop Adapter
- **URI Protocol**: Implements `mcp://` URI scheme for deep linking
- **Format**: Optimizes context structure for Claude's conversation format
- **Embedding**: Enables context embedding in conversation history
- **UI Integration**: Provides visual indicators for active contexts

#### Claude-Code Adapter
- **CLI Integration**: Extends Claude-Code with `--context mcp:UUID` parameter
- **Authentication**: Uses CLI credential store for seamless authentication
- **Streaming**: Supports progressive context delivery for terminal use
- **Status Display**: Shows context retrieval progress in terminal

#### VS Code Adapter
- **Extension Integration**: Provides direct integration with VS Code extension API
- **Context Management**: Panel for viewing/managing active context packs
- **Query Generation**: Helps formulate effective LLM queries with context awareness
- **Workspace Settings**: Stores UUIDs in workspace settings for persistence

#### Cursor IDE Adapter
- **Comment Annotations**: Detects `@bonbon-context: UUID` in file comments
- **Context Menu**: Adds BonBon context menu options to file explorer
- **Hotkey Support**: Custom keyboard shortcuts for context operations
- **Inline UI**: Context status indicators inline with editor

#### Windsurf Adapter
- **Native Plugin**: Integrates with Windsurf's plugin architecture
- **Context Browser**: Provides searchable interface for available contexts
- **Session Management**: Maintains context across editing sessions
- **Co-pilot Panel**: Dedicated panel for context-aware AI assistance

#### GitHub Copilot Adapter
- **Comment Directives**: Detects `@github-copilot context-from: bonbon:UUID`
- **Mapping Layer**: Translates BonBon context to Copilot-compatible format
- **OAuth Integration**: Uses GitHub's OAuth flow for authentication
- **Suggestion Enhancement**: Improves Copilot suggestions with contextual data

### Security Considerations

- **Request Authentication**: All MCP requests require API key validation
- **Rate Limiting**: Prevents abuse of the context retrieval system
- **Expiration Policies**: Context packs have configurable TTL settings
- **Access Logs**: All retrievals are logged for audit purposes
- **Tool-Specific Auth**: Custom authentication flows for each integrated tool
- **Scoped Access**: Tools receive only the context they need based on permissions
- **Mutual TLS**: Optional mTLS for high-security environments
- **Content Security Policy**: Controls which tools can access specific context types

## Implementation Priorities (Aligned with User Stories)

1. **Phase 1: Core Analysis Engine** (Supports Miguel's "Understanding Component Interactions")
   - Dependency graph generation
   - Basic semantic search
   - Project structure analysis

2. **Phase 2: Documentation Harvesting** (Supports Sarah's "Documentation Harvesting and Integration")
   - crawl4ai integration
   - Content extraction and chunking
   - Documentation vectorization
   - LanceDB storage

3. **Phase 3: Context Pack Generation** (Supports Sarah's "Refactoring Complex Components")
   - Token optimization
   - Context selection
   - Relevance ranking

4. **Phase 4: MCP Protocol Support** (Supports Sarah's "Task Batching Workflow")
   - UUID generation and management
   - Pack storage and retrieval
   - Security mechanisms

5. **Phase 5: Tool Integrations** (Supports Miguel's "Debugging Complex Issues")
   - IDE plugins
   - Claude-Code integration
   - Client adapters

## Error Handling Strategy

1. **Graceful Degradation**: If dependency analysis fails, fall back to semantic search
2. **Token Budget Flexibility**: Adjust allocations if certain content types are unavailable
3. **Async Processing**: Long-running operations use WebSockets with progress updates
4. **User Correction**: Allow manual addition/removal of context components
5. **MCP Failover**: Provide alternative access methods if MCP retrieval fails

## Scaling Considerations

1. **Large Repositories**: Implement partial indexing and on-demand analysis
2. **Performance Optimization**: Cache analysis results for unchanged files
3. **Distributed Processing**: Option to offload vector processing to separate service
4. **Memory Management**: Stream large files instead of loading entirely in memory
5. **MCP Request Volume**: Implement queuing and prioritization for high-traffic scenarios

## Integration with Development Environments

1. **Claude-Code Integration**: Direct support via MCP protocol
2. **CLI Tools**: Command-line interface for automation
3. **IDE Extensions**: Future plugins for VSCode and JetBrains IDEs
4. **CI/CD Integration**: Hooks for automated context generation in pipelines
