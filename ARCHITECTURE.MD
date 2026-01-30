graph TB
    subgraph "Interface Utilisateur"
        CLI[CLI<br/>clap + colored]
        VSCODE[VS Code Extension<br/>LSP Client]
        INTELLIJ[IntelliJ Plugin<br/>LSP Client]
    end

    subgraph "Core Application"
        ROUTER[Command Router<br/>Dispatch des commandes]
        
        subgraph "LLM Layer - Multi-Provider"
            MGR[LLM Manager<br/>Gestion centralisée]
            CONFIG[Configuration<br/>TOML Parser]
            
            subgraph "Providers Cloud"
                CLAUDE[Claude Provider<br/>Anthropic API]
                OPENAI[OpenAI Provider<br/>GPT-4 API]
                GEMINI[Gemini Provider<br/>Google API]
            end
            
            subgraph "Providers Local"
                OLLAMA[Ollama Provider<br/>localhost:11434]
                LLAMA[LlamaCpp Provider<br/>localhost:8080]
            end
            
            CUSTOM[Custom Provider<br/>URL configurable]
        end
        
        subgraph "Memory System"
            CACHE[Cache Court Terme<br/>LRU In-Memory]
            VECTOR[Vector Store<br/>Qdrant Embedded]
            GRAPH[Knowledge Graph<br/>Dépendances]
            HISTORY[SQLite History<br/>Persistance]
        end
        
        subgraph "Code Analysis"
            PARSER[Tree-sitter Parser<br/>Multi-langage]
            AST[AST Analyzer<br/>Analyse syntaxique]
            METRICS[Code Metrics<br/>Complexité]
            DEPS[Dependency Graph<br/>Relations]
        end
        
        subgraph "Features"
            GEN[Code Generator<br/>Génération contextuelle]
            REF[Refactoring Engine<br/>Transformations]
            DOC[Documentation Gen<br/>Markdown + Diagrams]
            ERR[Error Detector<br/>Analyse statique]
        end
    end

    subgraph "External Services"
        CLAUDE_API[Anthropic API<br/>api.anthropic.com]
        OPENAI_API[OpenAI API<br/>api.openai.com]
        GEMINI_API[Google Gemini API<br/>generativelanguage...]
        OLLAMA_SRV[Ollama Server<br/>Local Models]
        LLAMA_SRV[Llama.cpp Server<br/>GGUF Models]
    end

    subgraph "Storage"
        FS[File System<br/>Source Code]
        VECTDB[(Vector DB<br/>Embeddings)]
        SQLDB[(SQLite<br/>History + Cache)]
        CFGFILE[config.toml<br/>Provider Config]
    end

    %% User Interactions
    CLI --> ROUTER
    VSCODE --> ROUTER
    INTELLIJ --> ROUTER

    %% Router to Features
    ROUTER --> GEN
    ROUTER --> REF
    ROUTER --> DOC
    ROUTER --> ERR

    %% Features to LLM Manager
    GEN --> MGR
    REF --> MGR
    DOC --> MGR
    ERR --> MGR

    %% LLM Manager Configuration
    CONFIG --> MGR
    CFGFILE --> CONFIG

    %% Provider Selection
    MGR --> CLAUDE
    MGR --> OPENAI
    MGR --> GEMINI
    MGR --> OLLAMA
    MGR --> LLAMA
    MGR --> CUSTOM

    %% Providers to External Services
    CLAUDE --> CLAUDE_API
    OPENAI --> OPENAI_API
    GEMINI --> GEMINI_API
    OLLAMA --> OLLAMA_SRV
    LLAMA --> LLAMA_SRV

    %% Features to Analysis
    GEN --> PARSER
    REF --> PARSER
    DOC --> PARSER
    ERR --> PARSER

    %% Parser outputs
    PARSER --> AST
    AST --> METRICS
    AST --> DEPS

    %% Memory System Integration
    MGR --> CACHE
    CACHE --> VECTOR
    VECTOR --> VECTDB
    MGR --> HISTORY
    HISTORY --> SQLDB
    DEPS --> GRAPH

    %% File System Access
    PARSER --> FS
    GEN --> FS
    REF --> FS
    DOC --> FS

    %% Styling
    classDef userInterface fill:#4A90E2,stroke:#2E5C8A,color:#fff
    classDef llmProvider fill:#50C878,stroke:#2E7D4E,color:#fff
    classDef localProvider fill:#FFB347,stroke:#CC8A2E,color:#fff
    classDef memory fill:#9B59B6,stroke:#6C3483,color:#fff
    classDef analysis fill:#E74C3C,stroke:#A93226,color:#fff
    classDef storage fill:#95A5A6,stroke:#5D6D7E,color:#fff
    classDef external fill:#F39C12,stroke:#B9770E,color:#fff

    class CLI,VSCODE,INTELLIJ userInterface
    class CLAUDE,OPENAI,GEMINI llmProvider
    class OLLAMA,LLAMA,CUSTOM localProvider
    class CACHE,VECTOR,GRAPH,HISTORY memory
    class PARSER,AST,METRICS,DEPS analysis
    class FS,VECTDB,SQLDB,CFGFILE storage
    class CLAUDE_API,OPENAI_API,GEMINI_API,OLLAMA_SRV,LLAMA_SRV external