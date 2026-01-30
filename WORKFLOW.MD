sequenceDiagram
    participant User as 👤 Utilisateur
    participant CLI as CLI/IDE
    participant Router as Command Router
    participant Config as Config Manager
    participant LLMManager as LLM Manager
    participant Memory as Memory System
    participant Parser as Code Parser
    participant Provider as Provider (Claude/GPT/etc)
    participant API as API Externe/Local
    participant FileSystem as File System

    Note over User,FileSystem: Scénario: Génération de code avec sélection de provider

    User->>CLI: codecraft generate "Add auth middleware" --provider claude-remote
    activate CLI
    
    CLI->>Router: parse_command(args)
    activate Router
    
    Router->>Config: load_config()
    activate Config
    Config->>FileSystem: read config.toml
    FileSystem-->>Config: configuration
    Config-->>Router: LLMConfig
    deactivate Config
    
    Router->>LLMManager: new(config)
    activate LLMManager
    
    LLMManager->>LLMManager: initialize_providers()
    
    alt Provider spécifié par l'utilisateur
        LLMManager->>Provider: create_provider("claude-remote")
        activate Provider
        Provider->>Provider: validate_config()
        Provider-->>LLMManager: ClaudeProvider
        deactivate Provider
    else Provider par défaut
        LLMManager->>LLMManager: get_default_provider()
    end
    
    LLMManager-->>Router: manager
    deactivate LLMManager
    
    Router->>Router: analyze_workspace()
    activate Router
    
    Router->>FileSystem: list_project_files()
    FileSystem-->>Router: file_list
    
    Router->>Parser: parse_relevant_files()
    activate Parser
    Parser->>FileSystem: read_files()
    FileSystem-->>Parser: file_contents
    Parser->>Parser: build_ast()
    Parser->>Parser: extract_context()
    Parser-->>Router: ParsedContext
    deactivate Parser
    
    Router->>Memory: retrieve_relevant_context("auth middleware")
    activate Memory
    
    Memory->>Memory: search_vector_db()
    Memory->>Memory: get_recent_history()
    Memory->>Memory: traverse_dependency_graph()
    
    Memory-->>Router: ContextBundle
    deactivate Memory
    
    deactivate Router
    
    Router->>Router: build_prompt(user_request, context)
    
    Router->>LLMManager: generate(request, provider="claude-remote")
    activate LLMManager
    
    LLMManager->>Provider: generate(enriched_request)
    activate Provider
    
    alt Provider Remote (Cloud)
        Provider->>Provider: prepare_api_request()
        Provider->>API: POST /v1/messages
        activate API
        
        alt API Key valide
            API->>API: authenticate()
            API->>API: process_request()
            
            alt Streaming activé
                loop Chunks de réponse
                    API-->>Provider: stream_chunk
                    Provider-->>LLMManager: StreamChunk
                    LLMManager-->>Router: partial_response
                    Router-->>CLI: display_progressively()
                    CLI-->>User: 📝 Code généré...
                end
            else Réponse complète
                API-->>Provider: complete_response
            end
            
        else Rate Limit / Erreur Temporaire
            API-->>Provider: 429 Too Many Requests
            Provider->>Provider: exponential_backoff()
            Provider->>API: retry_request()
        end
        
        deactivate API
        
    else Provider Local (Ollama/LlamaCpp)
        Provider->>API: POST /api/generate
        activate API
        Note over API: Serveur local<br/>localhost:11434
        API->>API: load_model()
        API->>API: inference()
        API-->>Provider: local_response
        deactivate API
    end
    
    Provider-->>LLMManager: LLMResponse
    deactivate Provider
    
    LLMManager->>LLMManager: validate_response()
    LLMManager-->>Router: validated_code
    deactivate LLMManager
    
    Router->>Router: post_process(code)
    Router->>Router: format_code()
    Router->>Router: add_documentation()
    
    Router->>Memory: update_context(new_code)
    activate Memory
    Memory->>Memory: update_vector_index()
    Memory->>Memory: store_in_history()
    Memory->>FileSystem: save_to_db()
    deactivate Memory
    
    Router-->>CLI: GeneratedCode + metadata
    deactivate Router
    
    CLI->>FileSystem: write_file(path, code)
    FileSystem-->>CLI: success
    
    CLI-->>User: ✅ Code généré et sauvegardé
    deactivate CLI
    
    opt Validation optionnelle
        User->>CLI: codecraft test --validate
        CLI->>Router: run_tests()
        Router->>FileSystem: execute_test_suite()
        FileSystem-->>Router: test_results
        Router-->>CLI: validation_report
        CLI-->>User: 📊 Tests: 15/15 passed
    end
    
    Note over User,FileSystem: Fin du processus - Code généré avec le provider sélectionné