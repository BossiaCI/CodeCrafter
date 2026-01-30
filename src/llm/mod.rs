// Module principal pour la gestion des LLM (Large Language Models)

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod providers;
pub mod config;
pub mod streaming;


/// Type de provider LLM supporté
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum LLMProviderType {
    Claude,
    OpenAI,
    Gemini,
    Ollama, // Pour des modèles locaux
    LlamaCpp, // Pour des modèles locaux
    Mistral,
    AzureOpenAI,
    Custom, // Pour des providers personnalisés
}

/// Configuration d'un provider LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMProviderConfig {
    /// Type de provider
    pub provider_type: LLMProviderType,

    /// Nom du modèle (ex: "gpt-4", "claude-sonnet-4.5", etc.)
    pub model_name: String,

    /// Mode local ou distant
    pub deployment: DeploymentMode,

    /// URL de base de l'API (pour les providers distants)
    pub base_url: Option<String>,

    /// Clé API (pour les providers distants)
    pub api_key: Option<String>,

    /// Headers additionnels pour les requêtes API
    pub headers: HashMap<String, String>,

    /// Paramètres spécifiques au provider/modèle
    pub parameters: ModelParameters,

    /// Timeout en secondes
    pub timeout_seconds: u64,

    /// Nombre de tentatives en cas d'échec
    pub max_retries: u32,

}


/// Mode de déploiement du modèle
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum DeploymentMode {
    /// Modèle exécuté localement
    Local,

    /// Modèle exécuté à distance via une API
    Remote,

    /// Détection automatique du mode basée sur l'URL ou la configuration
    Auto,
}


/// Paramètres spécifiques au modèle LLM / Paramètres de génération du modèle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelParameters {
    /// Température pour la génération de texte (0.0 - 2.0)
    pub temperature: f32,

    /// Top P (sampling) pour la génération de texte (0.0 - 1.0) / Nucleus Sampling
    pub top_p: f32,

    /// Nombre maximal de tokens à générer
    pub max_tokens: u32,


    /// Présence de pénalité
    pub presence_penalty: f32,

    /// Fréquence de pénalité
    pub frequency_penalty: f32,

    /// Stop sequences pour arrêter la génération
    pub stop_sequences: Vec<String>,
}

impl Default for ModelParameters {
    fn default() -> Self {
        ModelParameters {
            temperature: 0.7,
            top_p: 0.95,
            max_tokens: 4096,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            stop_sequences: vec![],
        }
    }
}


/// Message dans une conversation avec le LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMMessage {
    /// Rôle de l'auteur du message (user, assistant, system)
    pub role: Role,
    /// Contenu du message
    pub content: String,
    /// Métadonnées additionnelles (optionnel)
    pub metadata: Option<HashMap<String, String>>,
}


/// Rôle de l'auteur du message
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
    System,
}



/// Requête pour générer une réponse du LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMRequest {
    /// Messages de la conversation
    pub messages: Vec<LLMMessage>,
    /// Paramètres spécifiques au modèle
    pub parameters: Option<ModelParameters>,
    /// Indicateur de streaming
    pub stream: bool,
}

/// Réponse du LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMResponse {
    /// Contenu généré par le LLM
    pub content: String,
    /// Raison de fin de la génération (ex: stop sequence, max tokens, etc.)
    pub finish_reason: FinishReason,
    /// Utilisation des tokens (optionnel)
    pub usage: TokenUsage,
    /// Modele utilisé
    pub model: String,
    /// Métadonnées additionnelles (optionnel)
    pub metadata: Option<HashMap<String, String>>,
}


/// Raison de fin de la génération
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FinishReason {
    Stop,
    Length,
    ContentFilter,
    ToolUse,
}


/// Utilisation des tokens dans la requête/réponse
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Nombre de tokens dans la requête
    pub prompt_tokens: u32,
    /// Nombre de tokens dans la réponse
    pub completion_tokens: u32,
    /// Nombre total de tokens utilisés
    pub total_tokens: u32,
}



/// Chunk de la réponse en streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMStreamChunk {
    /// Contenu partiel généré
    pub delta: String,
    /// Raison de fin de la génération (optionnel)
    pub finish_reason: Option<FinishReason>,
    /// Métadonnées additionnelles (optionnel)
    pub metadata: Option<HashMap<String, String>>,
}


/// Trait principal pour tous les providers LLM
#[async_trait]
pub trait LLMProvider: Send + Sync {
    /// Générer une réponse du LLM (non streaming) complète
    async fn generate(&self, request: LLMRequest) -> Result<LLMResponse, LLMError>;

    /// Générer une réponse du LLM en streaming
    async fn generate_stream(
        &self,
        request: LLMRequest,
    ) -> Result<Box<dyn futures::stream<Item = Result<LLMStreamChunk, LLMError>> + Unpin + Send>, LLMError>;

    /// Compte les tokens dans une liste de messages
    fn count_tokens(&self, text: &str) -> Result<u32, LLMError>;

    /// Retourne le nom du provider 
    fn provider_name(&self) -> &str;

    /// Retourne le nom du modèle
    fn model_name(&self) -> &str;

    /// Vérifie que le provider est configuré correctement
    async fn health_check(&self) -> Result<(), LLMError>;
}



/// Erreur générique pour les opérations LLM
#[derive(Debug, thiserror::Error)]
pub enum LLMError {
    #[error("Configuration invalide: {0}")]
    InvalidConfig(String),
    
    #[error("Erreur d'authentification: {0}")]
    AuthenticationError(String),
    
    #[error("Erreur réseau: {0}")]
    NetworkError(String),
    
    #[error("Erreur API: {status} - {message}")]
    APIError { status: u16, message: String },
    
    #[error("Limite de tokens dépassée")]
    TokenLimitExceeded,
    
    #[error("Timeout de la requête")]
    Timeout,
    
    #[error("Modèle non trouvé: {0}")]
    ModelNotFound(String),
    
    #[error("Erreur de parsing: {0}")]
    ParseError(String),
    
    #[error("Erreur interne: {0}")]
    InternalError(String),
}

