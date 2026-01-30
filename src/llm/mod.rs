// Module principal pour la gestion des LLM (Large Language Models)

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::error::Error;
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
    pub headers: Option<HashMap<String, String>>,

    /// Paramètres spécifiques au provider/modèle
    pub parameters: ModelParameters,

    /// Timeout en secondes
    pub timeout_seconds: u64,

    /// Nombre de tentatives en cas d'échec
    pub max_retries: u32,

}