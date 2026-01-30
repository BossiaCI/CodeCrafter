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
    pub headers: Option<HashMap<String, String>>,

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
    pub temperature: Option<f32>,

    /// Top P (sampling) pour la génération de texte (0.0 - 1.0) / Nucleus Sampling
    pub top_p: Option<f32>,

    /// Nombre maximal de tokens à générer
    pub max_tokens: Option<u32>,


    /// Présence de pénalité
    pub presence_penalty: Option<f32>,

    /// Fréquence de pénalité
    pub frequency_penalty: Option<f32>,

    /// Stop sequences pour arrêter la génération
    pub stop_sequences: Option<Vec<String>>,
}