# 🚀 CodeCrafter - Assistant de Code IA Multi-Provider

<div align="center">

![CodeCrafter Logo](docs/assets/logo.png)

**Un assistant de code puissant et flexible avec support multi-LLM**

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/votre-org/codecraft/ci.yml?branch=main)](https://github.com/BossiaCI/codecrafter/actions)
[![Crates.io](https://img.shields.io/crates/v/codecrafter.svg)](https://crates.io/crates/codecrafter)

[Documentation](https://docs.codecrafter.dev) | [Guide de Démarrage](#installation) | [Exemples](#exemples-dutilisation)

</div>

## 📋 Table des Matières

- [Vue d'ensemble](#-vue-densemble)
- [Fonctionnalités](#-fonctionnalités)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Providers Supportés](#-providers-supportés)
- [Architecture](#-architecture)
- [Contribution](#-contribution)
- [License](#-license)

## 🎯 Vue d'ensemble

CodeCrafter est un assistant de code IA de nouvelle génération qui combine la puissance de plusieurs modèles de langage (Claude, GPT-4, Gemini, modèles locaux) avec des analyses statiques avancées pour offrir une expérience de développement exceptionnelle.

### Pourquoi CodeCrafter ?

- **🔀 Multi-Provider** : Choisissez entre Claude, GPT-4, Gemini, ou vos modèles locaux (Ollama, llama.cpp)
- **🏠 Local ou Cloud** : Travaillez offline avec des modèles locaux ou utilisez les APIs cloud
- **🧠 Mémoire Contextuelle** : Système de mémoire vectorielle qui se souvient de votre code
- **📊 Analyses Avancées** : Détection d'erreurs, métriques de complexité, graphes de dépendances
- **📚 Documentation Auto** : Génération de documentation complète avec diagrammes UML
- **🔄 Refactoring Intelligent** : Suggestions et application automatique de refactorings
- **🎓 Onboarding Rapide** : Aide les nouveaux développeurs à comprendre un codebase en heures, pas en jours
- **🔌 Intégrations IDE** : Support natif pour VS Code et IntelliJ via LSP

## ✨ Fonctionnalités

### 🤖 Génération de Code Contextuelle

```bash
codecrafter generate "Add JWT authentication middleware with refresh tokens"
```

- Analyse du style et conventions existants
- Respect des types et interfaces du projet
- Génération de tests automatique
- Documentation inline

### 🔍 Détection d'Erreurs Avancée

```bash
codecrafter analyze --deep
```

- Analyse statique multi-niveaux
- Détection de bugs potentiels (null pointers, race conditions)
- Suggestions de corrections avec explications
- Intégration avec clippy, eslint, etc.

### 🛠️ Refactoring Intelligent

```bash
codecrafter refactor src/legacy/ --strategy simplify
```

- Extraction de fonctions avec analyse de portée
- Renommage safe avec vérification des références
- Simplification de la complexité cyclomatique
- Migration vers patterns modernes

### 📖 Documentation Automatique

```bash
codecrafter document --full --output docs/
```

- README avec installation et usage
- Diagrammes UML (classes, séquences, composants)
- Documentation API avec exemples
- Tutoriels interactifs pour contributeurs

### 🎓 Compréhension de Code

```bash
codecrafter onboard
```

- Parcours guidé du codebase
- Visualisation des flux d'exécution
- Explications en langage naturel
- Identification des patterns clés

## 🚀 Installation

### Prérequis

- Rust 1.75 ou supérieur
- (Optionnel) Ollama pour modèles locaux
- (Optionnel) API keys pour providers cloud

### Via Cargo

```bash
cargo install codecrafter
```

### Depuis les Sources

```bash
git clone https://github.com/BossiaCI/CodeCrafter
cd codecrafter
cargo build --release
cargo install --path .
```

### Via Homebrew (macOS/Linux)

```bash
brew install codecrafter
```

### Binaires Précompilés

Téléchargez depuis [Releases](https://github.com/BossiaCI/CodeCrafter/releases)

## ⚙️ Configuration

### Configuration Rapide

```bash
# Initialiser la configuration
codecrafter init

# Ajouter un provider cloud
codecrafter provider add claude \
  --provider-type claude \
  --model claude-sonnet-4-5-20250929 \
  --deployment remote \
  --url https://api.anthropic.com \
  --api-key $ANTHROPIC_API_KEY \
  --default

# Ou un provider local
codecrafter provider add ollama \
  --provider-type ollama \
  --model codellama:13b \
  --deployment local \
  --url http://localhost:11434
```

### Fichier de Configuration

Le fichier de configuration se trouve à `~/.config/codecrafter/config.toml` :

```toml
default_provider = "claude"

[providers.claude]
provider_type = "claude"
model_name = "claude-sonnet-4-5-20250929"
deployment = "remote"
base_url = "https://api.anthropic.com"
api_key = "${ANTHROPIC_API_KEY}"

[providers.ollama-local]
provider_type = "ollama"
model_name = "deepseek-coder:33b"
deployment = "local"
base_url = "http://localhost:11434"
```

### Variables d'Environnement

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
export CODECRAFTER_PROVIDER="claude"
```

## 📖 Usage

### Commandes Principales

#### Initialisation

```bash
# Initialiser dans un projet
codecrafter init

# Avec provider spécifique
codecrafter init --provider ollama-local
```

#### Questions sur le Code

```bash
# Question simple
codecrafter ask "Comment fonctionne l'authentification ?"

# Avec provider spécifique
codecrafter ask "Explique ce module" --provider gpt4

# Question complexe avec analyse approfondie
codecrafter ask "Quels sont les points d'amélioration de performance ?" --deep
```

#### Génération de Code

```bash
# Générer une nouvelle fonctionnalité
codecrafter generate "Create a user management API with CRUD operations"

# Générer des tests
codecrafter generate tests src/auth.rs

# Générer du code dans un fichier spécifique
codecrafter generate "Add error handling" --file src/main.rs
```

#### Analyse de Code

```bash
# Analyse rapide
codecrafter analyze

# Analyse complète avec métriques
codecrafter analyze --deep --metrics

# Analyse d'un fichier spécifique
codecrafter analyze src/core/engine.rs
```

#### Refactoring

```bash
# Refactoring interactif
codecrafter refactor src/legacy/

# Application automatique
codecrafter refactor src/legacy/ --apply --strategy simplify

# Preview des changements
codecrafter refactor src/ --dry-run
```

#### Documentation

```bash
# Générer toute la documentation
codecrafter document --full

# Documentation d'un module
codecrafter document src/core/ --output docs/core.md

# Générer uniquement les diagrammes
codecrafter document --diagrams-only
```

#### Onboarding

```bash
# Lancer le mode onboarding
codecrafter onboard

# Générer un guide pour nouveaux contributeurs
codecrafter onboard --generate-guide --output CONTRIBUTING.md
```

### Gestion des Providers

```bash
# Lister les providers
codecrafter provider list

# Détails d'un provider
codecrafter provider show claude

# Tester un provider
codecrafter provider test ollama-local

# Définir le provider par défaut
codecrafter provider set-default gpt4

# Supprimer un provider
codecrafter provider remove old-provider
```

## 🌐 Providers Supportés

### Providers Cloud

| Provider | Modèles | Statut |
|----------|---------|--------|
| **Claude (Anthropic)** | Opus, Sonnet, Haiku | ✅ Complet |
| **OpenAI** | GPT-4, GPT-4 Turbo, GPT-3.5 | ✅ Complet |
| **Google Gemini** | Gemini Pro, Ultra | ✅ Complet |
| **Azure OpenAI** | GPT-4, GPT-3.5 | ✅ Complet |

### Providers Locaux

| Provider | Type | Statut |
|----------|------|--------|
| **Ollama** | Conteneur local | ✅ Complet |
| **llama.cpp** | Server natif | ✅ Complet |
| **Custom** | API personnalisée | ✅ Extensible |

### Ajouter un Provider Custom

```rust
use codecrafter::llm::{LLMProvider, LLMRequest, LLMResponse};

struct MyCustomProvider {
    // Configuration...
}

#[async_trait]
impl LLMProvider for MyCustomProvider {
    async fn generate(&self, request: LLMRequest) -> Result<LLMResponse, LLMError> {
        // Votre implémentation
    }
    // ... autres méthodes
}
```

## 🏗️ Architecture

### Vue d'ensemble

```
┌─────────────────────────────────────────────────┐
│              Interface Utilisateur              │
│  CLI │ VS Code Extension │ IntelliJ Plugin     │
└───────────────────┬─────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────┐
│              LLM Manager (Multi-Provider)       │
│  ┌──────────┬──────────┬──────────┬──────────┐ │
│  │ Claude   │ OpenAI   │ Gemini   │ Ollama   │ │
│  └──────────┴──────────┴──────────┴──────────┘ │
└───────────────────┬─────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────┐
│            Core Features & Analysis             │
│  ┌─────────────────────────────────────────┐   │
│  │ Memory System (Vector DB + Cache)      │   │
│  ├─────────────────────────────────────────┤   │
│  │ Code Parser (Tree-sitter)               │   │
│  ├─────────────────────────────────────────┤   │
│  │ Generator │ Refactor │ Docs │ Analysis  │   │
│  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

### Composants Principaux

- **LLM Layer** : Abstraction multi-provider avec fallback et retry
- **Memory System** : Cache multi-niveaux + Vector DB pour contexte sémantique
- **Code Analysis** : Tree-sitter pour parsing, graphes de dépendances
- **Features** : Modules pour génération, refactoring, documentation
- **CLI/LSP** : Interfaces utilisateur en ligne de commande et IDE

## 🔧 Développement

### Setup

```bash
git clone https://github.com/BossiaCI/CodeCrafter.git
cd codecrafter
cargo build
```

### Tests

```bash
# Tests unitaires
cargo test

# Tests d'intégration
cargo test --test integration

# Tests avec coverage
cargo tarpaulin --out Html
```

### Benchmarks

```bash
cargo bench
```

### Linting

```bash
cargo clippy --all-targets --all-features
```

## 📊 Performances

- **Initialisation** : < 30s pour 10k fichiers
- **Requête simple** : < 2s avec contexte
- **Documentation complète** : < 5min pour projet moyen
- **Mémoire** : < 500MB pour projet moyen

## 🗺️ Roadmap

### Version 0.2.0 (Q2 2025)

- [ ] Support de 10+ langages additionnels
- [ ] Intégration GitHub Copilot compatible
- [ ] Mode collaboratif multi-utilisateurs
- [ ] Plugin Neovim/Vim

### Version 0.3.0 (Q3 2025)

- [ ] Auto-apprentissage des patterns projet
- [ ] Génération de tests end-to-end
- [ ] Support des monorepos
- [ ] API REST pour intégrations custom

### Version 1.0.0 (Q4 2025)

- [ ] Système de plugins marketplace
- [ ] Support cloud natif (Kubernetes)
- [ ] Métriques et analytics avancées
- [ ] Enterprise features

## 🤝 Contribution

Les contributions sont les bienvenues ! Consultez [CONTRIBUTING.md](CONTRIBUTING.md) pour les guidelines.

### Comment Contribuer

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📝 License

Ce projet est sous double licence :

- MIT License ([LICENSE-MIT](LICENSE-MIT))
- Apache License 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

Vous pouvez choisir celle qui convient le mieux à votre usage.

## 🙏 Remerciements

- [Anthropic](https://anthropic.com) pour Claude
- [OpenAI](https://openai.com) pour GPT
- [Google](https://deepmind.google) pour Gemini
- [Ollama](https://ollama.ai) pour l'infrastructure locale
- La communauté Rust pour les excellentes bibliothèques

## 📞 Support

- 📖 [Documentation](https://docs.codecraft.dev)
- 💬 [Discord](https://discord.gg/codecraft)
- 🐛 [Issues](https://github.com/votre-org/codecraft/issues)
- 📧 Email: support@codecraft.dev

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=votre-org/codecrafter&type=Date)](https://star-history.com/#votre-org/codecrafter&Date)

---

<div align="center">
Made with ❤️ by the CodeCrafter Team
</div>