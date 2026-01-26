
## Documentation

#### Architecture Globale
L'application s'articule autour de plusieurs composant clés qui communiquent entre eux pour offir une expérience fluide et performante.

1. #### Structure de Modules Principaux
Module Core (core/) - Le coeur de l'application contenant la logique métier fondamentale, incluant l'analyse syntaxique (AST parsing), l'indexation du code, la gestion du contexte projet.

Module LLM (llm/) - L'interface avec les modèles de langage (API Claude, mais extensible à d'autres), gérant le streaming, la gestion des tokens, et les retry avec backoff exponentiel.

Module Memory (memory/) - Le système de mémoire contextuelle utilisant une base vectorielle (qdrant ou milvus) pour l'embedding et la récupération sémantique, avec un cache LRU pour les requêtes fréquentes.

Module Analysis (analysis/) - Les outils d'analyse statique du code, détection d'erreurs (via clippy pour Rust, eslint pour JS, etc.), calcul de métriques de complexité, et analyse de dépendances.

Module Refactoring (refactoring/) - Les stratégies de refactoring automatisé, la détection de code smell, et les suggestions d'amélioration basées sur les patterns.

Module Documentation (docs/) - La génération de documentation avec extraction automatique de commentaires, génération de diagrammes (PlantUML, Mermaid), et création de documentation interactive.

Module CLI (cli/) - L'interface en ligne de commande avec parsing des arguments (clap), gestion des commandes interactives, et output formaté avec couleurs.

Module Plugins (plugins/) - Les intégrations IDE avec protocole LSP (Language Server Protocol) pour VS Code et IntelliJ, webhooks pour événements, et système de plugins extensible.



2. #### Architecture de la Mémoire Contextuelle
Le système de mémoire est crucial pour maintenir un contexte cohérent. Il fonctionne sur plusieurs niveaux :

Niveau 1 - Cache Court Terme : Stockage en mémoire RAM des 10 dernières interactions avec timestamps et métadonnées, accès ultra-rapide (< 1ms).

Niveau 2 - Index Vectoriel : Embeddings de tous les fichiers du projet, requêtes sémantiques pour retrouver le contexte pertinent, mise à jour incrémentale lors des modifications.

Niveau 3 - Graph de Connaissance : Relations entre fonctions, classes, modules, graphe de dépendances traversable, détection de patterns d'utilisation.

Niveau 4 - Historique Persistant : Base SQLite locale pour l'historique des sessions, export/import pour partage d'équipe, compression des anciennes données.


3. #### Flux d'Exécution Typique
Voici comment l'assistant traite une requête utilisateur :

Phase 1 - Réception : L'utilisateur soumet une commande via CLI ou IDE. Le parser identifie le type de requête (génération, analyse, refactoring, documentation). Le système charge le contexte du workspace.

Phase 2 - Analyse Contextuelle : Le module Memory récupère les fichiers pertinents via recherche vectorielle. L'AST des fichiers concernés est parsé et analysé. Le graphe de dépendances est construit pour comprendre les relations.

Phase 3 - Enrichissement du Prompt : Le contexte est assemblé en incluant les fichiers pertinents, l'historique récent, les conventions du projet (détectées automatiquement), et les erreurs courantes du projet.

Phase 4 - Génération LLM : La requête enrichie est envoyée au modèle avec streaming pour affichage progressif. Les réponses sont validées syntaxiquement avant présentation. Les suggestions sont classées par pertinence et confiance.

Phase 5 - Post-traitement : Le code généré est formaté selon les conventions du projet. Les tests sont suggérés ou générés automatiquement. La documentation inline est ajoutée. L'index vectoriel est mis à jour.


4. #### Fonctionnalités Clés

Génération de Code Contextuelle : Analyse du style de code existant (naming conventions, patterns architecturaux), respect des types et interfaces existants, suggestions de tests unitaires pertinents, détection automatique du framework/bibliothèques utilisés.

Détection d'Erreurs Avancée : Analyse statique multi-niveaux (syntaxe, sémantique, logique), détection de bugs potentiels (null pointer, race conditions), suggestions de corrections avec explication pédagogique, intégration avec les outils natifs du langage.

Refactoring Intelligent : Extraction de fonctions/méthodes avec analyse de portée, renommage safe avec vérification des références, simplification de la complexité cyclomatique, conversion vers patterns modernes du langage.

Documentation Automatique : Génération de README avec description, installation, usage, diagrammes UML générés depuis le code (classes, séquences, composants), documentation API avec exemples d'utilisation, tutoriels interactifs pour nouveaux contributeurs.

Compréhension de Code : Mode "onboarding" pour nouveaux développeurs avec parcours guidé du codebase, visualisation interactive des flux d'exécution, explication en langage naturel de portions complexes, identification des points d'entrée et patterns clés.


5. #### Stack Technologique

Langage Principal : Rust (édition 2021+) pour performance et sécurité mémoire.

Parsing & AST : tree-sitter pour parsing multi-langage performant, syn pour parsing Rust spécifique, support de 20+ langages populaires.

Base Vectorielle : qdrant embedded ou milvus lite, fastembed pour génération d'embeddings, dimension 384 ou 768 selon le modèle.

CLI : clap v4 pour parsing des arguments, indicatif pour progress bars et spinners, colored pour output formaté.

LSP : tower-lsp pour l'implémentation du protocole, async runtime avec tokio, support WebSocket pour IDE web.

Persistance : SQLite avec sqlx pour historique et cache, serde pour sérialisation/désérialisation, compression avec zstd.

Tests : proptest pour property-based testing, criterion pour benchmarks de performance, couverture > 80%.


6. #### Interface Utilisateur

##### Mode CLI interactif:

$ codecraft init
✓ Projet analysé : 234 fichiers, 45k lignes
✓ Index créé : 1,245 embeddings

$ codecraft ask "Comment fonctionne l'authentification ?"
🔍 Analyse du contexte...
📝 L'authentification utilise JWT avec refresh tokens.
   Fichiers clés : auth/jwt.rs, middleware/auth.rs
   
   [Diagramme de séquence]
   
   Voulez-vous voir le code détaillé ? (o/n)

#### Commandes Principales :

init : Initialise l'assistant dans le projet

ask <question> : Pose une question sur le code

generate <description> : Génère du nouveau code

refactor <path> : Refactorise un fichier/module

document : Génère la documentation complète

analyze : Analyse complète avec métriques

explain <path> : Explique un fichier en détail

Intégration VS Code : Commandes dans la palette (Ctrl+Shift+P), panneau latéral avec arbre de navigation, diagnostics inline avec quick fixes, hover pour explications contextuelles.

Intégration IntelliJ : Actions dans le menu Tools, toolwindow dédiée, inspections personnalisées, intentions avec Alt+Enter.


7. #### Performance et Scalabilité
Optimisations : Traitement parallèle des fichiers (rayon), indexation incrémentale (seulement les fichiers modifiés), cache multi-niveaux (mémoire → disque → API), compilation en release avec LTO et optimisations agressives.

Métriques Cibles : Initialisation d'un projet 10k fichiers : < 30s, requête simple avec contexte : < 2s, génération de documentation complète : < 5min, utilisation mémoire : < 500MB pour projet moyen.