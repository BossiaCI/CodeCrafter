
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


## Mode CLI interactif:

$ codecraft init
✓ Projet analysé : 234 fichiers, 45k lignes
✓ Index créé : 1,245 embeddings

$ codecraft ask "Comment fonctionne l'authentification ?"
🔍 Analyse du contexte...
📝 L'authentification utilise JWT avec refresh tokens.
   Fichiers clés : auth/jwt.rs, middleware/auth.rs
   
   [Diagramme de séquence]
   
   Voulez-vous voir le code détaillé ? (o/n)