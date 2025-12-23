# Critères de classification de complexité documentaire

**Version** : 1.0
**Date** : 2025-10-23
**Usage** : Dataset pour entraînement du modèle VL de routing OCR/VLM

---

## 🎯 Objectif

Classifier automatiquement les documents pour router vers la pipeline optimale :
- **LOW complexity** → OCR simple (~100ms)
- **HIGH complexity** → VLM reasoning (~2000ms)

**Métrique prioritaire** : **HIGH recall = 100%** (ne jamais manquer un document complexe)

---

## 📋 Critères de classification

### ✅ LOW Complexity (Simple - Route to OCR)

Documents contenant **UNIQUEMENT du texte** sans éléments visuels complexes.

#### Caractéristiques :
- ✅ Texte en paragraphes
- ✅ Listes à puces ou numérotées
- ✅ Titres et sous-titres
- ✅ Formulaires simples (champs textuels uniquement)
- ✅ Documents Word/PDF textuels
- ✅ Pages de livres (texte uniquement)
- ✅ Factures simples (texte + montants)
- ✅ Lettres officielles
- ✅ Emails textuels
- ✅ Pages web textuelles

#### Exemples d'images LOW :
1. Page de livre (paragraphes de texte)
2. Email avec texte et listes
3. Formulaire simple avec champs texte
4. Facture simple (lignes de texte + total)
5. Lettre officielle
6. Document Word textuel
7. Page web article (texte uniquement)

#### Contre-exemples (ce qui N'EST PAS LOW) :
- ❌ Formulaires avec cases à cocher/radio buttons
- ❌ Documents avec logos/images (même petits)
- ❌ Pages avec tableaux complexes
- ❌ Documents avec mise en page en colonnes multiples

---

### 🔥 HIGH Complexity (Complex - Route to VLM)

Documents contenant **au moins UN** élément visuel complexe nécessitant un raisonnement visuel.

#### Caractéristiques :

##### 1. Graphiques (Charts)
- ✅ **Bar charts** (graphiques à barres)
- ✅ **Line graphs** (graphiques linéaires)
- ✅ **Pie charts** (camemberts)
- ✅ **Scatter plots** (nuages de points)
- ✅ **Histogrammes**
- ✅ **Graphiques combinés** (barres + lignes)
- ✅ **Graphiques 3D**

##### 2. Tableaux (Tables)
- ✅ **Tableaux de données** (grilles avec headers)
- ✅ **Matrices** (données numériques)
- ✅ **Tableaux complexes** (merged cells, nested tables)
- ✅ **Spreadsheets** (feuilles Excel/Calc)
- ✅ **Calendriers** (grilles de dates)

##### 3. Cartes (Maps)
- ✅ **Cartes géographiques**
- ✅ **Cartes topographiques**
- ✅ **Plans de ville**
- ✅ **Cartes thématiques** (météo, population, etc.)
- ✅ **Heat maps**

##### 4. Diagrammes (Diagrams)
- ✅ **Flowcharts** (organigrammes)
- ✅ **Architecture diagrams** (systèmes, réseaux)
- ✅ **UML diagrams** (class, sequence, etc.)
- ✅ **Mind maps** (cartes mentales)
- ✅ **Organization charts** (organigrammes hiérarchiques)
- ✅ **Process diagrams** (BPMN, etc.)
- ✅ **Circuit diagrams** (électroniques)
- ✅ **Venn diagrams**

##### 5. Infographies
- ✅ **Infographies** (visualisations de données complexes)
- ✅ **Timelines** (lignes de temps visuelles)
- ✅ **Dashboards** (tableaux de bord)
- ✅ **Schémas explicatifs**

##### 6. Visuels mixtes
- ✅ Documents avec **images + texte**
- ✅ **Présentations** (slides avec graphiques)
- ✅ **Rapports** avec visualisations
- ✅ **Articles scientifiques** avec figures
- ✅ **Magazines** avec photos et graphiques
- ✅ **Brochures** marketing

##### 7. Autres éléments complexes
- ✅ **Formulaires complexes** (cases à cocher, radio buttons, sections)
- ✅ **Tickets/Boarding passes** (codes-barres, QR codes, layouts complexes)
- ✅ **Cartes d'identité/Passeports** (photos, hologrammes, layouts structurés)
- ✅ **Menus** de restaurant (colonnes, sections, prix, images)
- ✅ **Catalogues** produits (grilles de produits avec images)

#### Règle simple : "Si un humain a besoin de regarder attentivement pour comprendre → HIGH"

---

## ⚖️ Cas limites (Edge cases)

### Borderline cases → Classifier comme HIGH (principe de précaution)

| Document | Classification | Raison |
|----------|----------------|---------|
| Facture avec logo | **HIGH** | Présence d'image (logo) |
| Formulaire avec checkbox | **HIGH** | Élément visuel (cases) |
| Email avec signature image | **HIGH** | Présence d'image |
| Document multi-colonnes | **HIGH** | Layout complexe |
| Page web avec menu | **HIGH** | Structure visuelle |
| Tableau simple (2x2) | **HIGH** | Structure tabulaire |
| Liste avec puces graphiques | **LOW** | Si puces sont juste des symboles texte |
| Liste avec icônes | **HIGH** | Si icônes sont des images |

**Règle d'or** : En cas de doute → **HIGH** (mieux router vers VLM inutilement que manquer un document complexe)

---

## 📊 Distribution cible du dataset

### Ratio recommandé : 40% LOW / 60% HIGH

**Justification** :
1. Légèrement biaisé vers HIGH pour garantir recall = 100%
2. Reflète la réalité des documents modernes (plus de visuels)
3. Compense le biais naturel des modèles vers la classe majoritaire

### Taille minimale

| Split | LOW | HIGH | Total |
|-------|-----|------|-------|
| **Train** | 200 | 300 | 500 |
| **Val** | 50 | 75 | 125 |
| **Test** | 50 | 75 | 125 |
| **TOTAL** | **300** | **450** | **750** |

**Recommandation** : Viser **1000 images** (400 LOW / 600 HIGH) pour meilleure robustesse

---

## 🎯 Métriques de qualité du dataset

### Checklist de validation

- [ ] **Équilibre** : 40/60 ou 50/50 LOW/HIGH ✅
- [ ] **Variété HIGH** :
  - [ ] Au moins 50 charts différents
  - [ ] Au moins 50 tables différentes
  - [ ] Au moins 30 maps
  - [ ] Au moins 40 diagrams
  - [ ] Au moins 30 infographies
  - [ ] Au moins 50 documents mixtes
- [ ] **Variété LOW** :
  - [ ] Au moins 100 pages texte pure
  - [ ] Au moins 50 formulaires simples
  - [ ] Au moins 50 documents officiels (lettres, factures)
- [ ] **Qualité** :
  - [ ] Images claires (pas de bruit excessif)
  - [ ] Résolution suffisante (min 224x224 après crop)
  - [ ] Annotations correctes (vérification manuelle échantillon)
- [ ] **Diversité** :
  - [ ] Langues variées (anglais, français, espagnol, etc.)
  - [ ] Styles variés (moderne, ancien, manuscrit, etc.)
  - [ ] Formats variés (portrait, paysage, carré)

---

## 🚀 Pipeline de génération

### 1. Images synthétiques (50% du dataset)

**Avantages** :
- ✅ Contrôle total sur les labels
- ✅ Diversité garantie
- ✅ Rapide à générer

**Bibliothèques** :
- `matplotlib` : charts, graphs
- `PIL/Pillow` : tables, forms, text layouts
- `plotly` : 3D charts, interactive graphs
- `faker` : données réalistes

### 2. Datasets publics (30% du dataset)

**Sources** :
- **DocVQA** : documents variés avec questions
- **ChartQA** : charts avec questions
- **PlotQA** : graphiques avec données
- **InfographicsVQA** : infographies
- **TabFact** : tableaux de données
- **TextVQA** : documents avec texte + images

### 3. Web scraping (20% du dataset)

**Sources** :
- Wikimedia Commons (images libres)
- Google Images (filtré CC0/Public Domain)
- Archive.org (documents historiques)

---

## 📝 Format des annotations

### Fichier : `annotations.csv`

```csv
image_path,label,category,description
train/low_001.png,0,text,Page de livre avec paragraphes
train/high_001.png,1,bar_chart,Bar chart ventes mensuelles
train/high_002.png,1,table,Tableau de données 5x10
val/low_001.png,0,form,Formulaire simple champs texte
```

### Labels :
- **0** : LOW complexity
- **1** : HIGH complexity

### Categories (optionnel, pour analyse) :
**LOW** : `text`, `form`, `letter`, `email`, `simple_document`
**HIGH** : `bar_chart`, `line_graph`, `pie_chart`, `scatter_plot`, `table`, `map`, `flowchart`, `diagram`, `infographic`, `mixed`, `complex_form`

---

## ✅ Checklist de validation finale

Avant utilisation du dataset pour entraînement :

- [ ] **750-1000 images** générées
- [ ] **Ratio 40/60 ou 50/50** LOW/HIGH respecté
- [ ] **Annotations** correctes (vérification manuelle 10%)
- [ ] **Diversité** des catégories HIGH (charts, tables, maps, diagrams, etc.)
- [ ] **Qualité** des images (résolution, clarté)
- [ ] **Split train/val/test** respecté (70/15/15)
- [ ] **Format standardisé** (224x224 après preprocessing)

---

**Auteur** : Claude Code
**Date** : 2025-10-23
**Version** : 1.0
