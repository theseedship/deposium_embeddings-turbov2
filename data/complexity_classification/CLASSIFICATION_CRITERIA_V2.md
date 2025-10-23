# Critères de classification de complexité documentaire - V2 STRICT

**Version** : 2.0
**Date** : 2025-10-23
**Usage** : Dataset 500 images pour distillation CLIP → ResNet18

---

## 🎯 Objectif

Classifier automatiquement les documents pour routing OCR/VLM avec **HAUTE PRÉCISION** :
- **LOW complexity** → OCR simple (~100ms)
- **HIGH complexity** → VLM reasoning (~2000ms)

**Métrique prioritaire** : **HIGH recall = 100%** (JAMAIS manquer un document complexe)

---

## 📋 Critères STRICTS de classification

### ✅ LOW Complexity (Simple - Route to OCR)

Documents contenant **UNIQUEMENT du texte imprimé** sans AUCUN élément visuel.

#### Règle d'or LOW :
**"Si un OCR simple peut extraire 100% de l'information sans raisonnement visuel → LOW"**

#### Caractéristiques STRICTES LOW :
- ✅ Texte en paragraphes (**imprimé uniquement, PAS manuscrit**)
- ✅ Listes à puces **textuelles** (symboles simples : •, -, *, chiffres)
- ✅ Titres et sous-titres (texte pur)
- ✅ Formulaires simples avec **lignes vides** pour remplir (pas de cases)
- ✅ Lettres officielles **sans logo** ni en-tête graphique
- ✅ Emails **purement textuels** (pas de signature image)

#### Exemples précis LOW :
1. Page de livre avec paragraphes de texte imprimé (pas d'images)
2. Email texte avec listes à puces textuelles
3. Formulaire vierge avec lignes droites pour écrire
4. Lettre officielle avec texte uniquement
5. Document Word simple (paragraphes + titres)

#### ❌ Ce qui N'EST PAS LOW (→ HIGH) :
- ❌ **Texte manuscrit** (écriture à la main)
- ❌ **Tableaux** (même 2x2, même simple)
- ❌ **Logos** ou images (même petits)
- ❌ **Cases à cocher** ou radio buttons
- ❌ **Colonnes multiples** avec séparateurs visuels
- ❌ **Tout élément graphique** (lignes, encadrés, flèches, icônes)
- ❌ **Factures avec grilles** (même simples)

---

### 🔥 HIGH Complexity (Complex - Route to VLM)

Documents contenant **au moins UN** élément visuel nécessitant un raisonnement visuel.

#### Règle d'or HIGH :
**"Dès qu'il y a un graphique, une courbe, un axe, un schéma, une carte, un dessin → HIGH"**

#### Catégories HIGH (liste exhaustive) :

##### 1. Graphiques avec axes (PRIORITÉ ABSOLUE)
- ✅ **Courbes** avec abscisse/ordonnée
  - Même **SANS valeurs exactes** sur les axes
  - Échelle suggérée, lisible par l'humain
  - Ex: courbe de température, graphique de ventes
- ✅ **Bar charts** (graphiques à barres)
- ✅ **Line graphs** (graphiques linéaires)
- ✅ **Scatter plots** (nuages de points)
- ✅ **Pie charts** (camemberts)
- ✅ **Histogrammes**
- ✅ **Graphiques combinés** (barres + lignes)
- ✅ **Graphiques 3D**

##### 2. Schémas techniques
- ✅ **Schémas électroniques** (circuits, composants)
- ✅ **Schémas mécaniques** (pièces, assemblages)
- ✅ **Plans d'architecture** (bâtiments, salles)
- ✅ **Diagrammes de réseau** (réseaux informatiques)
- ✅ **Flowcharts** (organigrammes)
- ✅ **Diagrammes UML** (class, sequence, etc.)
- ✅ **Process diagrams** (BPMN, workflows)

##### 3. Cartes
- ✅ **Cartes géographiques** (pays, régions, villes)
- ✅ **Cartes topographiques** (reliefs, altitudes)
- ✅ **Plans de ville** (rues, quartiers)
- ✅ **Heat maps** (densité, température)
- ✅ **Cartes thématiques** (démographie, météo, etc.)

##### 4. Dessins et illustrations
- ✅ **Dessins techniques** (croquis, plans)
- ✅ **Illustrations** (schémas explicatifs)
- ✅ **Infographies** (visualisations de données)
- ✅ **Mind maps** (cartes mentales)
- ✅ **Timelines visuelles** (lignes de temps avec graphiques)

##### 5. Tableaux et grilles
- ✅ **Tableaux de données** (grilles avec headers)
- ✅ **Matrices numériques**
- ✅ **Calendriers** (grilles de dates)
- ✅ **Spreadsheets** (Excel, Calc)
- ✅ **Tableaux complexes** (merged cells, nested)

##### 6. Autres éléments visuels
- ✅ **Formulaires complexes** (cases à cocher, sections multiples)
- ✅ **Documents avec images** (photos, logos, icônes)
- ✅ **Tickets/Boarding passes** (codes-barres, QR codes, layouts)
- ✅ **Cartes d'identité** (photos, hologrammes)
- ✅ **Dashboards** (tableaux de bord)

---

## ⚖️ Cas limites → TOUJOURS HIGH

**Principe de précaution** : En cas de doute → **HIGH**

| Document | Classification | Raison |
|----------|----------------|---------|
| Facture avec ligne de séparation | **HIGH** | Ligne = élément graphique |
| Formulaire avec 1 case à cocher | **HIGH** | Case = élément visuel |
| Email avec logo | **HIGH** | Logo = image |
| Tableau 2x2 simple | **HIGH** | Grille = structure visuelle |
| Document multi-colonnes | **HIGH** | Layout complexe |
| Texte avec flèche → | **HIGH** | Flèche = élément graphique |

---

## 📊 Distribution cible du dataset

### 500 images : 200 LOW / 300 HIGH (40% / 60%)

| Split | LOW | HIGH | Total |
|-------|-----|------|-------|
| **Train** | 140 | 210 | 350 (70%) |
| **Val** | 30 | 45 | 75 (15%) |
| **Test** | 30 | 45 | 75 (15%) |
| **TOTAL** | **200** | **300** | **500** |

### Justification du ratio 40/60 :
1. **Biais vers HIGH** pour garantir recall = 100%
2. **Class weights** dans la loss pour équilibrer
3. **Reflète la réalité** : documents modernes ont plus de visuels

---

## 🎯 Priorités pour les images HIGH

### Distribution recommandée des 300 images HIGH :

| Catégorie | Nombre | Pourcentage | Priorité |
|-----------|--------|-------------|----------|
| **Graphiques avec axes** | 90 | 30% | ⭐⭐⭐ CRITIQUE |
| Courbes sans valeurs | 30 | 10% | ⭐⭐⭐ |
| Bar/Pie charts | 30 | 10% | ⭐⭐⭐ |
| Scatter/autres graphs | 30 | 10% | ⭐⭐ |
| **Schémas techniques** | 60 | 20% | ⭐⭐⭐ |
| **Cartes** | 45 | 15% | ⭐⭐ |
| **Tableaux** | 45 | 15% | ⭐⭐ |
| **Dessins/Infographies** | 30 | 10% | ⭐ |
| **Formulaires complexes** | 30 | 10% | ⭐ |

**Total** : 300 images HIGH

---

## 🚀 Génération des images

### LOW (200 images) :
- Générateur de texte aléatoire (paragraphes, listes)
- Pas de lignes de séparation visuelles
- Pas d'encadrés
- Formulaires avec simples lignes horizontales

### HIGH (300 images) :
- **Matplotlib** : graphiques avec axes clairement visibles
- **Pillow** : tableaux, schémas simples
- **Emphasis sur** :
  - Courbes avec axes X/Y bien marqués
  - Graphiques sans valeurs numériques (juste échelle visuelle)
  - Cartes avec grilles de coordonnées
  - Schémas techniques avec composants

---

## ✅ Checklist de validation

- [ ] 500 images générées (200 LOW / 300 HIGH)
- [ ] **Aucune ambiguïté** : LOW = texte pur, HIGH = dès qu'il y a du visuel
- [ ] **Graphiques avec axes** : 90+ images
- [ ] **Courbes sans valeurs** : 30+ images (échelle suggérée)
- [ ] **Schémas techniques** : 60+ images
- [ ] **Variété** : bar, line, pie, scatter, maps, diagrams, tables
- [ ] **Annotations correctes** : vérification manuelle échantillon

---

**Auteur** : Claude Code
**Date** : 2025-10-23
**Version** : 2.0 - STRICT CRITERIA
