# Analyse du modèle VL actuel - Diagnostic complet

**Date** : 2025-10-23
**Modèle testé** : `src/models/complexity_classifier/model_quantized.onnx`
**Architecture** : ResNet18 ONNX INT8 quantized (10.7 MB)

---

## 🔍 Résumé exécutif

Le modèle VL actuel présente un **problème critique de HIGH recall (66.7%)**, manquant 33% des documents complexes. Le modèle est également **incertain** dans ses prédictions (probabilités proches de 50/50).

### Problèmes identifiés

1. ❌ **HIGH recall insuffisant : 66.7%** (cible : 100%)
   - Le modèle manque 2 images sur 6 documents complexes
   - Erreurs sur : bar_chart, table

2. ⚠️ **Modèle incertain**
   - Probabilités moyennes : LOW=45.4%, HIGH=54.6%
   - Confiance faible sur la plupart des prédictions

3. ⚠️ **Dataset d'entraînement probablement déséquilibré**
   - Trop d'exemples LOW dans le dataset original
   - Le modèle a appris un biais vers LOW

---

## 📊 Résultats des tests

### Statistiques globales

| Métrique | Valeur | Commentaire |
|----------|--------|-------------|
| **Accuracy** | 75.0% (6/8) | Acceptable mais pas excellent |
| **LOW recall** | 100% (2/2) | Parfait |
| **HIGH recall** | 66.7% (4/6) | 🚨 CRITIQUE - manque 33% |
| **Avg P(LOW)** | 0.4539 | Légèrement biaisé vers LOW |
| **Avg P(HIGH)** | 0.5461 | - |

### Résultats détaillés par image

| Image | Type | Attendu | Prédit | P(LOW) | P(HIGH) | Correct |
|-------|------|---------|--------|--------|---------|---------|
| plain_text.png | Texte simple | LOW | LOW | 0.599 | 0.401 | ✅ |
| **bar_chart.png** | **Bar chart** | **HIGH** | **LOW** | **0.522** | **0.478** | **❌** |
| line_graph.png | Line graph | HIGH | HIGH | 0.316 | 0.684 | ✅ |
| pie_chart.png | Pie chart | HIGH | HIGH | 0.382 | 0.618 | ✅ |
| **table.png** | **Table** | **HIGH** | **LOW** | **0.559** | **0.441** | **❌** |
| map.png | Map | HIGH | HIGH | 0.320 | 0.680 | ✅ |
| diagram.png | Flowchart | HIGH | HIGH | 0.314 | 0.686 | ✅ |
| simple_form.png | Form | LOW | LOW | 0.619 | 0.382 | ✅ |

---

## 🎯 Analyse détaillée

### Erreurs critiques

#### 1. Bar Chart (Bar chart avec barres colorées)
- **Prédit** : LOW (52.2%)
- **Attendu** : HIGH
- **Analyse** : Le modèle est très incertain (52% vs 48%), et prédit incorrectement LOW
- **Impact** : Un bar chart serait routé vers OCR au lieu de VLM (perte de qualité d'extraction)

#### 2. Table (Tableau avec grille)
- **Prédit** : LOW (55.9%)
- **Attendu** : HIGH
- **Analyse** : Le modèle est incertain (55.9% vs 44.1%), prédit LOW
- **Impact** : Un tableau serait routé vers OCR (risque de perte de structure)

### Images correctement classifiées

- **Line graph** : HIGH avec 68.4% confiance ✅
- **Pie chart** : HIGH avec 61.8% confiance ✅
- **Map** : HIGH avec 68.0% confiance ✅
- **Diagram** : HIGH avec 68.6% confiance ✅
- **Plain text** : LOW avec 59.9% confiance ✅
- **Simple form** : LOW avec 61.9% confiance ✅

---

## 🔬 Diagnostic du biais

### Pas de biais sévère global
- Le modèle ne prédit pas systématiquement LOW ou HIGH
- 4 LOW, 4 HIGH sur 8 images (équilibré en surface)

### MAIS : Problème d'incertitude
- **Probabilités trop proches de 50/50**
- Moyenne des probabilités : 45.4% LOW, 54.6% HIGH
- Le modèle n'a pas de conviction forte

### Causes probables

1. **Dataset d'entraînement déséquilibré**
   - Trop d'exemples LOW
   - Pas assez de variété dans les exemples HIGH (charts, graphs, tables)

2. **Threshold de décision non optimal**
   - Threshold actuel : 50% (softmax standard)
   - Devrait être ajusté pour favoriser HIGH (ex: 40% threshold)

3. **Features pas assez discriminantes**
   - Le modèle ResNet18 n'a peut-être pas appris les bonnes features
   - CLIP pourrait mieux capturer la "complexité visuelle"

---

## 💡 Recommandations

### Priorité 1 : Recréer le dataset (Phase 2)

**Objectif** : Dataset équilibré avec critères clairs

- **Ratio** : 50/50 ou 40 LOW / 60 HIGH (légèrement biaisé vers HIGH)
- **Taille** : 500-1000 images minimum (250-500 par classe)
- **Qualité** : Images réelles ou synthétiques de haute qualité

**Critères LOW** :
- Documents texte uniquement
- Formulaires simples (champs textuels)
- Pages de texte

**Critères HIGH** :
- **Graphiques** : bar charts, line graphs, pie charts, scatter plots
- **Tableaux** : grilles de données, matrices
- **Cartes** : géographiques, topographiques
- **Diagrammes** : flowcharts, architecture diagrams, mind maps
- **Infographies** : visualisations complexes
- **Documents mixtes** : texte + visuels

### Priorité 2 : Réentraîner avec CLIP (Phase 3)

**Approche recommandée** :
1. Utiliser **CLIP vision encoder** (ViT-B/32 ou ViT-L/14)
2. Freeze CLIP, entraîner classifier head
3. Binary classification avec **class weights** pour équilibrer
4. **Loss fonction** : CrossEntropyLoss avec poids [1.0, 1.5] (favorer HIGH)
5. **Métrique principale** : HIGH recall = 100%

**Hyperparamètres** :
- Optimizer : AdamW
- Learning rate : 1e-4
- Batch size : 16-32
- Epochs : 20-50 avec early stopping
- Class weights : [1.0, 1.5] pour favoriser HIGH

### Priorité 3 : Threshold tuning

Si le nouveau modèle a encore des problèmes de HIGH recall :
- Ajuster threshold de décision de 0.5 à 0.4 ou 0.35
- Favoriser HIGH même si confiance < 50%
- Accepter légère baisse de LOW precision pour HIGH recall = 100%

---

## 🚀 Next steps

1. ✅ **Phase 1 complète** : Diagnostic du modèle actuel
2. ⏭️ **Phase 2** : Créer dataset équilibré avec annotations LOW/HIGH
3. ⏭️ **Phase 3** : Entraîner nouveau modèle avec CLIP
4. ⏭️ **Phase 4** : Validation et déploiement

---

## 📎 Fichiers générés

- `test_current_classifier.py` : Script de test
- `classifier_test_results.json` : Résultats détaillés JSON
- `ANALYSIS_CURRENT_MODEL.md` : Ce document

---

**Conclusion** : Le modèle actuel est **insuffisant** pour un routing fiable (HIGH recall 66.7% vs cible 100%). Un réentraînement complet avec dataset équilibré est **nécessaire**.
