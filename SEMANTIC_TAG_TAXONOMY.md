# Semantic Tag Taxonomy for Enhanced Retrieval

## Document-Level Semantic Tags

### 🧠 Introspective Analysis (Internal/Subjective)
```json
{
  "introspective": {
    "aspect": "string",           // What internal dimension? (reasoning, reflection, analysis, evaluation)
    "details": "string",           // Specific introspective content
    "sentiment": "string",         // Internal emotional tone (thoughtful, concerned, optimistic, skeptical)
    "entities": ["string"],        // Internal references (concepts, theories, mental models)
    "impact": "string",            // Internal significance (insight, realization, question, hypothesis)
    "contextual_impact": "string"  // How it affects understanding/perspective
  }
}
```

**Examples:**
- `aspect`: "critical analysis", "self-reflection", "hypothesis formation"
- `sentiment`: "intellectually curious", "methodologically cautious"
- `impact`: "paradigm shift", "conceptual breakthrough", "methodological concern"

---

### 🌍 Extrospective Analysis (External/Objective)
```json
{
  "extrospective": {
    "aspect": "string",           // What external dimension? (observation, fact, event, measurement)
    "details": "string",           // Specific factual content
    "sentiment": "string",         // Objective tone (neutral, descriptive, factual, alarming)
    "entities": ["string"],        // External entities (people, orgs, places, objects)
    "impact": "string",            // External significance (consequence, outcome, change)
    "contextual_impact": "string"  // Broader implications
  }
}
```

**Examples:**
- `aspect`: "empirical observation", "statistical finding", "policy change"
- `sentiment`: "factually neutral", "urgently concerning", "positively transformative"
- `impact`: "regulatory change", "market disruption", "environmental damage"

---

### 📋 Core Metadata
```json
{
  "metadata": {
    "title": "string",              // Document/section title
    "overview": "string",           // Brief summary
    "document_type": "string",      // article, report, legal, contract, analysis
    "domain": "string",             // law, finance, science, policy, business
    "confidence_score": "float"     // 0-1, confidence in tag accuracy
  }
}
```

---

### ⏰ Temporal Context
```json
{
  "temporal": {
    "primary_date": "ISO8601",      // Main event/publication date
    "date_range": {
      "start": "ISO8601",
      "end": "ISO8601"
    },
    "temporal_relevance": "string",  // "historical", "current", "future-oriented", "timeless"
    "temporal_markers": ["string"]   // "Q1 2024", "end of fiscal year", "long-term"
  }
}
```

---

### 📍 Geospatial Context
```json
{
  "geospatial": {
    "locations": [
      {
        "name": "string",           // "Paris", "European Union", "Asia-Pacific"
        "type": "string",           // city, region, country, continent, zone
        "relevance": "string"       // primary, secondary, mentioned
      }
    ],
    "geographic_scope": "string"    // local, regional, national, international, global
  }
}
```

---

### 🎭 Sentiment & Tone
```json
{
  "sentiment": {
    "overall": "string",            // positive, negative, neutral, mixed
    "polarity_score": "float",      // -1 to +1
    "subjectivity": "float",        // 0 (objective) to 1 (subjective)
    "emotional_tone": ["string"],   // urgent, cautious, optimistic, alarming, celebratory
    "stakeholder_sentiment": {      // Different perspectives
      "consumer": "string",
      "business": "string",
      "regulator": "string"
    }
  }
}
```

---

### 🏢 Entities & Actors
```json
{
  "entities": {
    "people": ["string"],           // Named individuals
    "organizations": ["string"],    // Companies, NGOs, governments
    "products": ["string"],         // Products, services, brands
    "legislation": ["string"],      // Laws, regulations, policies
    "concepts": ["string"]          // Abstract concepts, frameworks, theories
  }
}
```

---

### 🌳 Ontological Classification
```json
{
  "ontology": {
    "concepts": [
      {
        "term": "string",           // "carbon neutrality", "risk mitigation"
        "category": "string",       // sustainability, risk, compliance, innovation
        "definition": "string",     // Contextual definition
        "relationships": [
          {
            "type": "string",       // "causes", "requires", "conflicts_with", "enables"
            "target": "string"      // Related concept
          }
        ]
      }
    ],
    "taxonomies": ["string"],       // Domain-specific classification systems used
    "domain_keywords": ["string"]   // Key terms for domain filtering
  }
}
```

---

### ♻️ Sustainability & ESG
```json
{
  "sustainability": {
    "esg_dimensions": {
      "environmental": {
        "relevant": "boolean",
        "keywords": ["string"],     // emissions, renewable, biodiversity
        "impact_level": "string"    // high, medium, low
      },
      "social": {
        "relevant": "boolean",
        "keywords": ["string"],     // diversity, labor, community
        "impact_level": "string"
      },
      "governance": {
        "relevant": "boolean",
        "keywords": ["string"],     // compliance, transparency, ethics
        "impact_level": "string"
      }
    },
    "sdg_alignment": [              // UN Sustainable Development Goals
      {
        "goal_number": "integer",   // 1-17
        "goal_name": "string",
        "relevance": "string"       // primary, secondary, tangential
      }
    ],
    "sustainability_keywords": ["string"],
    "circular_economy": "boolean",
    "climate_action": "boolean"
  }
}
```

---

### ⚠️ Risk & Compliance
```json
{
  "risk": {
    "risk_types": ["string"],       // operational, financial, regulatory, reputational
    "risk_level": "string",         // low, medium, high, critical
    "mitigation_strategies": ["string"],
    "compliance_frameworks": ["string"],  // GDPR, SOX, ISO27001, TCFD
    "regulatory_impact": "string"   // none, moderate, significant, transformative
  }
}
```

---

### 🎯 Intent & Purpose
```json
{
  "intent": {
    "primary_purpose": "string",    // inform, persuade, analyze, report, regulate
    "target_audience": ["string"],  // investors, regulators, consumers, experts
    "action_items": ["string"],     // Specific actions recommended/required
    "urgency": "string"            // low, medium, high, immediate
  }
}
```

---

### 🔗 Relationships & Context
```json
{
  "relationships": {
    "related_documents": [
      {
        "id": "string",
        "relationship": "string",   // supersedes, references, contradicts, complements
        "relevance_score": "float"
      }
    ],
    "cited_sources": ["string"],
    "legal_precedents": ["string"],
    "industry_context": "string"
  }
}
```

---

## Combined Example

```json
{
  "metadata": {
    "title": "EU Carbon Border Adjustment Mechanism Impact Report",
    "overview": "Analysis of CBAM effects on European manufacturing",
    "document_type": "policy_analysis",
    "domain": "environmental_policy",
    "confidence_score": 0.92
  },

  "introspective": {
    "aspect": "critical_evaluation",
    "details": "Questioning effectiveness of phased implementation",
    "sentiment": "analytically_cautious",
    "entities": ["policy_design_flaw", "implementation_timeline"],
    "impact": "methodological_concern",
    "contextual_impact": "May require policy revision"
  },

  "extrospective": {
    "aspect": "regulatory_change",
    "details": "CBAM applies to cement, steel, aluminum from Oct 2023",
    "sentiment": "factually_neutral",
    "entities": ["European Commission", "manufacturing_sector"],
    "impact": "increased_compliance_costs",
    "contextual_impact": "Competitiveness concerns for EU industry"
  },

  "temporal": {
    "primary_date": "2024-03-15",
    "temporal_relevance": "current",
    "temporal_markers": ["transitional period 2023-2026", "full implementation 2026"]
  },

  "geospatial": {
    "locations": [
      {"name": "European Union", "type": "region", "relevance": "primary"},
      {"name": "China", "type": "country", "relevance": "secondary"}
    ],
    "geographic_scope": "international"
  },

  "sentiment": {
    "overall": "mixed",
    "polarity_score": 0.15,
    "emotional_tone": ["cautious", "forward-looking"],
    "stakeholder_sentiment": {
      "consumer": "neutral",
      "business": "concerned",
      "regulator": "optimistic"
    }
  },

  "sustainability": {
    "esg_dimensions": {
      "environmental": {
        "relevant": true,
        "keywords": ["carbon_leakage", "emissions_reduction", "climate_neutrality"],
        "impact_level": "high"
      }
    },
    "sdg_alignment": [
      {"goal_number": 13, "goal_name": "Climate Action", "relevance": "primary"}
    ],
    "climate_action": true
  },

  "ontology": {
    "concepts": [
      {
        "term": "carbon_border_adjustment",
        "category": "climate_policy",
        "relationships": [
          {"type": "aims_to_prevent", "target": "carbon_leakage"},
          {"type": "requires", "target": "emissions_monitoring"}
        ]
      }
    ]
  },

  "risk": {
    "risk_types": ["regulatory", "financial"],
    "risk_level": "high",
    "compliance_frameworks": ["EU_ETS", "CBAM"],
    "regulatory_impact": "transformative"
  }
}
```

---

## Usage for Filtered Retrieval

**Query Examples:**
1. "Find documents with high environmental impact from 2024 about EU policy with concerned business sentiment"
2. "Show introspective analyses with methodological concerns about climate action in Asia-Pacific"
3. "Retrieve high-risk regulatory documents affecting the manufacturing sector with urgent action items"

---

## Implementation Notes

- **Extraction Pipeline**: NER → Sentiment Analysis → Temporal/Geo Extraction → Ontology Mapping
- **Storage**: JSON metadata stored alongside vector embeddings
- **Indexing**: Multi-dimensional indexes on tags for fast filtering
- **Hybrid Search**: Vector similarity + tag filtering + keyword matching
