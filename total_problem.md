

## Problem

Current semantic image retrieval systems such as CLIP-based retrievers primarily represent an image and a text query as global embeddings and retrieve images using embedding similarity.

This works well for broad queries such as:

> “a dog on a beach”

but performs poorly when the query contains **multiple fine-grained constraints, attributes, object relationships, exclusions, or spatial information**.

For example:

> “Find an image of a golden retriever running on a beach, next to a red surfboard, with no other people visible.”

A conventional embedding-based retriever may return images containing a golden retriever, beaches, or surfboards individually, while failing to satisfy the complete semantic composition of the query.

The fundamental problem is that the meaning of an image is not simply the sum of its visual concepts. It also depends on **relationships between objects, attributes, spatial arrangements, actions, and the absence or presence of specific concepts**.

## Objective

Build a semantic image retrieval system that can understand and retrieve images according to **compositional semantic meaning**, rather than relying only on global image-text embedding similarity.

The system should represent a query in terms of:

- Objects
    
- Attributes
    
- Actions
    
- Object-object relationships
    
- Spatial relationships
    
- Scene/context
    
- Positive constraints
    
- Negative constraints
    

For example:

> “A person wearing a blue shirt standing beside a red car, but no other people.”

should be interpreted approximately as:

```text
Objects:
    person
    car

Attributes:
    shirt = blue
    car = red

Relationship:
    person BESIDE car

Negative constraint:
    number_of_other_people = 0
```

The retrieval system should then rank images according to how well they satisfy the **complete semantic structure** of the query.

## Interactive Retrieval

The system should additionally support iterative user feedback.

Example:

```text
User:
    Find pictures of dogs on beaches.

System:
    Returns top-K images.

User:
    More like image #4, but the dog should be running.

System:
    Updates the semantic query and retrieves new results.

User:
    Golden retriever specifically.

System:
    Further refines the retrieval.

User:
    Remove images containing people.

System:
    Applies the negative constraint and retrieves again.
```

The system therefore needs to maintain a representation of the user's evolving search intent rather than treating every query independently.

## Core Research Question

Can a multimodal retrieval system improve fine-grained image retrieval by explicitly modeling:

```text
Objects
    +
Attributes
    +
Actions
    +
Relationships
    +
Spatial information
    +
Negative constraints
    +
User feedback
```

instead of relying exclusively on a single global image-text embedding?

## Proposed System

A possible architecture is:

```text
                    Text Query
                        │
                        ▼
                 Query Understanding
                        │
                        ▼
              Semantic Query Structure
             ┌──────────┼──────────┐
             │          │          │
          Objects   Attributes  Relations
             │          │          │
             └──────────┼──────────┘
                        │
                        ▼
                 Candidate Retrieval
                        │
                        ▼
              Fine-Grained Reranker
                        │
                        ▼
                  Top-K Images
                        │
                        ▼
                  User Feedback
                        │
                        └──────────► Query Update
```

The first stage should retrieve a reasonably large candidate set efficiently.

The second stage should perform expensive fine-grained semantic matching between the query and candidate images.

## Baselines

The system should be compared against established retrieval approaches such as:

1. CLIP
    
2. SigLIP
    
3. Dual-encoder image-text retrieval
    
4. Cross-encoder / multimodal reranking
    
5. Vision-Language Model based retrieval
    

The goal is not merely to obtain better Recall@K on ordinary image retrieval, but to specifically evaluate whether the proposed system handles **compositional and constraint-heavy queries better**.

## Evaluation

Evaluate using standard retrieval metrics:

- Recall@1
    
- Recall@5
    
- Recall@10
    
- mAP
    
- nDCG
    

Additionally introduce compositional evaluation metrics measuring whether retrieved images satisfy:

- Object correctness
    
- Attribute correctness
    
- Relationship correctness
    
- Spatial correctness
    
- Negative constraints
    
- Overall query satisfaction
    

For example:

```text
Query:
"Person wearing a blue shirt beside a red car."

Retrieved image:
    Person          ✓
    Blue shirt      ✓
    Red car         ✓
    Beside relation ✓

Semantic satisfaction:
    4 / 4
```

versus:

```text
Retrieved image:
    Person          ✓
    Blue shirt      ✓
    Red car         ✓
    Beside relation ✗

Semantic satisfaction:
    3 / 4
```

## Key Challenge

The central challenge is that **global embedding similarity does not necessarily imply semantic correctness**.

Two images can have highly similar embeddings while differing in an important detail:

```text
Query:
"Dog chasing a ball."

Image A:
Dog chasing ball.

Image B:
Dog sitting beside ball.
```

Both contain:

```text
dog + ball
```

but only Image A satisfies the relationship:

```text
dog CHASING ball
```

Therefore, the system needs to reason about **what is happening in the image**, not merely which concepts appear in it.

## Desired Outcome

The final system should be capable of answering queries such as:

> “Find a photo of a person riding a bicycle on a mountain road, wearing a red jacket, with another cyclist behind them, but no cars.”

and retrieve images that satisfy the complete semantic description.

The research contribution would be a retrieval framework that moves from:

```text
Global similarity
        ↓
Concept-level similarity
        ↓
Compositional semantic matching
        ↓
Interactive intent-aware retrieval
```

The strongest version of the problem is therefore:

> **Design an interactive multimodal image retrieval system capable of understanding compositional semantics, object relationships, attributes, spatial constraints, negative constraints, and iterative user feedback, and demonstrate that explicit semantic reasoning improves retrieval quality over conventional embedding-based retrieval.**