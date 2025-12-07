---
layout: distill
title: Where's the Chicken? Unpacking Spatial Awareness in Vision-Language Models
description: Modern vision-language models (VLMs) have achieved impressive success in recognizing and describing visual content, yet they continue to struggle with understanding spatial relationships. The limitation persists even with massive data and model scaling, suggesting that the root of the problem lies in the architecture and training objective rather than data alone. This post examines the underlying causes and discusses why recent proposed fixes, while promising, remain insufficient to achieve robust spatial reasoning.
date: 2026-04-27
future: true
htmlwidgets: true
hidden: true

# Mermaid diagrams
mermaid:
  enabled: true
  zoomable: true

# Anonymize when submitting
authors:
  - name: Anonymous

# must be the exact same name as your blogpost
bibliography: 2026-04-27-spatial-awareness.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: The "What" vs. the "Where"
  - name: Why Does the VLM Architecture Forget Position?
    subsections:
      - name: The Evolution of Positional Encodings
      - name: The "Semantic Loudness"
  - name: Do VLMs Look at the Right Place?
    subsections:
      - name: Text Dominates, Vision Follows
      - name: Misdirected Gaze
  - name: What are VLMs Designed For?
  - name: Path Towards Spatially-Aware VLMs

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

## Introduction: The "What" vs. the "Where"

To truly understand an image, we have to treat it as more than a collection of pixels. As an image is a 2D projection of a fundamentally 3D world, mentally reconstructing the scene requires at least two components: recognizing "**what**" is in the image and understanding "**where**" those things are located. 

This notion of "where" comes in two forms. One is the *absolute where*: identifying an object's position on the image plane, often by drawing a bounding box around it. The other is the *relational where*: reasoning about how objects are situated relative to one another (e.g., "the chick is behind the cup" or "the car is to the left of the tree"). Both forms are important, but in this post we focus on the latter: how models reason about spatial relationships.

Combing back to the problem of image understanding, we cannot reliably infer the scene behind an image without knowing both what is present and where things are in relation to each other. Let's consider a simple scene with two chicks sitting near a cup. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/2026-04-27-spatial-awareness/1.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    An image with two chicks and a cup on a wooden table. From the camera's viewpoint, a chick with a purple ribbon is in front of a ceramic cup, while a chick with a blue bonnet is behind the cup.
</div>

If someone asks, "*Grab me the chick behind the cup*," the instruction only makes sense if we can correctly identify the "cup" (*what*) and accurately interpret what "behind" means in the appropriate reference frame (*relational where*). For example, if "behind" is defined relative to the camera's viewpoint, it refers to the object that is farther from the camera than the cup. This kind of relational reasoning is fundamental to real-world systems such as autonomous vehicles and robotic arms, where understanding both the objects and their spatial relationships is critical for safe and reliable action.

Modern vision-language models (VLMs), such as Gemini and ChatGPT, have become remarkably good at the *what*. When asked to describe an image or generate a caption, they often produce accurate and detailed responses. As these models grow larger and get trained on increasingly more datasets, their ability to recognize objects and describe their visible content continues to improve. In many evaluations<d-cite key="lin2014microsoft"></d-cite><d-cite key="antol2015vqa"></d-cite>, they already achieve near human-level performance at identifying what is present in an image.

However, when it comes to reasoning about *where* things are relative to each other, these same models often fall short.

This <mark><b>"what" vs. "where" paradox</b></mark> in modern VLMs becomes especially clear in combined reasoning tasks like "*Find the hidden object*."<d-footnote>View the actual conversation <a href="https://gemini.google.com/share/f5622207a4e3">here</a>.</d-footnote>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/2026-04-27-spatial-awareness/2.png" %}
    </div>
</div>
<div class="caption">
    Task asking Gemini 3 Pro to identify the hidden object.
</div>

In this example, the model (Gemini 3 Pro) correctly identifies the hidden object (the golden ring) but mislocates it. It claims that the ring is to the *right* of the red Snoopy doghouse, when in fact it is located *below* it. These mistakes may seem minor, but they expose a deeper and persistent limitation of modern VLMs: strong semantic recognition does not guarantee accurate relational spatial understanding.

We might expect this gap to shrink as models grow larger and are trained on more data. Yet, despite massive datasets and billions of parameters in state-of-the-art systems, this spatial weakness remains. A growing body of research suggests that the issue lies in the core architecture and training objectives of VLMs. In other words, the limitation stems not only from how much data we provide, but also from how these models are built and what they are fundamentally optimized to prioritize.

In this post, we take a closer look at the architectural roots of this spatial blindness in modern VLMs. We examine the building blocks that process visual information, the training objectives that favor certain capabilities over others, and why many recent fixes still fall short of fully solving the problem. Understanding why models struggle with this relational "where" and how we might overcome this limitation is a key step toward building vision systems that truly understand the worlds they see.

## Why Does the VLM Architecture Forget Position?
To understand why VLMs often fail to capture spatial relationships, we need to look at their architectural foundation: the **Transformer**. Since its introduction in *Attention is All You Need*<d-cite key="vaswani2017attention"></d-cite>, the Transformer has become the backbone of modern language and vision-language models due to the power of its core building block, **self-attention**. In brief, self-attention measures how relevant one token (or word) is to another. For example, in the sentence, "*Here is a yellow chick*," the method learns that "*yellow*" is closely tied to "*chick*" because it describes it.

However, pure self-attention has a critical blind spot: it does not care about order. If we only rely on relevance between tokens, the sentences "*The chick is in front of the cup*" and "*The cup is in front of the chick*" look identical, even though they describe entirely different spatial situations. Without any notion of token position, the model collapses the sentence into something like a bag-of-words representation, where only co-occurrence matters, not who is in front of whom.

When we move from text to images, the problem intensifies. A typical VLM first chops an image into a grid of patches (for example, 16x16 or 32x32), converts the patches into tokens, and then feeds the resulting sequence into the Transformer. Without positional information, the model treats the image as a bag-of-patches. It may recognize that there is a chick and a cup, but it has no built-in mechanism to tell whether the chick patch is to the left, above, or behind the cup patch.


### The Evolution of Positional Encodings
To fix this, researchers developed a series of positional encodings that tag tokens or patches with location information. Many of these methods were originally developed to handle word order in sentences, but here we focus on what happens when they are applied to images.

<details class="bg-transparent border-0 shadow-none p-0">
<summary class="positionencoding">Rough graphical representation of positional encodings on 2D images</summary>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/2026-04-27-spatial-awareness/4.png" %}
    </div>
</div>
<div class="caption">
    Illustration of how different positional encodings represent the same 2×2 image grid. This is a very rough schematic representation of APE, RPE, and RoPE. It is not the actual values or operations used in real models.
</div>
</details>

The original Transformer paper introduces **absolute position encoding (APE)**, which assigns a unique fixed vector to every patch. This approach is fine when all input images share the same size, but it fails once the resolution changes. If a model is trained on 224x224 images, it learns positional vectors tied to that exact grid. At test time, when we give a larger image, the model has no natural way to represent the extra patches. A common workaround is to stretch or interpolate the learned embeddings to a new size, but this distorts the spatial relationships. The model essentially overfits to the specific absolute locations, which hurts its ability to generalize to new resolutions or exhibit translation-invariant behavior.

To move beyond fixed grid sizes, later work introduced **relative position encoding (RPE)**<d-cite key="shaw2018rpe"></d-cite>. Instead of defining where a patch is globally, RPE says where it is *relative to other patches* (for example, "The chick patch is two steps away from the cup patch"). As self-attention already operates on pairwise relationships, this feels like a natural fit, and the model can handle variable-length inputs and different layouts more gracefully. But for 2D images, this comes with a trade-off. RPE largely discards absolute coordinates, which are crucial for localization-heavy tasks like object detection. If you care about *exactly* where a pixel or patch is on the image, a purely relative scheme can degrade performance on tasks that demand precise spatial grounding<d-cite key="wu2021rethinking"></d-cite>. In addition, many RPE implementations do not play well with key–value (KV) caching, since relative biases depend on the current sequence length, which forces the model to recompute more attention scores and leads to higher memory and computation costs for long contexts.

Today, the de facto standard for large language models and many VLMs is **rotary positional encoding (RoPE)**<d-cite key="su2024roformer"></d-cite>. RoPE aims to get the best of both APE and RPE: it preserves absolute position information while using rotation operations to naturally represent relative distances. This design tends to generalize better to longer sequences and has proven robust in text generation. But this method is built for 1D token sequences. When we apply it to images, we need to flatten the 2D grid into a 1D line, which breaks spatial locality: patches that are neighbors vertically might end up far apart in the flattened sequence. To address this, models like **Qwen2-VL**<d-cite key="wang2024qwen2"></d-cite> introduce a multimodal variant (**M-RoPE**) that splits the rotary embedding into temporal, height, and width components, applying separate rotations to each. This better preserves the 2D (and temporal) structure of visual data. Still, these encodings are fundamentally 2D: they do not natively capture depth or full 3D spatial relationships without extra geometric signals.

### The "Semantic Loudness"
Let's imagine there exists a positional encoding scheme that perfectly preserves 3D structure. Even with such an ideal encoding, a growing line of work suggests that spatial reasoning would still remain a persistent weakness. The problem is not only *how good* our positional encodings are, but *how the Transformer processes them internally*. Recent work points to a deeper issue inside the Transformer itself: a phenomenon often referred to as **embedding norm suppression**<d-cite key="qi2025beyondsemantics"></d-cite>. Transformers excel at building high-dimensional vectors that encode **what** is in each patch (e.g., "a yellow chick," "a ceramic cup"). These semantic embeddings tend to have large vector magnitudes; in other words, they are *loud* in the embedding space. Positional encodings, by contrast, are comparatively *quiet*, contributing only a small magnitude to the final representation. During self-attention, the "loud" semantic content dominates the computation, effectively drowning out the "quieter" positional signals. The model often behaves as if position barely matters, even though we have carefully encoded it.

This effect becomes evident in a simple permutation test. Take an image, break it into patches as usual, but then randomly shuffle the order of the visual tokens so that the original spatial layout is destroyed. Many VLMs show only a negligible drop in performance on standard benchmarks. For instance, consider the following captioning example:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/2026-04-27-spatial-awareness/3.png" %}
    </div>
</div>
<div class="caption">
    Result of the permutation test with Qwen3-VL. Even after the shuffling the visual tokens, the model produces similar captions that are nearly identical to the original, preserving key features (highlighted in brown, yellow, and blue).
</div>

Even after random shuffling, the generated captions remain nearly identical. The model still identifies the objects (e.g., "chick" and "blue bonnet") and produces a reasonable description of the scene. This suggests that, despite mechanisms like M-RoPE, the model effectively treats the image as a bag-of-semantic-features, with positional encodings contributing little to the final decision.

To counteract this, recent work proposes **embedding norm normalization**<d-cite key="qi2025beyondsemantics"></d-cite>. The idea is simple: rescale the magnitudes of the visual feature embeddings so that the semantic content and positional signals are on a comparable scale. This encourages the model to pay more attention to positional cues, improving spatial reasoning without sacrificing overall semantic understanding.

However, balancing signal magnitudes is only a partial fix. It reveals that positional information is "quiet" relative to semantics, but even once that is corrected, deeper issues remain. Spatial reasoning goes beyond simply knowing coordinates. Encodings like APE, RPE, and RoPE, even when combined with embedding norm normalization, can indicate where tokens are and how far apart they are, but they do not by themselves capture the *structure* of the scene. Many spatial questions depend on richer relations such as containment ("Is the water *in* the cup?"), occlusion ("Is the chick *behind* the cup?"), and relative depth. These concepts require structured spatial reasoning and domain-specific visual priors, not just louder positional coordinates. In other words, making position information "loud enough to hear" is necessary, but not sufficient for VLMs to reason about space in the way humans do.


## Attention Allocation Problem: Do VLMs Look at the Right Place?
Beyond how models encode position, another line of work asks a different question: *where* do models actually focus their attention? These studies suggest that many spatial errors arise not because positional information is missing, but because the model looks at the wrong place. In VLMs, this happens at two levels: **between modalities** (text vs. image) and **within the image itself** (relevant vs. irrelevant regions). 

### Text Dominates, Vision Follows
VLMs typically process two inputs: the image (e.g., a chick and a cup on a table) and the text (e.g., "Describe the image" or "Where is the chick relative to the cup?"). Architecturally, the visual backbone converts the image patches into patch embeddings and the language backbone (usually a strong LLM) encodes the text. These two pieces are then fused so the model can reason over both modalities jointly. 

However, prior work<d-cite key="zhou2024analyzing"></d-cite><d-cite key="sun2024aligning"></d-cite> repeatedly finds that these inputs do not contribute equally. As LLM backbones are typically much more heavily trained and semantically richer than the visual encoder, the combined model tends to develop a strong textual bias. This mirrors the *semantic loudness* effect we saw earlier: rich language semantics drown out the weaker visual evidence. One study<d-cite key="chen2025why"></d-cite> quantifies this imbalance: roughly 90% of the attention flows to textual tokens, with only about 10% going to the visual tokens. The model leans heavily on the prompt and its internal language statistics, and only lightly consults the image.

This imbalance also contributes to the failures we see in spatial reasoning: the model hallucinates what *should* be there instead of faithfully reporting what *is* there. If its textual knowledge says that "clouds are usually above the grass," the model may confidently assert that relation even if, in the actual image, the scene is upside down and the clouds appear below. The model often sees what it expects from language, not what the pixels actually show. In this view, many spatial errors are less about raw model capability and more about how attention is distributed between modalities: VLMs tend to trust text more than vision.

A natural reaction is to simply increase attention to visual tokens. But that alone is not enough. If the model looks more at the image but still focuses on the wrong regions, spatial reasoning will continue to fail.

### Misdirected Gaze
Even if we manage to encourage a VLM to rely more on visual input, a second failure mode appears: the model may simply look in the wrong place.

Consider asking, "Is the chick in front of or behind the cup?" Ideally, the model should concentrate its attention on the patches containing the chick and the cup. In practice, empirical analyses<d-cite key="chen2025why"></d-cite> shows that VLMs often scatter their attention across irrelevant regions, such as the table surface or the background wall, while paying relatively little attention to the objects mentioned in the question. In these cases, the model is technically "looking at the image," but not at the evidence needed to answer the question.

To address this, a recent work proposes **AdaptVis**<d-cite key="chen2025why"></d-cite>, a training-free method that redirects attention at inference time. The key idea is to use the model's own prediction confidence to adapt its visual attention pattern dynamically. When the model is confident, AdaptVis sharpens the attention distribution, narrowing the focus to regions the model already considers relevant and suppressing background noise. When the model is uncertain, AdaptVis broadens the attention window, encouraging exploration of wider area and potentially discovering objects and relationships it initially overlooked. This confidence-guided adjustment shows strong empirical gains and suggests that many spatial failures arise not from the lack of capability to reason about space, but from *failing to focus on the right evidence at the right time*.

Methods like AdaptVis highlight that attention allocation is an important piece of spatial reasoning. They show that VLMs often underuse the visual information they already encode, and that steering attention can meaningfully improve behavior.

However, AdaptVis is not a complete solution. As the method operates purely at inference time, it cannot change the model's internal representation or how the spatial relations are learned in the first place. If the model never learned concepts like "in front of" vs. "behind," simply pointing its attention more precisely will not fully close those gaps. Attention steering helps the model make better use of what it already knows; but by itself, it cannot guarantee the richer, human-like spatial reasoning that many applications require.

## From Global Glance to Localized Training: What are VLMs Designed For?
The methods introduced so far (embedding norm normalization and AdaptVis) can be viewed as post-hoc fixes. They operate on top of an already trained architecture, leaving both the original model design and its pretraining objectives unchanged. While these approaches do improve performance, a growing body of work argues that true spatial awareness cannot simply be patched in with auxiliary mechanisms. Instead, the root of the problem lies in how these models are trained from the beginning.

The dominant paradigm for large-scale VLM training is **Contrastive Language-Image Pretraining (CLIP)**<d-cite key="radford2021learning"></d-cite>. In this framework, a model is trained on large collections of image–caption pairs (e.g., COCO) to match each image with its corresponding text description. Fudamentally, this is *global*: the model aligns the *entire image* with the *entire caption*. As a result, it is strongly encourages to encode the overall meaning and semantic themes of a scene. This is precisely what gives modern VLMs their impressive ability to describe images fluently and accurately. But it also builds in a limitation: the training objective rewards learning **what** is in the image, while largely ignoring **where** those entities are located. Spatial structure is, at most, learned only indirectly.

To address this limitation at its source, recent work proposes **Contrastive Localized Language-Image Pretraining (CLOC)**<d-cite key="chen2025contrastive"></d-cite>, a framework that explicitly incorporates spatial grounding into the pretraining objective. Under CLOC, the model is no longer trained solely to align the whole image with the whole caption. Instead, it is encouraged to learn both object identity and object location, producing representations that are grounded at a finer, region-aware level. In principle, this allows the visual backbone to encode spatial information directly into its features, rather than relying on later-stage corrections.

But this kind of localized pretraining comes with a substantial cost. Adopting it typically requires retraining the visual backbone from scratch, which is computationally expensive and time-consuming (especially for state-of-the-art VLMs that already represent years of training effort). This motivates a parallel line of research focusing on **modular interventions**: methods that aim to inject spatial awareness into existing, frozen models without full retraining. One example is **Positional Insert (PIN)**<d-cite key="dorkenwald2024pin"></d-cite>, an input-agnostic, learnable spatial prompt with a small number of parameters that can be inserted into a frozen VLM to unlock object localization capabilities. Another is **Twist & Scout**<d-cite key="bhowmik2025twist"></d-cite>, which modifies the language model's decoder with a dual mixture-of-experts design: one expert remains frozen to preserve the original CLIP-style semantics, while a second expert is specialized for location grounding. At inference time, the model dynamically switches between these experts depending on the task.

These methods represent an important first step toward models that are natively capable of object localization, but they also highlight several remaining fundamental challenges. For unified models like CLOC, beyond the cost of retraining, a central question is how much spatial information can be injected without affecting global semantic understanding. It remains unclear how to balance learning *what* and learning *where* within a single system: how much representational capacity and attention should be devoted to semantics vs. spatial structure, and how that balance should adapt across different levels of visual complexity. In modular designs, maintaining strong semantic experts while adding dedicated spatial experts increases computational and memory overhead. Because these approaches rely on frozen backbones, they also limit how deeply spatial information can be integrated into the visual feature hierarchy; spatial cues are often introduced only at later stages (for example, at the decoder), constraining the model's ability to form truly grounded representations.

Taken together, these challenges suggest that localized and modular interventions are promising, but they are unlikely to solve spatial reasoning on their own. Overcoming the **"what" vs. "where"** divide will likely require coordinated advances in model architecture, training objectives, supervision signals, and other parts of the learning pipeline, rather than isolated fixes applied at a single point.

## Conclusion: Path Towards Spatially-Aware VLMs
Stepping back, a consistent picture emerges: today's VLMs are very good at answering "*What's in the image?*" but far less reliable at answering "*Where is it relative to that?*" This weakness does not disappear with more data or larger models. Instead, it makes us trace back to how these systems are built and trained: Transformers are naturally position-agnostic, positional encodings get drowned out by rich semantic features, and training pipelines reward capturing the overall gist more than spatial precision.

Throughout this post, we have seen several promising attempts to narrow this gap: more expressive positional encodings (from APE to RPE to RoPE and M-RoPE) to better preserve the 2D layout of images, embedding norm normalization to help positional signals compete with loud semantic embeddings, attention-steering methods to guidemodels to look at the righr regions, and training strategies to push models to encode not only what objects are present, but also where they are in the scene.

Still, none of these approaches fully solves spatial reasoning by its own. Spatial awareness does not emerge from plugging in a single clever module; it is a property that arises from the interaction among the architecture, attention, and training objectives. Moving toward spatially-aware VLMs will likely require these strands to work together and a deeper understanding of each individual component itself: positional schemes that respect 2D (and ultimately 3D) structure, mechanisms that reliably guide attention to the right evidence, and objectives that genuinely demand spatial precision. If we want agents that can act safely and effectively in the physical world, we must treat **where** not as a byproduct of **what**, but as a first-class goal of visual intelligence.