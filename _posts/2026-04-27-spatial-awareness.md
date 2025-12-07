---
layout: distill
title: Where's the Chicken? Unpacking Spatial Awareness in Vision-Language Models
description: Modern vision-language models (VLMs) have achieved impressive success in recognizing and describing visual content, yet they continue to struggle with understanding spatial relationships. The limitation persists despite massive data and model scaling, suggesting that the root of the problem lies in the architecture and training objective rather than data alone. This post examines the underlying causes and discusses why recent proposed fixes, while promising, remain insufficient to achieve robust spatial reasoning.
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
      - name: The "Semantic Loudness" Problem
  - name: Do VLMs Look at the Right Place?
    subsections:
      - name: Text Dominates, Vision Follows
      - name: Misdirected Gaze Problem
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

True image understanding requires us to look beyond a collection of pixels. As an image is a 2D projection of a fundamentally 3D world, mentally reconstructing the scene requires two essential components: recognizing "**what**" is in the image and understanding "**where**" they are located. 

This concept of "where" takes two forms. One is the absolute where: identifying an object’s position on the image plane, typically by drawing a bounding box around the object. The other is the ***relational where***: reasoning about how objects are situated relative to one another (e.g., "the chick is behind the cup" or "the car is to the left of the tree"). This post focuses on the latter: how models reason about spatial relationships. Without knowing both what is present and where things are in relation to each other, we cannot reliably infer the scene behind the image. To illustrate, let’s consider a simple example: two chick sitting near a cup. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/2026-04-27-spatial-awareness/1.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    An image with two chicks and a cup on a wooden table. From the camera’s viewpoint, a chick with a purple ribbon is in front of a ceramic cup, while a chick with a blue bonnet is behind the cup.
</div>

If someone asks, "*Grab me the chick behind the cup*," the instruction makes sense only if we can correctly identify the "cup" (*what*) and accurately interpret what "behind" means in the appropriate reference frame (*relational where*). For example, if "behind" is defined relative to the camera’s viewpoint, it refers to the object that is farther from the camera than the cup. This kind of relational reasoning is fundamental to real-world systems such as autonomous vehicles and robotic arms, in which understanding both the objects and their spatial relationships is critical for safe and reliable action.

Modern vision-language models (VLMs), such as Gemini and ChatGPT, have become remarkably good at the "***what***." When asked to describe an image or generate a caption, they often produce accurate and detailed responses. As these models grow larger and get trained on increasingly vast datasets, their ability to recognize objects and describe their visible content continues to improve. In many evaluations<d-cite key="lin2014microsoft"></d-cite><d-cite key="antol2015vqa"></d-cite>, they demonstrate nearly human-level performance in identifying what is present in an image.

Yet, when it comes to reasoning "**where**" things are relative to each other, these same models often fall short.

This "what" vs. "where" paradox in modern VLMs becomes clear in combined reasoning tasks such as "*Find the hidden object*," as shown in the following example.<d-footnote>View the actual conversation <a href="https://gemini.google.com/share/f5622207a4e3">here</a></d-footnote>.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/2026-04-27-spatial-awareness/2.png" %}
    </div>
</div>
<div class="caption">
    Task asking Gemini 3 Pro to identify the hidden object.
</div>

In this example, the model (Gemini 3 Pro) correctly identifies the hidden object (the golden ring) but incorrectly describes its location. Specifically, it states that the ring is to the *right* of the red house (sticker), when in fact it is located *below* it. These errors may appear subtle, but they reveal a deep and persistent limitation of modern VLMs: strong semantic recognition does not guarantee accurate relational spatial understanding.

We might expect this gap to shrink as models grow larger and receive more training data. However, despite massive datasets and billions of parameters, as seen in the most advanced systems, this spatial weakness persists. A growing body of research suggests that the issue runs in the core architecture and the objectives VLMs are trained on. In other words, the limitation may not only stem from how much data we provide, but also from how these models are built and what they are fundamentally trained to prioritize.

In this post, we take a closer look at the architectural roots of this spatial blindness in modern VLMs. We examine the building blocks that processes visual information in these models, the training objectives that prioritize specific skills over others, and why recently proposed fixes still fall short of fully resolving the problem. Understanding why models struggle with this relational "where," and how we can overcome this limitation is a key step towards building vision systems that truly understand the world they see.


## Why Does the VLM Architecture Forget Position?
To understand why VLMs frequently fail to capture spatial relationships, we must look at their architectural foundation: the **Transformer**. Since its introduction in *Attention is All You Need*<d-cite key="vaswani2017attention"></d-cite>, Transformer has become the backbone of modern language and vision-language models due to the power of its core building block, **self-attention**. In brief, self-attention calculates the relevance of one token (or a word) to another. For example, in the sentence, "*Here is a yellow chick*," the method identifies that "*yellow*" is a descriptor highly relevant to "*chick*."

However, pure self-attention has a critical blind spot: it does not care about order. If we only rely on the relevance between tokens, the sentences "The chick is in front of the cup" and "The cup is in front of the chick" look identical, even though they describe entirely different spatial situations. Without any notion of token position, the model reduces the sentence to a bag-of-words, where only co-occurrence matters, not who is in front of whom.

When we move from text to images, the problem intensifies. A typical VLM first chops an image into a grid of patches (for example, 16x16 or 32x32), converts the patches into tokens, and then feeds the resulting sequence of tokens into the Transformer. Without positional information, the model views this image as a bag-of-patches. It perceives the existence of a chick and a cup, but it has no structural mechanism for knowing whether the patch containing the chick is located to the left, above, or behind the patch containing the cup. 


### The Evolution of Positional Encodings
To fix this, researchers developed various positional encodings to tag these patches with location data.

The original Transformer paper introduces absolute position encoding (APE), which assigns a unique fixed vector to every token or patch. Yet, APE is inherently brittle to varying resolutions. If a model is trained on 224 x 224-sized images, it learns specific vectors for that exact grid size. At inference time, if we input a larger image, the model cannot naturally handle the new coordinates. A naive fix is to stretch the learned positional embedding to match the new size, but this distorts the spatial relationships. The model essentially overfits to specific absolute locations, hindering its ability to generalize to unseen resolutions or recognize translation-invariant features.

To move beyond fixed grid sizes, Shaw et al. (2018) introduced relative position encoding (RPE)<d-cite key="shaw2018rpe"></d-cite>. Instead of defining where a token is globally, RPE defines it by its pairwise distance to other tokens (e.g., "Patch A is two steps away from Patch B"). Since self-attention operates on token-to-token relationships, RPE aligns naturally with the self-attention mechanism and handles varying input sizes well. However, for 2D images, this is suboptimal because RPE discards absolute coordinates. For localization-heavy tasks, such as object detection, knowing exactly where a pixel is on the image is crucial. By focusing purely on relative distance, RPE degrades performance on tasks requiring precise spatial grounding<d-cite key="wu2021rethinking"></d-cite>. 

The current de facto standard for large language models (LLMs) and VLMs is rotary position encoding (RoPE)<d-cite key="su2024roformer"></d-cite>. RoPE mathematically combines the advantages of both APE and RPE: it maintains absolute position information but uses rotation matrices to model relative distances naturally. This allows for better generalization to longer sequence lengths and has proven to be robust in text generation. Standard RoPE is designed for 1D text sequences. To apply it to an image, the 2D grid is typically flattened into a 1D line. This process destroys spatial fidelity, as vertically adjacent pixels might end up far apart in the flattened sequence. To address this, models like Qwen2-VL<d-cite key="wang2024qwen2"></d-cite> introduce a multimodal variant (M-RoPE), which decomposes the rotary embedding into distinct temporal, height, and width components, applying separate rotations to each dimension to preserve the structural integrity of the visual data. Although M-RoPE improves the representation of 2D grid position, it remains fundamentally limited in tasks requiring 3D spatial reasoning, as it cannot natively encode depth cues or volumetric spatial relationships without auxiliary geometric data.

### The "Semantic Loudness" Problem
Even if we hypothesize the existence of an advanced method that perfectly preserves the 3D structure, a growing line of research on positional encoding suggests that spatial reasoning will remain a persistent weakness, regardless of the sophistication of the encoding scheme. Recent work suggests that the issue is not only how good our positional encodings are, but how the Transformer processes them internally. Qi et al. (2025) identify a phenomenon they call **embedding norm suppression**<d-cite key="qi2025beyondsemantics"></d-cite>. Transformers excel at building high-dimensional vectors that encode "what" is in each patch (e.g., "a yellow chick", "a ceramic cup"). These semantic embeddings tend to have large vector magnitudes; simply, they are loud in the embedding space. In contrast, positional encodings are often subtle, contributing only a small magnitude to the final representation. During self-attention calculation, the "loud" semantic content dominates the computation, drowning out the softer positional signals. Effectively, the model often behaves as if the position barely matters, even though we have carefully encoded it.

This effect becomes evident in a simple permutation test<d-cite key="qi2025beyondsemantics"></d-cite>. Take an image, break it into patches as usual, but then randomly shuffle the order of the visual tokens to destroy the original spatial layout. Many VLMs show only a negligible drop in performance on standard benchmarks. For instance, consider the following captioning example:

Even after random shuffling, the generated captions remain nearly identical. The model still identifies the objects (e.g., "chick" and "blue bonnet") and produces a reasonable description of the scene. This suggests that, despite mechanisms like RoPE, the model effectively treats the image as a bag-of-semantic-features, with positional encodings contributing little to the final decision.

To counteract this, Qi et al. propose **embedding norm normalization**<d-cite key="qi2025beyondsemantics"></d-cite>. The idea is simple: rescale the magnitudes of the visual feature embeddings so that the semantic content and positional signals are on a comparable scale. This simple intervention encourages the model to attend more to positional cues, improving spatial reasoning without degrading overall semantic understanding.

While balancing the signal magnitude is a crucial step, it is not a complete solution. It reveals that positional information is "quiet" relative to semantics, but even once that is fixed, deeper issues remain. Spatial reasoning goes beyond simply knowing the coordinates. Encodings like APE, RPE, and RoPE, along with embedding norm normalization, can indicate where tokens are and how far apart they are, but they do not by themselves capture the structure of the scene. Many spatial questions rely on complex relations such as containment (e.g., "Is the water *in* the cup?"), occlusion (e.g., "Is the chick *behind* the cup?"), and relative depth. These concepts require spatial reasoning and domain-specific visual priors, not just louder positional coordinates. In other words, making the position information "loud enough to hear" is necessary, but not sufficient for VLMs to reason about space in the way humans do.


## Attention Allocation Problem: Do VLMs Look at the Right Place?
Beyond how models encode position, a parallel line of research studies where models dedicate their focus. These works suggest that spatial errors arise not from insufficient positional information but from misallocated attention. In VLMs, this happens in two layers: between modalities (text vs. image) and within the image (relevant vs. irrelevant regions). 

### Text Dominates, Vision Follows
VLMs often process two sources of input: the image (e.g., a chick and a cup on a table), and the text (e.g., “Describe the image” or “Where is the chick relative to the cup?”). Architecturally, this is implemented with a visual backbone that encodes the image into patch tokens and a language backbone (usually a strong LLM) that encodes the text. These two pieces of information are then fused so that the model can reason over both modalities jointly.

However, prior works<d-cite key="zhou2024analyzing"></d-cite><d-cite key="sun2024aligning"></d-cite> repeatedly finds that these two inputs do not contribute equally. As LLM backbones are typically much more heavily trained and semantically richer than the visual encoder, the combined model often exhibits a textual bias. Chen et al. (2025)<d-cite key="chen2025why"></d-cite>, for example, quantify this imbalance: roughly 90% of the attention flows to textual tokens, with only about 10% going to the visual tokens. The model leans heavily on the text prompt and its internal language statistics, and only lightly consults the image.

This leads to a familiar failure mode in spatial reasoning. The model hallucinates what “should” be instead of varying what “is” there. If the textual prior suggests that “clouds are usually above the grass,” the model may confidently assert the relation even if, in the actual image, the scene is upside down and the clouds appear below. The model often sees what it expects from language, not what the pixels actually show. Spatial errors, in this view, are often less about raw capability and more about imbalances between modalities. VLMs tend to trust text more than vision. 

A naive reaction is to try to increase attention to the visual tokens. But this alone is not sufficient. If the model focuses more on the image but does not look at the right parts, spatial reasoning still fails.

### Misdirected Gaze Problem
Even if we successfully encourage a VLM to rely more on visual input, a second failure mode appears: the model may simply look in the wrong place.

Consider asking the question, “Is the chick in front of or behind the cup?” Ideally, the model should focus on the patches containing the chick and the cup. However, empirical analysis<d-cite key="chen2025why"></d-cite> shows that VLMs frequently scatter their attention across irrelevant regions, such as the table surface, the background wall, or other high-contrast noises, while paying relatively little attention to the actual objects mentioned in the question. In such cases, the model is technically “looking at the image,” but not at the evidence needed to answer the question.

To address this, Chen et al. proposed AdaptVis<d-cite key="chen2025why"></d-cite>, a training-free method that redirects attention at inference time. The key idea is to use the model’s own prediction confidence to adapt its visual attention pattern dynamically. When the model is confident in its prediction, AdaptVis will sharpen the attention distribution, narrowing the focus to the regions the model already considers relevant and suppressing background noise. When the model is uncertain, AdaptVis broadens the attention window, encouraging it to explore a wider area of the image and potentially discover objects and relationships it initially overlooked. This confidence-guided adjustment has shown strong empirical gains. It suggests that many spatial failures do not stem from the lack of capability to reason about space; instead, the model often has the necessary information but fails to direct its focus to where it matters most.

Overall, methods like AdaptVis highlight that attention allocation is a valuable piece in spatial reasoning. They show that VLMs often underuse the visual information they already encode, and that steering attention can meaningfully improve behavior.

However, AdaptVis is not a complete solution. As the method operates purely at inference time, it cannot change the model’s internal representation or how the spatial relations are encoded. If the model has never formed a robust notion of concepts like “in front of” vs. “behind,” simply pointing its attention more precisely will not fully resolve those gaps. Attention steering can only help the model use what it knows more effectively. By itself, it cannot guarantee the richer, human-like spatial reasoning that many applications demand.

## From Global Glance to Localized Training: What are VLMs Designed For?
The methods introduced so far (norm normalization and AdaptVis) can be viewed as post-hoc fixes. That is, they operate on top of an already trained architecture, leaving both the original model design and its pretraining objectives unchanged. While these approaches did show some improvements, an increasing number of studies argue that true spatial awareness cannot simply be derived by adding an auxiliary mechanism. Instead, the root of the problem lies in how these models are trained from the beginning.

The dominant paradigm for training large-scale VLMs is Contrative Language-Image Pretraining (CLIP)<d-cite key="radford2021learning"></d-cite>. In this framework, a model is trained on extensive collections of image-caption pairs (e.g., COCO) to match each image to its corresponding text description. This training signal is fundamentally global: the model aligns the entire image with the entire caption, and as a result, it is strongly encouraged to encode the overall meaning and semantic themes of a scene. This design choice gives modern VLMs their impressive ability to describe images fluently and accurately. However, it also comes with a built-in limitation: the training objective rewards learning what is in the image, but does not penalize for ignoring where those entities are located. Spatial structure is, at most, learned only indirectly. 

To address this limitation at its source, Chen et al. (2024) propose Contrastive Localized Language-Image Pretraining (CLOC)<d-cite key="chen2025contrastive"></d-cite>, which is a framework that explicitly incorporates spatial grounding into the pretraining objective. Under CLOC, the model is no longer trained solely to align the whole image with the whole caption. Instead, it is encouraged to learn both the object identity and spatial localization to produce representations that are grounded at a finer, region-aware level. In principle, this enables the visual backbone to encode spatial information directly within its representation, rather than relying on later-stage corrections.

However, this approach comes with a substantial cost. Adopting localized pretraining typically requires retraining the visual backbone from scratch; this process is computationally expensive and time-consuming, given that many state-of-the-art VLMs result from years of training. This motivated a parallel line of research focusing on modular interventions: methods that aim to inject spatial awareness into existing, frozen models without requiring full retraining. Dorkenwald et al. (2024) introduce Positional Insert (PIN)<d-cite key="dorkenwald2024pin"></d-cite>, an input-agnostic, learnable spatial prompt with a small number of parameters that can be inserted into a frozen VLM to unlock object localization capabilities. Bhowmik et al. (2025) propose Twist & Scout<d-cite key="bhowmik2025twist"></d-cite>, which modifies the language model's decoder through a dual mixture-of-experts (MoE) design: one expert remains frozen to preserve the original CLIP-style semantics, while a second expert is specialized for location grounding. At inference time, the model dynamically switches between these experts depending on the task.

While these methods represent an essential first step towards models that are natively capable of object localization, several fundamental challenges remain. For unified models (such as CLOC), in addition to the cost of retraining, a central issue is how much spatial information can be injected without degrading global semantic understanding. It is still unclear how to balance learning what and learning where within a single system: how much representational capacity and attention should be allocated to semantics versus spatial structure, and how this balance should adapt across different levels of visual complexity. In modular designs, preserving strong semantic experts while adding a dedicated spatial expert substantially increases computational overhead and memory footprint. Moreover, as this approach relies on frozen backbones, it limits how deeply spatial information can be integrated into the visual feature hierarchy. As a result, spatial cues are often introduced only at later stages (e.g., at the decoder level), constraining the model’s ability to form truly spatially grounded representations. Taken together, these challenges suggest that while localized and modular interventions offer promising directions, they are unlikely to fully resolve spatial reasoning on their own. Overcoming the “what” vs. “where” divide will likely require coordinated advances in model architecture, training objectives, supervision signals, and various other aspects of the learning pipeline, rather than isolated fixes applied at a single point. 

## Conclusion: Path Towards Spatially-Aware VLMs
Stepping back, a consistent picture emerges: today’s VLMs excel at answering “*What’s in the image?*” but are less reliable at “*Where is it?*” This weakness does not disappear with more data or larger models. Instead, it traces back to how these systems are built and trained: Transformers that are naturally position-agnostic, positional encodings that are often drowned by rich semantic features, and training pipelines that prioritize general gist over spatial precision. 

Throughout the post, we have seen several promising attempts to patch this gap. More expressive positional encodings (APE to RPE to RoPE and M-RoPE) better preserve the 2D structure of images, and norm normalization helps positional information to compete with loud semantic embeddings. Attention-steering methods such as AdaptVis show that models often can reason spatially when they are guided to look at the right image regions. Localized and modular training strategies push the model to encode not only object identity but also their location in the scene.

However, none of these by themselves fully solves spatial reasoning. Spatial awareness does not arise from plugging in a single module; it is a property that emerges from the interaction among architecture, attention, and the training signal. Moving towards a spatially-aware VLM will likely require these strands to work together: positional schemes that respect 2D (and 3D) structure, mechanisms that can reliably guide attention to the right evidence, and objectives that demand spatial precision. To build agents that can truly act in the physical world, we must treat “**where**” not as a byproduct of “**what**”, but as a fundamental component of vision.