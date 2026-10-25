# Product Requirements Document (PRD) & Design Brief
**Project Codename:** Spectra Guard (Instagram Synthetic Media & Deepfake Detection Clone)  
**Version:** 1.0  
**Target Platform:** Mobile (iOS / Android Native Experience, Dark Theme)  
**Status:** In Prototype / Active Design Iteration

---

## 1. Executive Summary & Objective

The objective of **Spectra Guard** is to demonstrate an authentic, consumer-grade social media platform (modeled after Instagram’s native mobile design patterns) with seamless, non-intrusive integration of **automated deepfake, synthetic media, and AI persona detection**.

Rather than introducing disruptive, alarming cybersecurity banners that confuse ordinary users or degrade platform retention, the app employs **native integrity patterns**—utilizing Instagram’s authentic blue accents (`#0095F6`), discreet inline badges, subtle video overlays, and transparent bottom-sheet forensic diagnostics.

---

## 2. Problem Statement & User Value

- **The Challenge:** Generative AI tools and face-swapping engines make hyper-realistic synthetic media ubiquitous. Traditional detection interfaces are either hidden completely or display heavy, technical "cybersecurity warnings" that cause panic, misattribution, or visual clutter.
- **The Solution:** Embed real-time AI attribution and forensic transparency natively into the standard Instagram browsing experience:
  - Keep standard feeds (Explore, Direct Messages) 100% clean and familiar.
  - Apply minimal, unobtrusive indicators ("Altered with AI" / "Suspected Deepfake" + small shield) where synthetic media is detected.
  - Empower users with clear, educational, on-demand explanations through a native bottom sheet.

---

## 3. Core Screens & Architecture

### 3.1. Home Feed (`Home Feed - Deepfake Detection`)
* **Purpose:** Core consumption flow with feed posts and story carousel.
* **Key Features:**
  - Standard Instagram dark mode header with logo, direct messages, and notification triggers.
  - Multi-colored organic story rings (`#fdf497` to `#d6249f`), avoiding red/alert tinting.
  - Post header with user attribution and timestamp.
  - Discrete inline tag: **"Altered with AI"** badge in Instagram blue (`#0095F6`) with subtle sparkle icon.
  - Tappable tag opens forensic context modal without disrupting feed position.

### 3.2. Reels Feed (`Reels - Deepfake Detection Overlay`)
* **Purpose:** Short-form, full-bleed vertical video feed with active media integrity screening.
* **Key Features:**
  - Full-screen 9:16 vertical video presentation with creator metadata and audio attribution.
  - Minimal inline detection overlay: Small shield icon paired with `Suspected Deepfake` with subtle drop shadow for high contrast across varying video scenes.
  - Zero heavy pill containers or confusing technical metrics (no raw A/V desync or FFT frequency scores).
  - Tap-to-inspect gesture triggering the deepfake forensic sheet.

### 3.3. Deepfake Forensic Analysis Sheet (`Deepfake Forensic Analysis Sheet`)
* **Purpose:** On-demand transparency modal explaining why a post or reel was flagged.
* **Key Features:**
  - Native bottom-sheet presentation with grab handle and dismiss action.
  - Plain-language copy (*"About this content • Meta AI Info"* / *"AI or synthetic media suspected"*).
  - Transparent detection signals:
    - *Visual patterns* (facial boundary warping, lighting inconsistencies).
    - *Audio synthesis* (voice clone vs. authentic audio).
    - *Content Provenance* (C2PA metadata verification).
  - User controls: *"Why am I seeing this?"*, *"How AI content is identified"*, *"Request a human review"*, and *"Report incorrect label"*.

### 3.4. Flagged AI Persona Profile (`Flagged AI Persona Profile`)
* **Purpose:** Account page for an influencer or creator identified as predominantly synthetic / AI-generated.
* **Key Features:**
  - Standard Instagram profile layout: Avatar, post count, followers/following, bio, story highlights.
  - Compact inline indicator: Small `🛡️ Suspected AI` pill placed directly adjacent to the username (no large disruptive banners).
  - Authentic Instagram primary CTA: Bold blue (`#0095F6`) "Follow" button.
  - Media grid tiles with subtle top-right sparkle badges indicating synthetic content.

### 3.5. Explore & Search (`Instagram Explore & Search`)
* **Purpose:** Clean, unflagged discovery grid demonstrating baseline platform behavior.
* **Key Features:**
  - Top search input bar with category filter chips (*For you*, *Architecture*, *Travel*, *Fashion*, *Art*).
  - Asymmetric 3-column masonry grid mixing static photo squares and 2×2 Reels video tiles.

### 3.6. Direct Messages (`Instagram Direct Messages`)
* **Purpose:** Peer-to-peer chat interface maintaining platform authenticity.
* **Key Features:**
  - Top active status/notes tray with bubble statuses.
  - Categorized inbox tabs (*Primary*, *General*, *Requests*).
  - Conversation list with online presence indicators, unread states, and quick-camera triggers.

---

## 4. Visual Design System & UX Principles

| Dimension | Specification |
| :--- | :--- |
| **Color Scheme** | Pure dark mode: `#121212` / `#1c1b1b` surface, `#262626` borders, `#fafafa` primary text, `#8e8e8e` secondary text. |
| **Integrity Accent** | Native Instagram Blue (`#0095F6` / `#1877F2`), replacing alarming alert reds. |
| **Typography** | Roboto Flex / SF Pro Display with strict hierarchical weight distribution (Medium 14px for usernames, Regular 13px for body/meta). |
| **Iconography** | Discreet outline shield (`🛡️`), subtle four-point sparkle (`✨`), and standard Instagram glyphs. |
| **Copy Tone** | Calm, educational, non-accusatory (*"Altered with AI"*, *"Potential digital manipulation"*). |

---

## 5. Technical & Algorithmic Integration Model

1. **Passive Client Ingestion:** Posts ingest media manifests carrying C2PA cryptographic metadata.
2. **Hybrid Inference Engine:**
   - Server-side CNN/Vision Transformer scans for boundary artifacting, blink frequency, and lip-sync coherence.
   - Client lightweight classifier computes real-time confidence scores.
3. **Thresholding & UI Mapping:**
   - `< 50% Confidence`: No badge (normal post).
   - `50% - 79% Confidence`: Subtle informational label (*"Altered with AI"*).
   - `≥ 80% Confidence`: Minimal overlay + detailed forensic sheet enabled.
   - Account-level heuristic: Accounts with >60% flagged posts receive the inline username badge.

---

## 6. Verification & Human Oversight Flow

- **Self-Declaration:** Creators can toggle *"Made with AI"* at upload.
- **Automated Detection Appeal:** Every flagged post and profile contains a 1-tap *"Request a review"* flow routing to human integrity moderators.
- **Feedback Loop:** *"Report incorrect label"* allows users to report false positives, feeding back into model retraining.
