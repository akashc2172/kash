# PostHog Design Brief

## **1. Fonts**
*   **Headings:** `IBM Plex Sans` (Weights: **700**, **800**)
*   **Body:** `IBM Plex Sans` (Weights: **400**, **500**, **600**)
*   **Monospace:** `ui-monospace`, `SFMono-Regular`, `Menlo`, `Monaco`, `Consolas` (Standard system mono stack)

## **2. Colors**
| Usage | Color | Hex |
| :--- | :--- | :--- |
| **Background (Main)** | Warm Beige | `#EEEFE9` |
| **Surface (Cards/Windows)** | Off-White | `#FDFDF8` |
| **Surface (Nav/Sidebar)** | Black | `#000000` |
| **Accent (Primary CTA)** | Deep Amber | `#CD8407` |
| **Accent (Secondary)** | PostHog Orange | `#EB9D2A` |
| **Text (Primary)** | Black | `#000000` |
| **Text (Secondary)** | Muted Gray | `#374151` |

## **3. Border Radius & Shadows**
*   **Radius:** `6px` for buttons/tags. `0px` (or very small 2-4px) for main containers/windows to maintain the "retro software" look.
*   **Shadow Style:** **Hard & Tactile**.
    *   Avoids soft/blurred drop shadows.
    *   **Buttons:** Use `border-bottom-width: 3px` to create a 3D "pressable" effect.
    *   **Cards:** Defined by solid `1px` or `1.5px` borders (often `#D0D1C9` or Black).

## **4. Motion Patterns**
*   **Hover:** Snappy and fast (`~0.15s`). Buttons often shift background color or border color instantly.
*   **Page Load:** Custom "Hydrating" progress bar loader.
*   **Interactions:** "Desktop icon" hover effects (background highlight). Tab switching is instant/crisp, avoiding long fades.

## **5. Layout Patterns**
*   **Metaphor:** "Retro Desktop OS". The UI often mimics a computer desktop with windows, icons, and a menu bar.
*   **Navigation:** Sticky top bar (Black with White text), reminiscent of a system menu bar.
*   **Hero:** Centered "Main Application Window" surrounded by floating "Desktop Icons" (features).
*   **Cards:** Styled as application windows or terminal blocks with distinct headers and solid borders.
*   **Grids:** Dense, information-heavy grids often separated by clear lines or "window" edges.
