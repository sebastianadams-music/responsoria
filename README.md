# responsoria 2026

Vibecoded with **agy** (Google Antigravity).

Live at [responsoria.sebastianadams.net](https://responsoria.sebastianadams.net).

## Initial Prompt

> Hi, I want you to make a generative music webapp based on Gesualdo's responsories - feria quinta
>
> I want you to use the Collegium Vocale Gent recording with Herreweghe from 2013 as source material (please download from YouTube using yt-dlp)
>
> The web app should create a soundscape using concatenative synthesis of samples from the source material. The segmented material should be arranged on the X-axis based on its MFCC and the spectral centroid on the Y-axis. 
>
> Segments should be created at syllable-boundaries where possible. 
>
> The amplitude/rhythmic profile of the source material should be used as the timeline of the collage and simply runs in real time from start to finish.
>
> The user should control the segments used based on mouse position (or on the mobile, the last finger position), with a knn of 5 to prevent boredom. 
>
> A new segment should be triggered on each new "note" from the amplitude envelope.
>
> At ends of phrases (hopefully you can use the amplitude envelope to detect phrase endings too), the mode should toggle between "user mode" and "answer mode". In user mode, the user's mouse position (or screen press position) selects segments, and the background is black with a white point at the cursor. In computer mode, the computer selects segments that are the "opposite" (farthest away in MFCC/spectral centroid) of the user's selections (please note: the timeline/amplitude envelope continues from start to finish without reference to whether it is user mode or computer mode, so the number or relative time-position of attacks will probably not line up)
>
> There should be no UI elements, and no explanatory notes. The only text should be "responsoria 2026" as a title.

## Versions

The latest build is always served at the root ([responsoria.sebastianadams.net](https://responsoria.sebastianadams.net)) and archived under `/vN`:

- **[v2](https://responsoria.sebastianadams.net/v2/)** *(current)*: Replaces farthest-point Answer Mode selection with a full 2D space inversion (`invertPoint`) that swaps interior ↔ perimeter regions while flipping both the MFCC (X) and Spectral Centroid (Y) axes, and adds versioned build archiving.
- **[v1](https://responsoria.sebastianadams.net/v1/)**: Initial public release with 150 ms+ syllable segmentation, envelope-valley DP phrase detection (15–40 s), 12-band vocal-tract formant cross-synthesis, same-sample granular Overlap-Add sustain, farthest-point Answer Mode, mobile/iOS Safari Web Audio unlock, and startup prompt.

## Running Locally

```bash
npm install
npm run dev
```

To build the static bundle (`dist/`) and archive a new version snapshot (e.g., `v3`):

```bash
npm run build
npm run snapshot -- v3
npm run build
```

## License

Released into the public domain under [The Unlicense](LICENSE).
