# Canvas loading investigation

Measured against `be9c1cadcd348ffd314b5f1648ff802735c37189` on the default
`claude-code-painter` branch. No production canvases, paid models, storage migration,
scoring, model inputs, replay pixels, or explicit PNG exports changed.

## Decision

Keep authoritative and transported tiles as PNG in this change. Fix obsolete work,
viewport priority, decoded-cache retention and benchmark preview invalidation first.
A format replacement alone would still download hundreds of obsolete replay tiles
and recompose unchanged benchmark canvases. Lossless WebP remains a possible future
**cached derivative**, but the measured aggregate saving here is only 8.7%.

The frontend now caps fetch/decode work at six tiles, starts near the viewport center,
aborts obsolete fetches, rejects stale completions, and batches painting with animation
frames. Cache keys include the canvas ID. Decoded objects use LRU eviction and explicit
bitmap disposal: at most `max(256, visible tile count)` objects, with the existing
6,000-tile draw guard bounding the working set. Typical 256-tile retention is about
16 MiB at four bytes/pixel; extreme zoom can still approach 375 MiB. This is a limit,
not a claim of a small memory footprint at every zoom. A multi-resolution overview is
future work. Browsers without `createImageBitmap` use Image decoding instead.

Benchmark cards preserve connected image elements across polls. Their new
`preview.png?seq=N` URL composes the requested historical snapshot at a fixed maximum
900-pixel side, including sequence zero. Negative/future sequence numbers are rejected
before an immutable response can be cached. A per-server 16 MiB byte LRU serializes
cache misses to avoid duplicate encodes. Eviction or process restart can cause another
encode. The explicit `/export.png` download remains live, PNG, and `no-store`.

## Reproducible experiment

`scripts/measure_canvas_loading.py` creates deterministic 128-pixel RGB fixtures:
sparse strokes, a gradient, seeded textured paint and lettering. It serves the real
canvas HTML with automatic initialization disabled; the fixture supplies a 64-tile
viewport and nine rapidly superseded versions. No application database is read. Its
separate preview experiment creates and deletes a temporary SQLite database.

Chromium 154.0.8037.57, Pillow 12.3.0, local HTTP server, CDP network throttling:
200 ms latency, 64,000 bytes/second download (512 kbit/s), 32,000 bytes/second upload.
These are single recorded synthetic runs, not production percentiles. Timings include
browser scheduling and instrumentation overhead. First useful center means tile (3,3)
in the 8×8 viewport; full completion means all 64 requested current versions drawn.
Scrubbing advances eight times at 30 ms intervals. Warm means page reload with HTTP
cache retained, decoded JavaScript objects discarded. Request/body counts are observed
by the fixture server: aborted response bodies count in full, **not actual wire bytes**.

| Scenario | Before | After |
|---|---:|---:|
| Cold first tile | 271 ms | 255 ms |
| Cold center tile | 4,512 ms | 454 ms |
| Cold complete viewport | 13,325 ms | 13,404 ms |
| Cold requests / response-body bytes | 64 / 773,520 | 64 / 773,520 |
| Warm complete viewport | 67 ms | 188 ms |
| Warm requests / bytes | 0 / 0 | 0 / 0 |
| Warm tile draw calls | 2,080 | 362 |
| Scrub first current tile | 103,630 ms | 586 ms |
| Scrub center tile | 109,325 ms | 786 ms |
| Scrub complete viewport | 118,326 ms | 14,436 ms |
| Scrub requests | 576 | 105 |
| Scrub response-body bytes | 6,961,680 | 1,257,268 |
| Scrub decoded objects retained | 576 | 64 |

Stationary cold completion remains bandwidth-bound. Animation-frame batching does
not reduce cold draw calls here (2,080 both ways), because slow completions are spaced
across frames. Warm reload regresses by 121 ms with bounded fetch/bitmap decoding,
while repeat draws reuse decoded objects without requests. The large improvement is
center priority and discarding obsolete versions, not a blanket speedup for every case.

A 1,024×1,024 preview from the same fixture mix produced 384,147 PNG bytes. Three
uncached export calls took 59.95 / 57.52 / 56.31 ms; the preview cache took
59.30 / 0.07 / 0.01 ms. Browser polls retain the same image node and unchanged URL;
reloading can reuse its immutable HTTP response. First generation still costs about
60 ms, and a changed sequence requires a new derivative. Sequential preview misses
share a lock, deliberately avoiding an encoding stampede but limiting parallelism.

## Codec comparison

Aggregate of four 128×128 fixtures, mean of five encode/decode repetitions per fixture,
then summed across fixtures. Python decode costs are separate from browser decode.
Max error is the largest absolute 8-bit RGB channel difference after round trip; it is
not a perceptual-quality score. AVIF quality 100 is **not pixel-exact** in this encoder.

| Codec | Bytes | Encode ms | Decode ms | Max error |
|---|---:|---:|---:|---:|
| PNG | 48,345 | 4.33 | 0.87 | 0 |
| WebP lossless, method 4 | 44,154 | 37.08 | 0.99 | 0 |
| WebP quality 85 | 14,770 | 5.77 | 1.28 | 91 |
| AVIF quality 85, speed 6, 4:4:4 | 21,510 | 82.94 | 4.49 | 41 |
| AVIF quality 100, speed 6, 4:4:4 | 62,729 | 107.92 | 9.63 | 3 |
| JPEG XL lossless, effort 7 | 46,932 | 67.34 | 11.62 | 0 |

Lossless WebP helps sparse strokes (844→296 bytes), gradients (298→68), and lettering
(3,033→1,354), but texture dominates bytes (44,170→42,436). Lossy formats can save much
more on texture but can also enlarge tiny gradients/lettering and alter tile-edge pixels.
They are not justified for exact canvas viewing by this experiment. JPEG XL saves only
2.9% overall and the tested Chromium build rejects its image. The optional
`pillow-jxl-plugin` is used only for measurement; Pillow's installed production codecs
include WebP and AVIF but not JXL. No new production dependencies are introduced.

The script also attempts actual browser `Image.decode()` for all formats and records
support and one-shot decode latency in the JSON. Such tiny timings include scheduling
and initialization overhead; they are not a codec ranking. PNG/WebP/AVIF decode in the
tested browser; JXL does not. Current documentation lists broader modern browser support
for WebP/AVIF and warns JPEG XL needs a fallback. A future negotiated transport must
retain PNG fallback and cache derivatives, rather than transcode every request:
[MDN image formats](https://developer.mozilla.org/en-US/docs/Web/Media/Guides/Formats/Image_types),
[Pillow encoder options](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html),
[Google WebP documentation](https://developers.google.com/speed/webp).

## Commands and tests

```sh
# No model credentials or production data needed; use the installed browser.
export UV_CACHE_DIR=/tmp/uv-cache
export PLAYWRIGHT_CHROMIUM_EXECUTABLE=/usr/bin/chromium
uv run --with playwright --with pillow-jxl-plugin python scripts/measure_canvas_loading.py
# Baseline uses the same fixture and instrumentation, with old HTML only:
git show be9c1ca:src/conveyor/commons.html > /tmp/commons-before.html
CANVAS_HTML=/tmp/commons-before.html uv run --with playwright --with pillow-jxl-plugin python scripts/measure_canvas_loading.py
uv run --group dev pytest
uv run --with playwright --group dev pytest tests/test_canvas_loading_browser.py tests/test_brushlab_browser.py -q
```

New browser gates exercise actual fetches, stale completions, center priority, pan/zoom,
backward replay, canvas identity, eviction, warm HTTP cache, and unchanged benchmark
image identity, for both bitmap decoding and the Image fallback. Missing Chromium fails
these tests. Playwright itself is optional in the default suite, so run the explicit
browser command to ensure they execute. Backend tests cover historical pixels, repeated
encode avoidance, bounded eviction, invalid sequences, immutable HTTP headers and
unchanged export behavior. The existing full suite assumes Claude is installed in one
launcher-options test; this environment needs `tests/fake_claude.py` wrapped as `claude`
on PATH to satisfy that assumption (no paid calls).

Raw measurements: [before](canvas-before.json), [after](canvas-after.json).

Validation on this change: 312 passed in the default suite (two optional browser modules
skipped there); the explicit Chromium gate ran four tests with **four passed, zero skipped**,
including the existing Brush lab browser workflow. `git diff --check` passed. No CSS,
layout, or mobile UI implementation was changed.
