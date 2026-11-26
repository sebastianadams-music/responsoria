/**
 * Inverts a point in [0, 1]^2 across both axes and swaps interior <-> perimeter:
 * points near the center map near the edges/corners and vice versa, while flipping X and Y.
 */
export function invertPoint(x, y, step = 0) {
  const u = x - 0.5;
  const v = y - 0.5;
  const rho = 2 * Math.max(Math.abs(u), Math.abs(v));

  if (rho < 1e-4) {
    const angle = step * 2.399963229728653;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const scale = 0.48 / Math.max(Math.abs(cos), Math.abs(sin));
    return {
      x: 0.5 + cos * scale,
      y: 0.5 + sin * scale,
    };
  }

  const invScale = -(1 - rho) / rho;
  return {
    x: Math.min(1, Math.max(0, 0.5 + u * invScale)),
    y: Math.min(1, Math.max(0, 0.5 + v * invScale)),
  };
}

/**
 * Concatenative segment selector supporting k-NN (k=5) user selection
 * and phrase-level inverted-space answer selection.
 */
export class SegmentSelector {
  /**
   * @param {Array} rawSegments Array of [spriteIdx, startSec, durSec, x, y, gainComp] or normalized objects
   */
  constructor(rawSegments = []) {
    this.segments = rawSegments.map((s) =>
      Array.isArray(s)
        ? {
            spriteIdx: s[0],
            start: s[1],
            dur: s[2],
            x: s[3],
            y: s[4],
            gainComp: s[5] ?? 1.0,
          }
        : s
    );
    this.userPhraseHistory = [];
    this.answerSourceHistory = [];
    this.answerStep = 0;
    this.currentAnswerRef = null;
    this.lastPlayedIdx = -1;
    this.lastUserPos = { x: 0.5, y: 0.5 };
  }

  updateLastUserPos(x, y) {
    this.lastUserPos = { x, y };
  }

  _candidateIndices(availableSprites) {
    const out = [];
    for (let i = 0; i < this.segments.length; i++) {
      if (!availableSprites || availableSprites.has(this.segments[i].spriteIdx)) {
        out.push(i);
      }
    }
    return out;
  }

  _pickFromTopK(scored, k, ascending) {
    if (scored.length === 0) return null;
    scored.sort((a, b) => (ascending ? a.distSq - b.distSq : b.distSq - a.distSq));
    const topCount = Math.min(k, scored.length);
    const pool = scored.slice(0, topCount);

    let filtered = pool;
    if (topCount > 1 && this.lastPlayedIdx !== -1) {
      const withoutLast = pool.filter((item) => item.idx !== this.lastPlayedIdx);
      if (withoutLast.length > 0) {
        filtered = withoutLast;
      }
    }

    const chosen = filtered[Math.floor(Math.random() * filtered.length)];
    this.lastPlayedIdx = chosen.idx;
    return { idx: chosen.idx, seg: this.segments[chosen.idx] };
  }

  /**
   * Select one of the k=5 nearest segments to (x, y).
   * Records selection into userPhraseHistory only when record=true (primary note change).
   */
  selectUser(x, y, k = 5, availableSprites = null, { record = true } = {}) {
    this.lastUserPos = { x, y };
    const candidates = this._candidateIndices(availableSprites);
    if (candidates.length === 0) return null;

    const scored = new Array(candidates.length);
    for (let i = 0; i < candidates.length; i++) {
      const idx = candidates[i];
      const seg = this.segments[idx];
      const dx = seg.x - x;
      const dy = seg.y - y;
      scored[i] = { idx, distSq: dx * dx + dy * dy };
    }

    const result = this._pickFromTopK(scored, k, true);
    if (result && record) {
      this.userPhraseHistory.push({
        x: result.seg.x,
        y: result.seg.y,
        idx: result.idx,
      });
    }
    return result;
  }

  /**
   * Snapshot the completed user phrase for answer mode and clear user history.
   */
  beginAnswerPhrase() {
    if (this.userPhraseHistory.length > 0) {
      this.answerSourceHistory = this.userPhraseHistory.slice();
    } else {
      this.answerSourceHistory = [{ x: this.lastUserPos.x, y: this.lastUserPos.y, idx: -1 }];
    }
    this.answerStep = 0;
    this.currentAnswerRef = null;
    this.userPhraseHistory = [];
  }

  /**
   * Select one of the k=5 nearest segments to the inverted coordinates of the
   * corresponding user selection (swapping center <-> perimeter and flipping X/Y axes).
   * When advance=false (bridging sustain grain), reuses currentAnswerRef.
   */
  selectAnswer(k = 5, availableSprites = null, { advance = true } = {}) {
    if (this.answerSourceHistory.length === 0) {
      this.answerSourceHistory = [{ x: this.lastUserPos.x, y: this.lastUserPos.y, idx: -1 }];
    }

    if (advance || !this.currentAnswerRef) {
      this.currentAnswerRef = this.answerSourceHistory[this.answerStep % this.answerSourceHistory.length];
      this.answerStep += 1;
    }
    const ref = this.currentAnswerRef;
    const target = invertPoint(ref.x, ref.y, this.answerStep);

    const candidates = this._candidateIndices(availableSprites);
    if (candidates.length === 0) return null;

    const scored = new Array(candidates.length);
    for (let i = 0; i < candidates.length; i++) {
      const idx = candidates[i];
      const seg = this.segments[idx];
      const dx = seg.x - target.x;
      const dy = seg.y - target.y;
      scored[i] = { idx, distSq: dx * dx + dy * dy };
    }

    return this._pickFromTopK(scored, k, true);
  }

  beginUserPhrase() {
    this.userPhraseHistory = [];
  }
}
