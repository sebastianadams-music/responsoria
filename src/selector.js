/**
 * Concatenative segment selector supporting k-NN (k=5) user selection
 * and phrase-level farthest-distance answer selection.
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
   * Select one of the k=5 farthest segments from the corresponding user selection.
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

    const candidates = this._candidateIndices(availableSprites);
    if (candidates.length === 0) return null;

    const scored = new Array(candidates.length);
    for (let i = 0; i < candidates.length; i++) {
      const idx = candidates[i];
      const seg = this.segments[idx];
      const dx = seg.x - ref.x;
      const dy = seg.y - ref.y;
      scored[i] = { idx, distSq: dx * dx + dy * dy };
    }

    return this._pickFromTopK(scored, k, false);
  }

  beginUserPhrase() {
    this.userPhraseHistory = [];
  }
}
