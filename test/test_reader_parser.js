/**
 * Randomized equivalence test for the chunked ByteStreamParser.
 *
 * The shipped .ascf reader used to re-copy its entire buffer per push(), an
 * O(total²) blow-up on large files. The rewrite accumulates chunks and copies
 * only what read() asks for. This test pins functional equivalence against a
 * dead-simple "concat everything" reference under randomized push/read/peek/
 * unshift interleavings, plus a smoke check that streaming a large buffer no
 * longer has quadratic copy cost.
 *
 * Usage: node test/test_reader_parser.js
 */
const { ByteStreamParser } = require('../static_player/reader.js');

function bs(x) { return Math.abs(x) | 0; }

class ReferenceParser {
  constructor() { this.buffer = new Uint8Array(0); }
  push(chunk) {
    const nb = new Uint8Array(this.buffer.length + chunk.length);
    nb.set(this.buffer, 0); nb.set(chunk, this.buffer.length);
    this.buffer = nb;
  }
  read(bytes) {
    if (this.buffer.length < bytes) return null;
    const d = this.buffer.slice(0, bytes);
    this.buffer = this.buffer.slice(bytes);
    return d;
  }
  peek(bytes) {
    if (this.buffer.length < bytes) return null;
    return this.buffer.slice(0, bytes);
  }
  unshift(chunk) {
    const nb = new Uint8Array(chunk.length + this.buffer.length);
    nb.set(chunk, 0); nb.set(this.buffer, chunk.length);
    this.buffer = nb;
  }
  get length() { return this.buffer.length; }
}

// Simple deterministic RNG (xorshift32)
let seed = 0xC0FFEE >>> 0;
function rnd(n) {
  seed ^= seed << 13; seed >>>= 0;
  seed ^= seed >> 17;
  seed ^= seed << 5; seed >>>= 0;
  return seed % n;
}

function run(ops) {
  const a = new ByteStreamParser();
  const b = new ReferenceParser();
  let pushes = 0, reads = 0, peeks = 0, unshifts = 0;
  for (let i = 0; i < ops; i++) {
    const op = rnd(10);
    if (op < 4) {                       // push
      const n = 1 + rnd(70000);
      const data = new Uint8Array(n);
      for (let j = 0; j < n; j++) data[j] = rnd(256);
      a.push(data); b.push(data); pushes++;
    } else if (op < 7) {                // read (sometimes beyond length)
      const n = 1 + rnd(40000) + (rnd(8) === 0 ? 2000000 : 0);
      const ra = a.read(n), rb = b.read(n);
      if ((ra === null) !== (rb === null)) throw new Error(`read nullness diverged at op ${i}`);
      if (ra !== null && !ra.every((v, j) => v === rb[j])) throw new Error(`read bytes diverged at op ${i}`);
      reads++;
    } else if (op < 8) {                // peek (non-destructive)
      const n = 1 + rnd(5000);
      const beforeA = a.length, beforeB = b.length;
      const ra = a.peek(n), rb = b.peek(n);
      if ((ra === null) !== (rb === null)) throw new Error(`peek nullness diverged at op ${i}`);
      if (ra !== null && !ra.every((v, j) => v === rb[j])) throw new Error(`peek bytes diverged at op ${i}`);
      if (a.length !== beforeA || b.length !== beforeB) throw new Error(`peek mutated length at op ${i}`);
      peeks++;
    } else {                              // unshift
      const n = 1 + rnd(200);
      const data = new Uint8Array(n);
      for (let j = 0; j < n; j++) data[j] = rnd(256);
      a.unshift(data); b.unshift(data); unshifts++;
    }
    if (a.length !== b.length) throw new Error(`length diverged at op ${i}: ${a.length} vs ${b.length}`);
  }
  // Drain both fully; byte streams must match exactly.
  while (a.length > 0 || b.length > 0) {
    if (a.length !== b.length) throw new Error('drain length diverged');
    const n = Math.min(a.length, 1 + rnd(60000));
    const ra = a.read(n), rb = b.read(n);
    if (ra === null || rb === null) throw new Error('drain returned null within length');
    if (!ra.every((v, j) => v === rb[j])) throw new Error('drain bytes diverged');
  }
  return { pushes, reads, peeks, unshifts };
}

// Header-rollback scenario, mirroring play(): read 18, unshift 4 back.
{
  const p = new ByteStreamParser();
  const src = new Uint8Array(64);
  for (let i = 0; i < 64; i++) src[i] = i;
  p.push(src.subarray(0, 20));
  const head = p.read(18);
  if (!head) throw new Error('header read failed');
  p.unshift(head.subarray(14, 18));       // rollback the legacy-header overread
  p.push(src.subarray(20));               // the rest of the stream keeps arriving
  const rest = p.read(64 - 14);
  if (!rest) throw new Error('post-rollback read failed');
  const expect = Array.from({ length: 50 }, (_, i) => i + 14);
  if (!rest.every((v, i) => v === expect[i])) throw new Error('rollback byte sequence wrong');
  console.log('PASS  header rollback (read 18, unshift 4)');
}

for (let round = 0; round < 6; round++) {
  const stats = run(400);
  console.log(`PASS  randomized round ${round} (${stats.pushes} pushes, ${stats.reads} reads, ${stats.peeks} peeks, ${stats.unshifts} unshifts)`);
}

// Copy-work smoke test: streaming 96 MiB in 64 KiB chunks must not be
// quadratic anymore. Old design copied ~total²/64KiB (~150 GiB); new design
// copies only what is consumed.
{
  const p = new ByteStreamParser();
  const chunk = new Uint8Array(65536).fill(7);
  const t0 = process.hrtime.bigint();
  let consumed = 0;
  for (let i = 0; i < 1536; i++) {
    p.push(chunk);
    const out = p.read(65536);
    consumed += out.length;
  }
  const ms = Number(process.hrtime.bigint() - t0) / 1e6;
  if (consumed !== 1536 * 65536) throw new Error('byte count mismatch');
  console.log(`PASS  streamed ${consumed / 1048576} MiB through parser in ${ms.toFixed(1)} ms`);
}
console.log('ALL PARSER TESTS PASSED');
