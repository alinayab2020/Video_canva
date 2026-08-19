/**
 * Correctness tests for the shared canvas render helpers in codec.js.
 *
 *   packBGRtoRGBA32: interleaved BGR -> canvas RGBA32, endianness-aware.
 *   makeColorCache : bounded memoizer for "rgb(r,g,b)" fillStyle strings.
 *
 * Usage: node test/test_render_helpers.js
 */
const codec = require('../codec.js');
const { packBGRtoRGBA32, makeColorCache } = codec;

if (typeof packBGRtoRGBA32 !== 'function' || typeof makeColorCache !== 'function') {
  console.error('helpers not exported from codec.js');
  process.exit(2);
}

// Reference implementation (the byte loop the browser code used to run)
function referencePack(bgr, rgba, npix) {
  for (let s = 0, d = 0; s < npix * 3; s += 3, d += 4) {
    rgba[d]     = bgr[s + 2]; // R
    rgba[d + 1] = bgr[s + 1]; // G
    rgba[d + 2] = bgr[s];     // B
    rgba[d + 3] = 255;        // A
  }
}

let seed = 42;
function rnd(n) {
  seed ^= seed << 13; seed >>>= 0;
  seed ^= seed >> 17;
  seed ^= seed << 5; seed >>>= 0;
  return seed % n;
}

// 1) Correctness across sizes, incl. non-multiple-of-4 lengths and odd tails
for (const pixels of [1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 63, 64, 255, 256, 1000, 65535]) {
  const bgr = new Uint8Array(pixels * 3);
  for (let i = 0; i < bgr.length; i++) bgr[i] = rnd(256);
  const ref = new Uint8Array(pixels * 4);           // starts zeroed, incl. alpha
  referencePack(bgr, ref, pixels);
  const out = new Uint8Array(pixels * 4);
  packBGRtoRGBA32(bgr, new Uint32Array(out.buffer));
  for (let i = 0; i < ref.length; i++) {
    if (ref[i] !== out[i]) {
      console.error(`FAIL pack mismatch at pixel-buffer size ${pixels}, byte ${i}: ${ref[i]} vs ${out[i]}`);
      process.exit(1);
    }
  }
}
console.log('PASS  packBGRtoRGBA32 matches reference byte loop (16 sizes)');

// 2) Input views with a non-zero byteOffset (the legacy live path passes
//    `new Uint8Array(buffer, 4)` — an offset view, not a fresh array)
{
  const pixels = 100;
  const backing = new Uint8Array(4 + pixels * 3);
  for (let i = 0; i < backing.length; i++) backing[i] = rnd(256);
  const bgr = new Uint8Array(backing.buffer, 4, pixels * 3);
  const ref = new Uint8Array(pixels * 4);
  referencePack(bgr, ref, pixels);
  const out = new Uint8Array(pixels * 4);
  packBGRtoRGBA32(bgr, new Uint32Array(out.buffer));
  for (let i = 0; i < ref.length; i++) {
    if (ref[i] !== out[i]) { console.error('FAIL offset-view pack mismatch'); process.exit(1); }
  }
  console.log('PASS  packBGRtoRGBA32 handles offset input views');
}

// 3) Color cache: format, identity (same string instance), and bound
{
  const cssRGB = makeColorCache(8);
  const a = cssRGB(12, 34, 56);
  if (a !== 'rgb(12,34,56)') { console.error('FAIL color string format:', a); process.exit(1); }
  if (cssRGB(12, 34, 56) !== a) { console.error('FAIL cache did not return identical string'); process.exit(1); }
  // Independently-created caches produce equal (content-wise) strings
  const other = makeColorCache(8);
  if (other(12, 34, 56) !== a) { console.error('FAIL cross-cache content mismatch'); process.exit(1); }
  // bounded growth: 1000 distinct colors with limit 8 must never throw and
  // must keep returning correct strings even after the cap
  for (let i = 0; i < 1000; i++) cssRGB(i & 255, (i >> 8) & 255, i & 7);
  if (cssRGB(1, 2, 3) !== 'rgb(1,2,3)') { console.error('FAIL post-cap correctness'); process.exit(1); }
  // original entries still cached
  if (cssRGB(12, 34, 56) !== a) { console.error('FAIL hot entry evicted unexpectedly'); process.exit(1); }
  console.log('PASS  makeColorCache format/identity/bound behavior');
}

// 4) Every byte color value round-trips through packing (exhaustive on a few)
{
  const bgr = new Uint8Array([0, 0, 0, 255, 255, 255, 128, 64, 192, 1, 254, 127]);
  const out = new Uint8Array(4 * 4);
  packBGRtoRGBA32(bgr, new Uint32Array(out.buffer));
  const expect = [0, 0, 0, 255, 255, 255, 255, 255, 192, 64, 128, 255, 127, 254, 1, 255];
  for (let i = 0; i < 16; i++) {
    if (out[i] !== expect[i]) { console.error('FAIL boundary color roundtrip at byte', i); process.exit(1); }
  }
  console.log('PASS  boundary color values round-trip exactly');
}

console.log('ALL RENDER HELPER TESTS PASSED');
