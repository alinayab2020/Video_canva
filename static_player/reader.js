/**
 * Chunked byte-stream accumulator.
 *
 * The old design kept one contiguous buffer and re-copied it on EVERY push()
 * and unshift() — O(total²) time for a stream delivered in N chunks (a 100 MB
 * .ascf in 64 KiB fetch chunks re-copied ~800 GB). This version stores the
 * chunks in a list and copies only the exact bytes a read() asks for:
 * amortized O(1) push, O(bytes read) total copy work, no pathological cases.
 *
 * Semantics are unchanged: read(n) returns exactly n bytes or null; unshift()
 * returns bytes to the front; `length` is the buffered byte count.
 */
class ByteStreamParser {
    constructor() {
        this._chunks = [];
        this._length = 0;
    }

    push(chunk) {
        if (chunk && chunk.length) {
            this._chunks.push(chunk);
            this._length += chunk.length;
        }
    }

    // Consume exactly `bytes` from the front, or return null when short.
    read(bytes) {
        if (this._length < bytes) return null;
        const out = new Uint8Array(bytes);
        let off = 0, need = bytes;
        while (need > 0) {
            const head = this._chunks[0];
            const take = Math.min(need, head.length);
            out.set(head.subarray(0, take), off);
            if (take === head.length) this._chunks.shift();
            else this._chunks[0] = head.subarray(take);
            off += take;
            need -= take;
        }
        this._length -= bytes;
        return out;
    }

    // Peek at the next `bytes` without consuming them (replaces peeking at the
    // old .buffer field, which no longer exists).
    peek(bytes) {
        if (this._length < bytes) return null;
        const out = new Uint8Array(bytes);
        let off = 0, need = bytes, i = 0;
        while (need > 0) {
            const chunk = this._chunks[i++];
            const take = Math.min(need, chunk.length);
            out.set(chunk.subarray(0, take), off);
            off += take;
            need -= take;
        }
        return out;
    }

    // Push bytes back to the front of the buffer (for header rollback)
    unshift(chunk) {
        if (chunk && chunk.length) {
            this._chunks.unshift(chunk);  // O(#chunks) pointer move, ~free
            this._length += chunk.length;
        }
    }

    get length() { return this._length; }
}

const textDecoder = new TextDecoder();
const CHAR_LUT = new Array(128);
for (let i = 0; i < 128; i++) CHAR_LUT[i] = String.fromCharCode(i);

class AscilinePlayer {
    static instances = [];

    constructor(containerEl, options = {}) {
        this.container = containerEl;
        
        // Find inner elements by class instead of ID
        this.player = this.container.querySelector('.ascii-player');
        this.canvas = this.container.querySelector('.ascii-canvas');
        this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
        this.statusEl = this.container.querySelector('.status') || document.createElement('div');
        this.overlay = this.container.querySelector('.play-overlay') || document.createElement('div');
        this.audioEl = this.container.querySelector('.ascii-audio');

        this.loop = options.loop !== undefined ? options.loop : true;
        this.muted = options.muted || false;
        this.bufferAll = options.bufferAll || false;

        this.state = 'IDLE'; // IDLE | PLAYING | PAUSED
        this.frameBuffer = [];
        this.allFrames = []; // Used if bufferAll is true
        this.BUFFER_SIZE = 60;
        this.codecDecoder = null;
        this.targetFps = 24;
        this.renderMode = 1;
        this.pixelMode = false;
        this.readyToRender = false;
        this.pauseStartTime = 0;
        this.streamStartTime = 0;
        this.currentAscfUrl = '';
        this.currentAudioUrl = '';

        this.gridCols = 0;
        this.gridRows = 0;
        this.charWidth = 0;
        this.charHeight = 0;
        this.xPos = null;
        this.yPos = null;
        this.dotImageData = null;
        this.dotImageData32 = null;
        this.selectionBuffer = null;
        // Canvas color-string cache (shared helper; see codec.js)
        this._cssRGB = (typeof AscilineCodec !== 'undefined' && AscilineCodec.makeColorCache)
            ? AscilineCodec.makeColorCache()
            : (r, g, b) => `rgb(${r},${g},${b})`;

        this.lastRenderTime = 0;
        this.frameCount = 0;
        this.currentFps = 0;
        this.lastFpsUpdate = 0;

        this.isStreaming = false;
        this.fetchAbortController = null;
        this.decodeQueue = Promise.resolve();
        this.pendingDecodes = 0;

        this.totalFrames = 0;   // read from header
        this.totalDuration = 0; // seconds, derived from totalFrames / targetFps
        this.currentTime = 0;   // seconds, updated each rendered frame

        this.renderFrame = this.renderFrame.bind(this);
        this.togglePause = this.togglePause.bind(this);

        this.container.addEventListener('click', (e) => {
            if (e.target.closest('.play-overlay')) return;
            if (window.getSelection().toString().length > 0) return;
            this.togglePause();
        });

        // Handle overlay click to play
        if (this.overlay) {
            this.overlay.addEventListener('click', () => {
                if (this.currentAscfUrl) {
                    this.play(this.currentAscfUrl, this.currentAudioUrl);
                }
            });
        }

        window.addEventListener('resize', () => {
            if (this.gridCols > 0 && this.gridRows > 0) {
                this.buildCanvas(this.gridCols, this.gridRows);
            }
        });

        AscilinePlayer.instances.push(this);
    }

    buildCanvas(cols, rows) {
        this.gridCols = cols;
        this.gridRows = rows;

        // For ASCII mode, measure actual char dimensions FIRST so container height uses correct aspect ratio
        let charWForLayout = 1, charHForLayout = 1;
        if (!this.pixelMode) {
            this.ctx.font = 'bold 8px Courier New';
            charWForLayout = this.ctx.measureText('M').width;
            charHForLayout = 8;
        }

        // Set container height based on TRUE aspect ratio (accounts for non-square characters)
        const containerW = this.container.clientWidth || window.innerWidth;
        const trueCanvasW = this.pixelMode ? cols : cols * charWForLayout;
        const trueCanvasH = this.pixelMode ? rows : rows * charHForLayout;
        const naturalH = Math.round(containerW * (trueCanvasH / trueCanvasW));
        const maxH = Math.round(window.innerHeight * 0.72);
        const containerH = Math.min(naturalH, maxH);
        this.container.style.height = containerH + 'px';

        const syncSize = (el) => {
            el.style.width  = this.container.clientWidth + 'px';
            el.style.height = this.container.clientHeight + 'px';
            el.style.objectFit = 'contain';
            el.style.position = 'absolute';
            el.style.top = '0';
            el.style.left = '0';
        };

        if (this.pixelMode) {
            this.canvas.width  = cols;
            this.canvas.height = rows;
            this.canvas.style.display = 'block';
            this.canvas.style.imageRendering = 'pixelated';
            this.dotImageData = this.ctx.createImageData(cols, rows);
            this.dotImageData32 = new Uint32Array(this.dotImageData.data.buffer);
            const d = this.dotImageData.data;
            for (let i = 3; i < d.length; i += 4) d[i] = 255;
            syncSize(this.canvas);
            this.player.style.display = 'none';
        } else {
            this.canvas.style.imageRendering = '';
            this.dotImageData = null;
            this.dotImageData32 = null;
            this.charWidth = charWForLayout;
            this.charHeight = charHForLayout;
            this.canvas.width  = cols * this.charWidth;
            this.canvas.height = rows * this.charHeight;
            this.canvas.style.display = 'block';

            this.selectionBuffer = new Uint8Array((cols + 1) * rows);
            for (let r = 0; r < rows; r++) this.selectionBuffer[r * (cols + 1) + cols] = 10;

            syncSize(this.canvas);

            const fitScale  = Math.min(this.container.clientWidth / this.canvas.width, this.container.clientHeight / this.canvas.height);
            const offsetX   = (this.container.clientWidth - (this.canvas.width * fitScale)) / 2;
            const offsetY   = (this.container.clientHeight - (this.canvas.height * fitScale)) / 2;

            // Position pre at exact VISUAL size (no transform scale) so text selection works correctly
            const visW = this.canvas.width * fitScale;
            const visH = this.canvas.height * fitScale;
            const scaledCharW = this.charWidth * fitScale;
            const scaledCharH = this.charHeight * fitScale;

            this.player.style.width  = visW + 'px';
            this.player.style.height = visH + 'px';
            this.player.style.position = 'absolute';
            this.player.style.top = offsetY + 'px';
            this.player.style.left = offsetX + 'px';
            this.player.style.transform = 'none';
            this.player.style.fontSize = scaledCharH + 'px';
            this.player.style.lineHeight = scaledCharH + 'px';
            this.player.style.letterSpacing = '0px';

            this.ctx.font = 'bold 8px Courier New';
            this.ctx.textBaseline = 'top';
            this.xPos = new Float32Array(cols);
            this.yPos = new Float32Array(rows);
            for (let c = 0; c < cols; c++) this.xPos[c] = c * this.charWidth;
            for (let r = 0; r < rows; r++) this.yPos[r] = r * this.charHeight;
        }
    }

    async play(ascfUrl, audioUrl = null) {
        if (this.state !== 'IDLE') return;
        
        this.currentAscfUrl = ascfUrl;
        this.currentAudioUrl = audioUrl;

        if (this.overlay) this.overlay.classList.add('hidden');
        if (this.statusEl) {
            this.statusEl.textContent = 'Downloading...';
            this.statusEl.style.color = 'var(--accent-primary, var(--accent-color))';
        }
        
        this.isStreaming = true;
        this.fetchAbortController = new AbortController();
        
        try {
            const response = await fetch(ascfUrl, { signal: this.fetchAbortController.signal });
            if (!response.ok) throw new Error("File not found");
            
            if (audioUrl && this.audioEl) {
                this.audioEl.pause();
                this.audioEl.src = audioUrl;
                this.audioEl.currentTime = 0;
                this.audioEl.muted = this.muted;
                this.audioEl.load();
            }

            const reader = response.body.getReader();
            const parser = new ByteStreamParser();
            let headerParsed = false;
            
            while (this.isStreaming) {
                if (!this.bufferAll && (this.frameBuffer.length + this.pendingDecodes) >= 90) {
                    await new Promise(r => setTimeout(r, 50));
                    continue;
                }
                const { done, value } = await reader.read();
                if (value) parser.push(value);
                
                if (!headerParsed && parser.length >= 18) {
                    // Read 18 bytes and detect format by magic:
                    // 'ASCF' = legacy 14-byte header (no totalFrames)
                    // 'ASC2' = v2 18-byte header (includes totalFrames uint32)
                    const headerBytes = parser.read(18);
                    const magic = textDecoder.decode(headerBytes.subarray(0, 4));
                    if (magic !== 'ASCF' && magic !== 'ASC2') throw new Error("Invalid ASCF file");

                    const view = new DataView(headerBytes.buffer, headerBytes.byteOffset, headerBytes.byteLength);
                    this.targetFps = view.getFloat32(4, false);
                    this.renderMode = view.getUint8(8);
                    this.pixelMode = view.getUint8(9) === 1;
                    const cols = view.getUint16(10, false);
                    const rows = view.getUint16(12, false);

                    if (magic === 'ASC2') {
                        // v2: last 4 bytes are totalFrames
                        this.totalFrames = view.getUint32(14, false);
                        this.totalDuration = this.totalFrames > 0 ? this.totalFrames / this.targetFps : 0;
                    } else {
                        // Legacy 'ASCF': no totalFrames — push the 4 extra bytes back as frame data
                        this.totalFrames = 0;
                        this.totalDuration = 0;
                        parser.unshift(headerBytes.subarray(14, 18));
                    }

                    this.buildCanvas(cols, rows);
                    
                    if (typeof AscilineCodec !== 'undefined' && this.renderMode > 1) {
                        this.codecDecoder = AscilineCodec.makeDecoder(this.pixelMode ? 3 : 4);
                    }
                    
                    headerParsed = true;
                    this.readyToRender = false;
                    this.state = 'PLAYING';
                    
                    const beginRendering = () => {
                        if (this.readyToRender) return;
                        this.readyToRender = true;
                        this.streamStartTime = performance.now();
                        this.lastRenderTime = performance.now();
                        this.lastFpsUpdate = this.lastRenderTime;
                        requestAnimationFrame(this.renderFrame);
                    };

                    if (this.audioEl && audioUrl) {
                        this.audioEl.play().catch(() => {});
                        if (this.audioEl.readyState >= 3) {
                            beginRendering();
                        } else {
                            this.audioEl.addEventListener('playing', beginRendering, { once: true });
                            setTimeout(() => { if (!this.readyToRender) beginRendering(); }, 500);
                        }
                    } else {
                        beginRendering();
                    }
                }
                
                if (headerParsed) {
                    while (parser.length >= 4) {
                        const head = parser.peek(4);
                        const view = new DataView(head.buffer, head.byteOffset, 4);
                        const frameLen = view.getUint32(0, false);

                        if (parser.length >= 4 + frameLen) {
                            parser.read(4);
                            const frameBytes = parser.read(frameLen);
                            
                            if (this.renderMode === 1) {
                                const text = textDecoder.decode(frameBytes);
                                const newlineIdx = text.indexOf('\n');
                                const frameIndex = parseInt(text.substring(0, newlineIdx));
                                const frameTime = frameIndex / this.targetFps;
                                const frameData = text.substring(newlineIdx + 1);
                                if (this.bufferAll) this.allFrames.push({ data: frameData, time: frameTime });
                                this.frameBuffer.push({ data: frameData, time: frameTime });
                            } else if (this.codecDecoder) {
                                const capturedBytes = frameBytes;
                                this.pendingDecodes++;
                                this.decodeQueue = this.decodeQueue.then(async () => {
                                    // Yield to the event loop to prevent blocking requestAnimationFrame 
                                    // when a large batch of synchronous DCT/ZLIB decodes arrives.
                                    await new Promise(r => setTimeout(r, 0));
                                    
                                    try {
                                        const { frameIndex, frame } = await this.codecDecoder.decode(capturedBytes);
                                        const frameTime = frameIndex / this.targetFps;
                                        if (this.bufferAll) this.allFrames.push({ data: frame, time: frameTime });
                                        this.frameBuffer.push({ data: frame, time: frameTime });
                                    } finally {
                                        this.pendingDecodes--;
                                    }
                                });
                            }
                        } else {
                            break;
                        }
                    }
                }
                
                if (done) {
                    if (this.statusEl) this.statusEl.textContent = 'Stream complete.';
                    this.isStreaming = false;
                    break;
                }
            }
        } catch (e) {
            if (e.name !== 'AbortError') {
                if (this.statusEl) {
                    this.statusEl.textContent = 'Playback Error: ' + e.message;
                    this.statusEl.style.color = '#ff0000';
                }
                setTimeout(() => this.finishStream(), 3000);
            }
        }
    }

    renderFrame(now) {
        if (this.state !== 'PLAYING' || !this.readyToRender) return;
        requestAnimationFrame(this.renderFrame);

        let masterClock;
        if (this.audioEl && this.audioEl.readyState >= 1 && !this.audioEl.paused) {
            masterClock = this.audioEl.currentTime;
        } else {
            masterClock = (now - this.streamStartTime) / 1000.0;
        }

        if (this.frameBuffer.length === 0) {
            if (this.isStreaming || this.pendingDecodes > 0) {
                if (this.statusEl) this.statusEl.textContent = 'Buffering...';
            } else {
                if (this.loop) {
                    if (this.statusEl) this.statusEl.textContent = 'Looping...';
                    if (this.audioEl) this.audioEl.pause();
                    this.state = 'IDLE';
                    if (this.currentAscfUrl) {
                        this.play(this.currentAscfUrl, this.currentAudioUrl);
                    }
                } else {
                    this.finishStream();
                }
            }
            return;
        }

        while (this.frameBuffer.length > 1 && this.frameBuffer[0].time < masterClock - 0.1) {
            this.frameBuffer.shift();
        }

        if (this.frameBuffer[0].time > masterClock + 0.05) {
            return;
        }

        const frameObj = this.frameBuffer.shift();
        const frame = frameObj.data;
        this.currentTime = frameObj.time;

        this.frameCount++;
        if (now - this.lastFpsUpdate >= 1000) {
            this.currentFps = this.frameCount;
            this.frameCount = 0;
            this.lastFpsUpdate = now;
            const modes = { 2: '64 Color', 3: '512 Color', 4: '32K Color', 5: '262K Color', 6: '16M Ultra' };
            const label = (modes[this.renderMode] || 'B&W') + (this.pixelMode ? ' PIXEL' : '');
            const elapsed = this.currentTime;
            const fmt = t => `${String(Math.floor(t/60)).padStart(2,'0')}:${String(Math.floor(t%60)).padStart(2,'0')}`;
            const timeStr = this.totalDuration > 0 ? ` | ${fmt(elapsed)} / ${fmt(this.totalDuration)}` : '';
            if (this.statusEl) {
                this.statusEl.textContent = `FPS: ${this.currentFps}/${Math.round(this.targetFps)} | Buf: ${this.frameBuffer.length}${timeStr} | ${label}`;
            }
        }

        this.lastRenderTime = now;

        if (this.renderMode === 1) {
            this.player.style.display = 'block';
            this.player.style.color = 'currentColor';
            this.player.textContent = frame;
        } else if (this.pixelMode) {
            // One 32-bit store per pixel via the shared packer (see codec.js)
            if (typeof AscilineCodec !== 'undefined' && AscilineCodec.packBGRtoRGBA32) {
                AscilineCodec.packBGRtoRGBA32(frame, this.dotImageData32);
            } else {
                const view = frame;
                const data = this.dotImageData.data;
                for (let src = 0, dst = 0; src < view.length; src += 3, dst += 4) {
                    data[dst]     = view[src + 2];
                    data[dst + 1] = view[src + 1];
                    data[dst + 2] = view[src];
                }
            }
            this.ctx.putImageData(this.dotImageData, 0, 0);
        } else {
            const view = frame;
            this.ctx.fillStyle = '#050505';
            this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
            this.ctx.font = 'bold 8px Courier New';
            this.ctx.textBaseline = 'top';

            let col = 0, row = 0, prevPacked = -1;
            for (let idx = 0; idx < view.length; idx += 4) {
                const packed = (view[idx+1] << 16) | (view[idx+2] << 8) | view[idx+3];
                if (packed !== prevPacked) {
                    this.ctx.fillStyle = this._cssRGB(view[idx+1], view[idx+2], view[idx+3]);
                    prevPacked = packed;
                }
                this.ctx.fillText(CHAR_LUT[view[idx]], this.xPos[col], this.yPos[row]);
                this.selectionBuffer[row * (this.gridCols + 1) + col] = view[idx];
                col++;
                if (col >= this.gridCols) { col = 0; row++; }
            }

            this.player.style.display = 'block';
            this.player.style.color = 'transparent';
            this.player.textContent = textDecoder.decode(this.selectionBuffer);
        }
    }

    stop() {
        this.finishStream();
    }

    finishStream() {
        this.state = 'IDLE';
        this.isStreaming = false;
        this.frameBuffer = [];
        this.allFrames = [];
        this.decodeQueue = Promise.resolve();
        this.pendingDecodes = 0;
        if (this.fetchAbortController) {
            this.fetchAbortController.abort();
            this.fetchAbortController = null;
        }
        if (this.audioEl) { this.audioEl.pause(); this.audioEl.src = ''; }
        if (this.ctx && this.canvas) this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        if (this.player) {
            this.player.textContent = '';
            this.player.style.display = 'none';
        }
        this.container.classList.remove('paused');
        if (this.overlay) this.overlay.classList.remove('hidden');
        if (this.statusEl) {
            this.statusEl.textContent = 'Ready';
            this.statusEl.style.color = 'rgba(255,255,255,0.6)';
        }
        this.readyToRender = false;
        this.pauseStartTime = 0;
    }

    togglePause() {
        if (this.state === 'PLAYING') {
            this.state = 'PAUSED';
            this.pauseStartTime = performance.now();
            
            if (this.audioEl && !this.audioEl.paused) {
                this.audioEl.pause();
            }
            
            this.container.classList.add('paused');
            if (this.statusEl) {
                this.statusEl.textContent = '❚❚ PAUSED';
                this.statusEl.style.color = '#888';
            }
        } else if (this.state === 'PAUSED') {
            this.state = 'PLAYING';
            
            const pauseDuration = performance.now() - this.pauseStartTime;
            this.streamStartTime += pauseDuration;
            this.pauseStartTime = 0;
            
            if (this.audioEl && this.audioEl.paused) {
                this.audioEl.play().catch(() => {});
            }

            this.container.classList.remove('paused');
            if (this.statusEl) {
                this.statusEl.textContent = 'Resuming...';
                this.statusEl.style.color = 'var(--accent-primary, var(--accent-color))';
            }
            
            this.lastRenderTime = performance.now();
            this.lastFpsUpdate = performance.now();
            this.frameCount = 0;
            requestAnimationFrame(this.renderFrame);
        }
    }

    seek(targetSec) {
        if (!this.bufferAll) return;
        if (this.totalDuration > 0) {
            targetSec = Math.max(0, Math.min(targetSec, this.totalDuration));
        }

        this.frameBuffer = this.allFrames.filter(f => f.time >= targetSec);

        if (this.audioEl) {
            this.audioEl.currentTime = targetSec;
            if (this.state === 'PLAYING') {
                this.audioEl.play().catch(() => {});
            }
        }

        this.streamStartTime = performance.now() - (targetSec * 1000);
        this.currentTime = targetSec;

        if (this.state !== 'PLAYING') {
            const tempState = this.state;
            this.state = 'PLAYING';
            this.renderFrame(performance.now());
            this.state = tempState;
        }
    }
}

// Global Spacebar Pause for all instances (browser only; guard keeps the
// module importable by the Node test harness)
if (typeof document !== 'undefined') {
    document.addEventListener('keydown', (e) => {
        if (e.code === 'Space') {
            e.preventDefault();
            AscilinePlayer.instances.forEach(player => {
                if (player.state === 'PLAYING' || player.state === 'PAUSED') {
                    player.togglePause();
                }
            });
        }
    });
}

// Node test harness hook (no effect in browsers). Placed last so every class
// binding is initialized by the time a require() evaluates this.
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ByteStreamParser, AscilinePlayer };
}
