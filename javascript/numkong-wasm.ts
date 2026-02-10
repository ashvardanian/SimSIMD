/**
 * @brief WASM wrapper for NumKong providing N-API compatible interface
 * @file javascript/numkong-wasm.ts
 * @date February 6, 2026
 *
 * This module wraps the Emscripten-compiled WASM module to provide the same
 * TypeScript API as the native N-API bindings. It handles:
 * - Memory management (malloc/free) with zero-copy when possible
 * - TypedArray type detection and dispatch
 * - Result extraction from WASM heap
 * - Error handling
 */

/* #region Emscripten Interface */

/**
 * Emscripten module interface
 */
interface EmscriptenModule {
  _malloc(size: number): number;
  _free(ptr: number): void;
  wasmMemory: { buffer: ArrayBuffer };

  // Distance functions - f32
  _nk_dot_f32(a: number, b: number, n: number, result: number): void;
  _nk_angular_f32(a: number, b: number, n: number, result: number): void;
  _nk_sqeuclidean_f32(a: number, b: number, n: number, result: number): void;
  _nk_euclidean_f32(a: number, b: number, n: number, result: number): void;

  // Distance functions - f64
  _nk_dot_f64(a: number, b: number, n: number, result: number): void;
  _nk_angular_f64(a: number, b: number, n: number, result: number): void;
  _nk_sqeuclidean_f64(a: number, b: number, n: number, result: number): void;
  _nk_euclidean_f64(a: number, b: number, n: number, result: number): void;

  // Distance functions - f16 (store as u16, result in f32)
  _nk_dot_f16(a: number, b: number, n: number, result: number): void;
  _nk_angular_f16(a: number, b: number, n: number, result: number): void;
  _nk_sqeuclidean_f16(a: number, b: number, n: number, result: number): void;
  _nk_euclidean_f16(a: number, b: number, n: number, result: number): void;

  // Distance functions - bf16 (store as u16, result in f32)
  _nk_dot_bf16(a: number, b: number, n: number, result: number): void;
  _nk_angular_bf16(a: number, b: number, n: number, result: number): void;
  _nk_sqeuclidean_bf16(a: number, b: number, n: number, result: number): void;
  _nk_euclidean_bf16(a: number, b: number, n: number, result: number): void;

  // Distance functions - i8 (result in i32)
  _nk_dot_i8(a: number, b: number, n: number, result: number): void;
  _nk_angular_i8(a: number, b: number, n: number, result: number): void;
  _nk_sqeuclidean_i8(a: number, b: number, n: number, result: number): void;
  _nk_euclidean_i8(a: number, b: number, n: number, result: number): void;

  // Distance functions - u8 (result in u32)
  _nk_dot_u8(a: number, b: number, n: number, result: number): void;
  _nk_angular_u8(a: number, b: number, n: number, result: number): void;
  _nk_sqeuclidean_u8(a: number, b: number, n: number, result: number): void;
  _nk_euclidean_u8(a: number, b: number, n: number, result: number): void;

  // Binary distances
  _nk_hamming_u1(a: number, b: number, n: number, result: number): void;
  _nk_hamming_u8(a: number, b: number, n: number, result: number): void;
  _nk_jaccard_u1(a: number, b: number, n: number, result: number): void;
  _nk_jaccard_u16(a: number, b: number, n: number, result: number): void;

  // Probability divergences
  _nk_kld_f32(a: number, b: number, n: number, result: number): void;
  _nk_kld_f64(a: number, b: number, n: number, result: number): void;
  _nk_jsd_f32(a: number, b: number, n: number, result: number): void;
  _nk_jsd_f64(a: number, b: number, n: number, result: number): void;

  // Capabilities
  _nk_capabilities(): number;
}

/* #endregion Emscripten Interface */

/* #region Heap Management */

let Module: EmscriptenModule | null = null;

// Pre-allocated 8-byte result buffer (covers f64/f32/i32/u32), allocated once in initWasm()
let resultPtr: number = 0;

// Heap views (created from wasmMemory buffer)
let HEAP8: Int8Array;
let HEAP16: Int16Array;
let HEAP32: Int32Array;
let HEAPU8: Uint8Array;
let HEAPU16: Uint16Array;
let HEAPU32: Uint32Array;
let HEAPF32: Float32Array;
let HEAPF64: Float64Array;

/**
 * Initialize the WASM module
 */
export function initWasm(wasmModule: EmscriptenModule): void {
  Module = wasmModule;

  // Create heap views from the WASM memory buffer
  const buffer = wasmModule.wasmMemory.buffer;
  HEAP8 = new Int8Array(buffer);
  HEAP16 = new Int16Array(buffer);
  HEAP32 = new Int32Array(buffer);
  HEAPU8 = new Uint8Array(buffer);
  HEAPU16 = new Uint16Array(buffer);
  HEAPU32 = new Uint32Array(buffer);
  HEAPF32 = new Float32Array(buffer);
  HEAPF64 = new Float64Array(buffer);

  // Pre-allocate an 8-byte result buffer (never freed during module lifetime)
  resultPtr = wasmModule._malloc(8);
}

/**
 * Bytes per element for each dtype string.
 */
function bytesPerElement(dtype: string): number {
  switch (dtype) {
    case 'f64': return 8;
    case 'f32': case 'i32': case 'u32': return 4;
    case 'f16': case 'bf16': return 2;
    case 'i8': case 'u8': case 'u1': return 1;
    default: throw new Error(`Unknown dtype: ${dtype}`);
  }
}

/**
 * Returns a TypedArray view into WASM memory at the given pointer.
 */
function heapView(dtype: string, ptr: number, length: number): any {
  if (!Module) throw new Error('WASM module not initialized');
  const buf = Module.wasmMemory.buffer;
  switch (dtype) {
    case 'f64': return new Float64Array(buf, ptr, length);
    case 'f32': return new Float32Array(buf, ptr, length);
    case 'i8': return new Int8Array(buf, ptr, length);
    case 'u8': case 'u1': return new Uint8Array(buf, ptr, length);
    case 'f16': case 'bf16': return new Uint16Array(buf, ptr, length);
    default: throw new Error(`Unsupported dtype for heapView: ${dtype}`);
  }
}

/**
 * Allocates a TypedArray on the WASM heap. The returned view is backed by WASM memory,
 * so distance calls using it avoid any copy overhead (zero-copy path).
 *
 * **Warning**: The returned view may be invalidated by WASM memory growth. Do not cache
 * it across calls that might trigger growth (e.g. other `wasmAlloc` calls or WASM functions
 * that allocate memory internally).
 */
export function wasmAlloc(dtype: string, length: number): any {
  if (!Module) throw new Error('WASM module not initialized');
  const bpe = bytesPerElement(dtype);
  const ptr = Module._malloc(length * bpe);
  return heapView(dtype, ptr, length);
}

/**
 * Frees a TypedArray that was previously allocated with `wasmAlloc`.
 * Throws if the array does not reside on the WASM heap.
 */
export function wasmFree(arr: any): void {
  if (!Module) throw new Error('WASM module not initialized');
  if (arr.buffer !== Module.wasmMemory.buffer)
    throw new Error('Array not allocated on WASM heap');
  Module._free(arr.byteOffset);
}

/* #endregion Heap Management */

/* #region Type Detection */

/**
 * Type information for dispatching
 */
interface TypeInfo {
  dtype: string;
  bytesPerElement: number;
  heapView: 'HEAP8' | 'HEAP16' | 'HEAP32' | 'HEAPU8' | 'HEAPU16' | 'HEAPU32' | 'HEAPF32' | 'HEAPF64';
  resultType: 'f32' | 'f64' | 'i32' | 'u32';
}

/**
 * Detect dtype from TypedArray constructor
 */
function detectType(arr: any): TypeInfo {
  if (arr instanceof Float64Array) {
    return { dtype: 'f64', bytesPerElement: 8, heapView: 'HEAPF64', resultType: 'f64' };
  } else if (arr instanceof Float32Array) {
    return { dtype: 'f32', bytesPerElement: 4, heapView: 'HEAPF32', resultType: 'f32' };
  } else if (arr instanceof Int8Array) {
    return { dtype: 'i8', bytesPerElement: 1, heapView: 'HEAP8', resultType: 'i32' };
  } else if (arr instanceof Uint8Array) {
    return { dtype: 'u8', bytesPerElement: 1, heapView: 'HEAPU8', resultType: 'u32' };
  }

  // Check for custom typed arrays from dtypes.ts
  const constructorName = arr.constructor.name;

  if (constructorName === 'Float16Array') {
    return { dtype: 'f16', bytesPerElement: 2, heapView: 'HEAPU16', resultType: 'f32' };
  } else if (constructorName === 'BFloat16Array') {
    return { dtype: 'bf16', bytesPerElement: 2, heapView: 'HEAPU16', resultType: 'f32' };
  } else if (constructorName === 'E4M3Array') {
    throw new Error('E4M3 not yet supported in WASM backend');
  } else if (constructorName === 'E5M2Array') {
    throw new Error('E5M2 not yet supported in WASM backend');
  } else if (constructorName === 'BinaryArray') {
    return { dtype: 'u1', bytesPerElement: 1, heapView: 'HEAPU8', resultType: 'u32' };
  }

  throw new Error(`Unsupported array type: ${constructorName}`);
}

/* #endregion Type Detection */

/* #region Distance Helpers */

/**
 * Write TypedArray to WASM heap
 */
function writeArray(ptr: number, arr: any, typeInfo: TypeInfo): void {
  if (!Module) throw new Error('WASM module not initialized');

  const offset = ptr / typeInfo.bytesPerElement;

  // Use the global heap view created from wasmMemory
  switch (typeInfo.heapView) {
    case 'HEAP8': HEAP8.set(arr, offset); break;
    case 'HEAP16': HEAP16.set(arr, offset); break;
    case 'HEAP32': HEAP32.set(arr, offset); break;
    case 'HEAPU8': HEAPU8.set(arr, offset); break;
    case 'HEAPU16': HEAPU16.set(arr, offset); break;
    case 'HEAPU32': HEAPU32.set(arr, offset); break;
    case 'HEAPF32': HEAPF32.set(arr, offset); break;
    case 'HEAPF64': HEAPF64.set(arr, offset); break;
  }
}

/**
 * Allocate WASM memory and copy array data into it. Returns the pointer.
 */
function allocAndCopy(arr: any, typeInfo: TypeInfo): number {
  if (!Module) throw new Error('WASM module not initialized');
  const ptr = Module._malloc(arr.length * typeInfo.bytesPerElement);
  writeArray(ptr, arr, typeInfo);
  return ptr;
}

/**
 * Read result from WASM heap
 */
function readResult(ptr: number, resultType: 'f32' | 'f64' | 'i32' | 'u32'): number {
  if (!Module) throw new Error('WASM module not initialized');

  switch (resultType) {
    case 'f64':
      return HEAPF64[ptr / 8];
    case 'f32':
      return HEAPF32[ptr / 4];
    case 'i32':
      return HEAP32[ptr / 4];
    case 'u32':
      return HEAPU32[ptr / 4];
  }
}

/* #endregion Distance Helpers */

/* #region Distance Functions */

/**
 * Generic distance function wrapper.
 * Uses zero-copy when arrays already live on the WASM heap.
 */
function distance(metric: string, a: any, b: any): number {
  if (!Module) {
    throw new Error('WASM module not initialized. Call initWasm() first.');
  }

  if (a.length !== b.length) {
    throw new Error(`Array length mismatch: ${a.length} !== ${b.length}`);
  }

  const typeInfo = detectType(a);
  const n = a.length;

  // Zero-copy: if the array's buffer IS the WASM memory, its byteOffset is the pointer
  const onHeapA = a.buffer === Module.wasmMemory.buffer;
  const onHeapB = b.buffer === Module.wasmMemory.buffer;
  const aPtr = onHeapA ? a.byteOffset : allocAndCopy(a, typeInfo);
  const bPtr = onHeapB ? b.byteOffset : allocAndCopy(b, typeInfo);

  try {
    // Call C function
    const fnName = `_nk_${metric}_${typeInfo.dtype}` as keyof EmscriptenModule;
    const fn = Module[fnName] as any;

    if (!fn || typeof fn !== 'function') {
      throw new Error(`Function ${fnName} not available in WASM module`);
    }

    // WASM expects BigInt for size_t (64-bit); resultPtr is pre-allocated
    fn(aPtr, bPtr, BigInt(n), resultPtr);

    // Read result
    return readResult(resultPtr, typeInfo.resultType);
  } finally {
    // Only free memory that we allocated (not zero-copy pointers)
    if (!onHeapA) Module._free(aPtr);
    if (!onHeapB) Module._free(bPtr);
  }
}

/**
 * @brief Computes the squared Euclidean distance between two vectors.
 */
export function sqeuclidean(a: any, b: any): number {
  return distance('sqeuclidean', a, b);
}

/**
 * @brief Computes the Euclidean distance between two vectors.
 */
export function euclidean(a: any, b: any): number {
  return distance('euclidean', a, b);
}

/**
 * @brief Computes the angular distance between two vectors.
 */
export function angular(a: any, b: any): number {
  return distance('angular', a, b);
}

/**
 * @brief Computes the dot product of two vectors.
 */
export function dot(a: any, b: any): number {
  return distance('dot', a, b);
}

/**
 * @brief Alias for dot product.
 */
export const inner = dot;

/**
 * @brief Computes the bitwise Hamming distance between two vectors.
 *
 * Note: Following N-API behavior, always treats input as u1 (binary/bit-packed),
 * even if passed as Uint8Array. Each byte represents 8 bits.
 */
export function hamming(a: Uint8Array | any, b: Uint8Array | any): number {
  if (!Module) {
    throw new Error('WASM module not initialized');
  }

  if (a.length !== b.length) {
    throw new Error(`Array length mismatch: ${a.length} !== ${b.length}`);
  }

  // Always use u1 (binary) dtype to match N-API behavior
  const dtype = 'u1';
  const n = a.length;
  const bpe = 1;
  const typeInfo: TypeInfo = { dtype, bytesPerElement: bpe, heapView: 'HEAPU8', resultType: 'u32' };

  const onHeapA = a.buffer === Module.wasmMemory.buffer;
  const onHeapB = b.buffer === Module.wasmMemory.buffer;
  const aPtr = onHeapA ? a.byteOffset : allocAndCopy(a, typeInfo);
  const bPtr = onHeapB ? b.byteOffset : allocAndCopy(b, typeInfo);

  try {
    const fn = Module._nk_hamming_u1 as any;

    if (!fn || typeof fn !== 'function') {
      throw new Error('Function _nk_hamming_u1 not available in WASM module');
    }

    // WASM expects BigInt for size_t (64-bit)
    fn(aPtr, bPtr, BigInt(n), resultPtr);

    return readResult(resultPtr, 'u32');
  } finally {
    if (!onHeapA) Module._free(aPtr);
    if (!onHeapB) Module._free(bPtr);
  }
}

/**
 * @brief Computes the bitwise Jaccard distance between two vectors.
 *
 * Note: Following N-API behavior, always treats input as u1 (binary/bit-packed),
 * even if passed as Uint8Array. Each byte represents 8 bits.
 */
export function jaccard(a: Uint8Array | any, b: Uint8Array | any): number {
  if (!Module) {
    throw new Error('WASM module not initialized');
  }

  if (a.length !== b.length) {
    throw new Error(`Array length mismatch: ${a.length} !== ${b.length}`);
  }

  // Always use u1 (binary) dtype to match N-API behavior
  const dtype = 'u1';
  const n = a.length;
  const bpe = 1;
  const typeInfo: TypeInfo = { dtype, bytesPerElement: bpe, heapView: 'HEAPU8', resultType: 'f32' };

  const onHeapA = a.buffer === Module.wasmMemory.buffer;
  const onHeapB = b.buffer === Module.wasmMemory.buffer;
  const aPtr = onHeapA ? a.byteOffset : allocAndCopy(a, typeInfo);
  const bPtr = onHeapB ? b.byteOffset : allocAndCopy(b, typeInfo);

  try {
    const fn = Module._nk_jaccard_u1 as any;

    if (!fn || typeof fn !== 'function') {
      throw new Error('Function _nk_jaccard_u1 not available in WASM module');
    }

    // WASM expects BigInt for size_t (64-bit)
    fn(aPtr, bPtr, BigInt(n), resultPtr);

    return readResult(resultPtr, 'f32');
  } finally {
    if (!onHeapA) Module._free(aPtr);
    if (!onHeapB) Module._free(bPtr);
  }
}

/**
 * @brief Computes the Kullback-Leibler divergence between two vectors.
 */
export function kullbackleibler(a: Float64Array | Float32Array, b: Float64Array | Float32Array): number {
  if (!Module) {
    throw new Error('WASM module not initialized');
  }

  if (a.length !== b.length) {
    throw new Error(`Array length mismatch: ${a.length} !== ${b.length}`);
  }

  const isF64 = a instanceof Float64Array;
  const dtype = isF64 ? 'f64' : 'f32';
  const bpe = isF64 ? 8 : 4;
  const heapView = isF64 ? 'HEAPF64' : 'HEAPF32';
  const resultType = isF64 ? 'f64' : 'f32';
  const typeInfo: TypeInfo = { dtype, bytesPerElement: bpe, heapView: heapView as any, resultType };

  const n = a.length;
  const onHeapA = a.buffer === Module.wasmMemory.buffer;
  const onHeapB = b.buffer === Module.wasmMemory.buffer;
  const aPtr = onHeapA ? a.byteOffset : allocAndCopy(a, typeInfo);
  const bPtr = onHeapB ? b.byteOffset : allocAndCopy(b, typeInfo);

  try {
    const fnName = `_nk_kld_${dtype}` as keyof EmscriptenModule;
    const fn = Module[fnName] as any;

    if (!fn || typeof fn !== 'function') {
      throw new Error(`Function ${fnName} not available in WASM module`);
    }

    // WASM expects BigInt for size_t (64-bit)
    fn(aPtr, bPtr, BigInt(n), resultPtr);

    return readResult(resultPtr, resultType);
  } finally {
    if (!onHeapA) Module._free(aPtr);
    if (!onHeapB) Module._free(bPtr);
  }
}

/**
 * @brief Computes the Jensen-Shannon divergence between two vectors.
 */
export function jensenshannon(a: Float64Array | Float32Array, b: Float64Array | Float32Array): number {
  if (!Module) {
    throw new Error('WASM module not initialized');
  }

  if (a.length !== b.length) {
    throw new Error(`Array length mismatch: ${a.length} !== ${b.length}`);
  }

  const isF64 = a instanceof Float64Array;
  const dtype = isF64 ? 'f64' : 'f32';
  const bpe = isF64 ? 8 : 4;
  const heapView = isF64 ? 'HEAPF64' : 'HEAPF32';
  const resultType = isF64 ? 'f64' : 'f32';
  const typeInfo: TypeInfo = { dtype, bytesPerElement: bpe, heapView: heapView as any, resultType };

  const n = a.length;
  const onHeapA = a.buffer === Module.wasmMemory.buffer;
  const onHeapB = b.buffer === Module.wasmMemory.buffer;
  const aPtr = onHeapA ? a.byteOffset : allocAndCopy(a, typeInfo);
  const bPtr = onHeapB ? b.byteOffset : allocAndCopy(b, typeInfo);

  try {
    const fnName = `_nk_jsd_${dtype}` as keyof EmscriptenModule;
    const fn = Module[fnName] as any;

    if (!fn || typeof fn !== 'function') {
      throw new Error(`Function ${fnName} not available in WASM module`);
    }

    // WASM expects BigInt for size_t (64-bit)
    fn(aPtr, bPtr, BigInt(n), resultPtr);

    return readResult(resultPtr, resultType);
  } finally {
    if (!onHeapA) Module._free(aPtr);
    if (!onHeapB) Module._free(bPtr);
  }
}

/* #endregion Distance Functions */

/* #region Capabilities */

/**
 * @brief Returns the runtime-detected SIMD capabilities as a bitmask.
 */
export function getCapabilities(): bigint {
  if (!Module) {
    throw new Error('WASM module not initialized');
  }

  // nk_capabilities returns a 64-bit value
  // In WASM/JS, we need to handle this as two 32-bit parts or use BigInt
  const caps = Module._nk_capabilities();

  // For now, return as BigInt (capability_t is nk_u64_t)
  return BigInt(caps);
}

/**
 * @brief Checks if a specific capability is available at runtime.
 */
export function hasCapability(cap: bigint): boolean {
  return (getCapabilities() & cap) !== 0n;
}

/* #endregion Capabilities */
