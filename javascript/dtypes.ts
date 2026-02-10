/**
 * @brief Custom TypedArray classes for non-native numeric types.
 * @file javascript/dtypes.ts
 * @author Ash Vardanian
 * @date February 3, 2026
 *
 * This file provides TypedArray wrappers for numeric types not natively supported
 * by JavaScript, using NumKong's SIMD-optimized conversion functions from the C library.
 */

// Import conversion functions from the compiled native module
// These will be defined in numkong.ts after the module is loaded
let conversionFunctions: {
  castF16ToF32: (bits: number) => number;
  castF32ToF16: (value: number) => number;
  castBF16ToF32: (bits: number) => number;
  castF32ToBF16: (value: number) => number;
  castE4M3ToF32: (bits: number) => number;
  castF32ToE4M3: (value: number) => number;
  castE5M2ToF32: (bits: number) => number;
  castF32ToE5M2: (value: number) => number;
  cast: (src: TypedArray, srcType: string, dst: TypedArray, dstType: string) => void;
};

// This will be called by numkong.ts after loading the module
export function setConversionFunctions(fns: typeof conversionFunctions) {
  conversionFunctions = fns;
}

// Type alias for any TypedArray
type TypedArray = Float64Array | Float32Array | Int8Array | Uint8Array | Uint16Array | Uint32Array;

/**
 * @brief IEEE 754 Half Precision Float (f16)
 *
 * 16-bit floating point: 1 sign bit, 5 exponent bits, 10 mantissa bits
 * Range: ~±65504, precision: ~3-4 decimal digits
 *
 * Common in GPU inference, model compression, and mixed-precision training.
 * Supported natively on Apple Silicon, NVIDIA GPUs (fp16), AMD GPUs.
 */
export class Float16Array extends Uint16Array {
  constructor(length: number | ArrayLike<number> | ArrayBufferLike, byteOffset?: number, arrayLength?: number) {
    if (typeof length === 'number') {
      super(length);
    } else if (ArrayBuffer.isView(length) || length instanceof ArrayBuffer) {
      super(length as any, byteOffset, arrayLength);
    } else {
      // Convert from array-like of numbers
      const src = length as ArrayLike<number>;
      const arr = new Uint16Array(src.length);
      if (conversionFunctions) {
        for (let i = 0; i < src.length; i++) {
          arr[i] = conversionFunctions.castF32ToF16(src[i]);
        }
      }
      super(arr);
    }
  }

  /**
   * @brief Converts the entire f16 array to f32 (Float32Array).
   * @returns Float32Array with decoded values
   */
  toFloat32Array(): Float32Array {
    if (!conversionFunctions) {
      throw new Error('Conversion functions not initialized');
    }
    const result = new Float32Array(this.length);
    for (let i = 0; i < this.length; i++) {
      result[i] = conversionFunctions.castF16ToF32(this[i]);
    }
    return result;
  }

  /**
   * @brief Gets the f32 value at the specified index.
   * @param index Array index
   * @returns Decoded f32 value
   */
  getFloat32(index: number): number {
    if (!conversionFunctions) {
      throw new Error('Conversion functions not initialized');
    }
    return conversionFunctions.castF16ToF32(this[index]);
  }

  /**
   * @brief Sets the value at the specified index from an f32 value.
   * @param index Array index
   * @param value f32 value to encode and store
   */
  setFloat32(index: number, value: number): void {
    if (!conversionFunctions) {
      throw new Error('Conversion functions not initialized');
    }
    this[index] = conversionFunctions.castF32ToF16(value);
  }
}

/**
 * @brief Brain Float 16 (bf16)
 *
 * 16-bit floating point: 1 sign bit, 8 exponent bits, 7 mantissa bits
 * Range: same as f32 (~±3.4e38), precision: ~2-3 decimal digits
 *
 * Designed by Google for TPUs, optimized for ML training (wider range than f16).
 * Supported on Google TPUs, Intel Sapphire Rapids, AMD Genoa, ARM Neoverse V2.
 * Truncated f32 (top 16 bits), making conversion very cheap.
 */
export class BFloat16Array extends Uint16Array {
  constructor(length: number | ArrayLike<number> | ArrayBufferLike, byteOffset?: number, arrayLength?: number) {
    if (typeof length === 'number') {
      super(length);
    } else if (ArrayBuffer.isView(length) || length instanceof ArrayBuffer) {
      super(length as any, byteOffset, arrayLength);
    } else {
      const src = length as ArrayLike<number>;
      const arr = new Uint16Array(src.length);
      if (conversionFunctions) {
        for (let i = 0; i < src.length; i++) {
          arr[i] = conversionFunctions.castF32ToBF16(src[i]);
        }
      }
      super(arr);
    }
  }

  toFloat32Array(): Float32Array {
    if (!conversionFunctions) {
      throw new Error('Conversion functions not initialized');
    }
    const result = new Float32Array(this.length);
    for (let i = 0; i < this.length; i++) {
      result[i] = conversionFunctions.castBF16ToF32(this[i]);
    }
    return result;
  }

  getFloat32(index: number): number {
    if (!conversionFunctions) {
      throw new Error('Conversion functions not initialized');
    }
    return conversionFunctions.castBF16ToF32(this[index]);
  }

  setFloat32(index: number, value: number): void {
    if (!conversionFunctions) {
      throw new Error('Conversion functions not initialized');
    }
    this[index] = conversionFunctions.castF32ToBF16(value);
  }
}

/**
 * @brief FP8 E4M3 (4-bit exponent, 3-bit mantissa)
 *
 * 8-bit floating point: 1 sign bit, 4 exponent bits, 3 mantissa bits
 * Range: ~±448, precision: ~1 decimal digit
 *
 * Optimized for forward pass inference with higher precision than E5M2.
 * Supported on NVIDIA Hopper H100 GPUs, AMD Instinct MI300.
 */
export class E4M3Array extends Uint8Array {
  constructor(length: number | ArrayLike<number> | ArrayBufferLike, byteOffset?: number, arrayLength?: number) {
    if (typeof length === 'number') {
      super(length);
    } else if (ArrayBuffer.isView(length) || length instanceof ArrayBuffer) {
      super(length as any, byteOffset, arrayLength);
    } else {
      const src = length as ArrayLike<number>;
      const arr = new Uint8Array(src.length);
      if (conversionFunctions) {
        for (let i = 0; i < src.length; i++) {
          arr[i] = conversionFunctions.castF32ToE4M3(src[i]);
        }
      }
      super(arr);
    }
  }

  toFloat32Array(): Float32Array {
    if (!conversionFunctions) {
      throw new Error('Conversion functions not initialized');
    }
    const result = new Float32Array(this.length);
    for (let i = 0; i < this.length; i++) {
      result[i] = conversionFunctions.castE4M3ToF32(this[i]);
    }
    return result;
  }

  getFloat32(index: number): number {
    if (!conversionFunctions) {
      throw new Error('Conversion functions not initialized');
    }
    return conversionFunctions.castE4M3ToF32(this[index]);
  }

  setFloat32(index: number, value: number): void {
    if (!conversionFunctions) {
      throw new Error('Conversion functions not initialized');
    }
    this[index] = conversionFunctions.castF32ToE4M3(value);
  }
}

/**
 * @brief FP8 E5M2 (5-bit exponent, 2-bit mantissa)
 *
 * 8-bit floating point: 1 sign bit, 5 exponent bits, 2 mantissa bits
 * Range: ~±57344, precision: <1 decimal digit
 *
 * Optimized for backward pass training with wider range than E4M3.
 * Supported on NVIDIA Hopper H100 GPUs, AMD Instinct MI300.
 */
export class E5M2Array extends Uint8Array {
  constructor(length: number | ArrayLike<number> | ArrayBufferLike, byteOffset?: number, arrayLength?: number) {
    if (typeof length === 'number') {
      super(length);
    } else if (ArrayBuffer.isView(length) || length instanceof ArrayBuffer) {
      super(length as any, byteOffset, arrayLength);
    } else {
      const src = length as ArrayLike<number>;
      const arr = new Uint8Array(src.length);
      if (conversionFunctions) {
        for (let i = 0; i < src.length; i++) {
          arr[i] = conversionFunctions.castF32ToE5M2(src[i]);
        }
      }
      super(arr);
    }
  }

  toFloat32Array(): Float32Array {
    if (!conversionFunctions) {
      throw new Error('Conversion functions not initialized');
    }
    const result = new Float32Array(this.length);
    for (let i = 0; i < this.length; i++) {
      result[i] = conversionFunctions.castE5M2ToF32(this[i]);
    }
    return result;
  }

  getFloat32(index: number): number {
    if (!conversionFunctions) {
      throw new Error('Conversion functions not initialized');
    }
    return conversionFunctions.castE5M2ToF32(this[index]);
  }

  setFloat32(index: number, value: number): void {
    if (!conversionFunctions) {
      throw new Error('Conversion functions not initialized');
    }
    this[index] = conversionFunctions.castF32ToE5M2(value);
  }
}

/**
 * @brief Binary Array (u1) - Bit-packed binary vectors
 *
 * 1-bit per element, packed into bytes (8 bits per byte)
 * Used for binary embeddings, hashing, and Hamming/Jaccard distances.
 *
 * Common in semantic search with binary quantization (Cohere, Voyage).
 */
export class BinaryArray extends Uint8Array {
  private _bitLength: number;

  constructor(bitLength: number) {
    const byteLength = Math.ceil(bitLength / 8);
    super(byteLength);
    this._bitLength = bitLength;
  }

  /**
   * @brief Gets the bit value at the specified index.
   * @param index Bit index (0 to bitLength-1)
   * @returns 0 or 1
   */
  getBit(index: number): number {
    if (index < 0 || index >= this._bitLength) {
      throw new RangeError('Index out of bounds');
    }
    const byteIndex = index >>> 3;  // index / 8
    const bitIndex = index & 7;     // index % 8
    return (this[byteIndex] >>> bitIndex) & 1;
  }

  /**
   * @brief Sets the bit value at the specified index.
   * @param index Bit index (0 to bitLength-1)
   * @param value 0 or 1
   */
  setBit(index: number, value: number): void {
    if (index < 0 || index >= this._bitLength) {
      throw new RangeError('Index out of bounds');
    }
    const byteIndex = index >>> 3;
    const bitIndex = index & 7;
    if (value) {
      this[byteIndex] |= (1 << bitIndex);
    } else {
      this[byteIndex] &= ~(1 << bitIndex);
    }
  }

  /**
   * @brief Returns the logical bit length of the array.
   */
  get bitLength(): number {
    return this._bitLength;
  }

  /**
   * @brief Creates a BinaryArray from a Float32Array (positive values = 1, else 0).
   * @param vector Source floating-point vector
   * @returns Binary array with quantized values
   */
  static fromFloat32Array(vector: Float32Array): BinaryArray {
    const binary = new BinaryArray(vector.length);
    for (let i = 0; i < vector.length; i++) {
      if (vector[i] > 0) {
        binary.setBit(i, 1);
      }
    }
    return binary;
  }

  /**
   * @brief Creates a BinaryArray from a Float64Array (positive values = 1, else 0).
   * @param vector Source floating-point vector
   * @returns Binary array with quantized values
   */
  static fromFloat64Array(vector: Float64Array): BinaryArray {
    const binary = new BinaryArray(vector.length);
    for (let i = 0; i < vector.length; i++) {
      if (vector[i] > 0) {
        binary.setBit(i, 1);
      }
    }
    return binary;
  }
}

/**
 * @brief Type guard to check if an object is a Float16Array.
 */
export function isFloat16Array(obj: any): obj is Float16Array {
  return obj instanceof Float16Array;
}

/**
 * @brief Type guard to check if an object is a BFloat16Array.
 */
export function isBFloat16Array(obj: any): obj is BFloat16Array {
  return obj instanceof BFloat16Array;
}

/**
 * @brief Type guard to check if an object is an E4M3Array.
 */
export function isE4M3Array(obj: any): obj is E4M3Array {
  return obj instanceof E4M3Array;
}

/**
 * @brief Type guard to check if an object is an E5M2Array.
 */
export function isE5M2Array(obj: any): obj is E5M2Array {
  return obj instanceof E5M2Array;
}

/**
 * @brief Type guard to check if an object is a BinaryArray.
 */
export function isBinaryArray(obj: any): obj is BinaryArray {
  return obj instanceof BinaryArray;
}
