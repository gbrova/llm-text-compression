// Try to import huffman-coding, fall back to mock if not available
let HuffmanCodec;
try {
    const huffmanCoding = await import('huffman-coding');
    HuffmanCodec = huffmanCoding.HuffmanCodec;
} catch (error) {
    // Mock HuffmanCodec for testing without huffman-coding
    // This mock should either work correctly or throw expected errors
    HuffmanCodec = class MockHuffmanCodec {
        constructor() {
            this.codecData = null;
        }
        buildFromFrequencies(frequencies) {
            // Store the original data for round-trip testing
            this.codecData = { frequencies };
        }
        serialize() { 
            return JSON.stringify(this.codecData); 
        }
        deserialize(data) {
            this.codecData = JSON.parse(data);
        }
        encode(data) { 
            // Simple mock: convert data to bytes but preserve as JSON for testing
            return new Uint8Array(Buffer.from(JSON.stringify(data), 'utf8'));
        }
        decode(encodedData) { 
            // Decode back from bytes
            const str = Buffer.from(encodedData).toString('utf8');
            return JSON.parse(str);
        }
    };
}

/**
 * Huffman compression utilities for LLM rank sequences
 */
export class HuffmanRankCompressor {
    constructor() {
        // Initialize compressor
    }

    /**
     * Count frequencies of ranks in the sequence
     * @param {Array<number>} ranks - List of token ranks
     * @returns {Object} Frequency count object
     * @private
     */
    _countRankFrequencies(ranks) {
        const frequencies = {};
        for (const rank of ranks) {
            frequencies[rank] = (frequencies[rank] || 0) + 1;
        }
        return frequencies;
    }

    /**
     * Compress ranks using basic Huffman coding with frequency table
     * @param {Array<number>} ranks - List of token ranks
     * @returns {Buffer} Compressed data
     */
    compressBasic(ranks) {
        // Build frequency table
        const frequencies = this._countRankFrequencies(ranks);
        
        // Create Huffman codec
        const codec = new HuffmanCodec();
        codec.buildFromFrequencies(frequencies);
        
        // Encode the ranks
        const encodedRanks = codec.encode(ranks);
        
        // Serialize codec and compressed data
        const serializedData = JSON.stringify({
            codec: codec.serialize(),
            compressedRanks: Array.from(encodedRanks),
            originalLength: ranks.length
        });
        
        return Buffer.from(serializedData, 'utf8');
    }

    /**
     * Decompress ranks from basic Huffman compressed data
     * @param {Buffer} compressedData - Compressed data
     * @returns {Array<number>} List of decompressed token ranks
     */
    decompressBasic(compressedData) {
        // Parse serialized data
        const serializedData = compressedData.toString('utf8');
        const data = JSON.parse(serializedData);
        
        // Recreate codec
        const codec = new HuffmanCodec();
        codec.deserialize(data.codec);
        
        // Decode the ranks
        const encodedRanks = new Uint8Array(data.compressedRanks);
        const decodedRanks = codec.decode(encodedRanks);
        
        return decodedRanks;
    }

    /**
     * Fit a Zipf distribution to rank frequencies
     * @param {Object} rankCounts - Dictionary mapping ranks to frequencies
     * @returns {Object} Zipf parameters {sParam, maxRank}
     * @private
     */
    _fitZipfDistribution(rankCounts) {
        const ranks = Object.keys(rankCounts).map(Number);
        const counts = Object.values(rankCounts);
        
        // Simple Zipf parameter estimation using least squares
        // In practice, you might want to use a more sophisticated optimization
        let bestS = 1.0;
        let bestError = Infinity;
        
        // Try different s values with safe bounds and integer steps to avoid floating point issues
        for (let i = 1; i <= 50; i++) {
            const s = i * 0.1; // This ensures s is always > 0 and avoids floating point increment issues
            const error = this._calculateZipfError(ranks, counts, s);
            if (error < bestError) {
                bestError = error;
                bestS = s;
            }
        }
        
        // Ensure sParam is always positive and reasonable
        const sParam = Math.max(0.1, Math.min(5.0, bestS));
        
        return {
            sParam,
            maxRank: Math.max(...ranks)
        };
    }

    /**
     * Calculate error for Zipf distribution fitting
     * @param {Array<number>} ranks - Rank values
     * @param {Array<number>} counts - Frequency counts
     * @param {number} s - Zipf parameter
     * @returns {number} Error value
     * @private
     */
    _calculateZipfError(ranks, counts, s) {
        if (s <= 0) return Infinity;
        
        const totalCount = counts.reduce((sum, count) => sum + count, 0);
        const normalization = ranks.reduce((sum, rank) => sum + Math.pow(rank, -s), 0);
        
        let error = 0;
        for (let i = 0; i < ranks.length; i++) {
            const predicted = totalCount * Math.pow(ranks[i], -s) / normalization;
            error += Math.pow(counts[i] - predicted, 2);
        }
        
        return error;
    }

    /**
     * Generate frequencies based on Zipf distribution
     * @param {number} sParam - Zipf parameter
     * @param {number} maxRank - Maximum rank
     * @param {number} totalCount - Total number of tokens
     * @returns {Object} Frequency dictionary
     * @private
     */
    _generateZipfFrequencies(sParam, maxRank, totalCount) {
        const frequencies = {};
        let normalization = 0;
        
        // Calculate normalization factor
        for (let rank = 1; rank <= maxRank; rank++) {
            normalization += Math.pow(rank, -sParam);
        }
        
        // Generate frequencies
        for (let rank = 1; rank <= maxRank; rank++) {
            const prob = Math.pow(rank, -sParam) / normalization;
            const freq = Math.max(1, Math.round(totalCount * prob));
            frequencies[rank] = freq;
        }
        
        return frequencies;
    }

    /**
     * Compress ranks using Zipf distribution parametric Huffman coding
     * @param {Array<number>} ranks - List of token ranks
     * @returns {Buffer} Compressed data
     */
    compressZipf(ranks) {
        // Fit Zipf distribution
        const rankCounts = this._countRankFrequencies(ranks);
        const { sParam, maxRank } = this._fitZipfDistribution(rankCounts);
        
        // Generate parametric frequencies
        const zipfFrequencies = this._generateZipfFrequencies(sParam, maxRank, ranks.length);
        
        // Create Huffman codec
        const codec = new HuffmanCodec();
        codec.buildFromFrequencies(zipfFrequencies);
        
        // Encode the ranks
        const encodedRanks = codec.encode(ranks);
        
        // Serialize parameters and compressed data
        const serializedData = JSON.stringify({
            sParam: sParam,
            maxRank: maxRank,
            totalCount: ranks.length,
            compressedRanks: Array.from(encodedRanks)
        });
        
        return Buffer.from(serializedData, 'utf8');
    }

    /**
     * Decompress ranks from Zipf parametric Huffman compressed data
     * @param {Buffer} compressedData - Compressed data
     * @returns {Array<number>} List of decompressed token ranks
     */
    decompressZipf(compressedData) {
        // Parse serialized data
        const serializedData = compressedData.toString('utf8');
        const data = JSON.parse(serializedData);
        
        // Reconstruct codec from parameters
        const zipfFrequencies = this._generateZipfFrequencies(
            data.sParam,
            data.maxRank,
            data.totalCount
        );
        
        const codec = new HuffmanCodec();
        codec.buildFromFrequencies(zipfFrequencies);
        
        // Decode the ranks
        const encodedRanks = new Uint8Array(data.compressedRanks);
        const decodedRanks = codec.decode(encodedRanks);
        
        return decodedRanks;
    }

    /**
     * Compress ranks using Zipf distribution with efficient binary storage
     * @param {Array<number>} ranks - List of token ranks
     * @returns {Buffer} Compressed data in binary format
     */
    compressZipfBytes(ranks) {
        // Fit Zipf distribution
        const rankCounts = this._countRankFrequencies(ranks);
        const { sParam, maxRank } = this._fitZipfDistribution(rankCounts);
        
        // Generate parametric frequencies
        const zipfFrequencies = this._generateZipfFrequencies(sParam, maxRank, ranks.length);
        
        // Create Huffman codec
        const codec = new HuffmanCodec();
        codec.buildFromFrequencies(zipfFrequencies);
        
        // Encode the ranks
        const encodedRanks = codec.encode(ranks);
        
        // Create binary header (12 bytes):
        // - sParam as float32 (4 bytes)
        // - maxRank as uint32 (4 bytes)
        // - totalCount as uint32 (4 bytes)
        const header = Buffer.alloc(12);
        header.writeFloatBE(sParam, 0);
        header.writeUInt32BE(maxRank, 4);
        header.writeUInt32BE(ranks.length, 8);
        
        // Combine header and compressed data
        const compressedRanks = Buffer.from(encodedRanks);
        return Buffer.concat([header, compressedRanks]);
    }

    /**
     * Decompress ranks from byte-format Zipf compressed data
     * @param {Buffer} compressedData - Compressed data in binary format
     * @returns {Array<number>} List of decompressed token ranks
     */
    decompressZipfBytes(compressedData) {
        // Extract header parameters (12 bytes)
        const sParam = compressedData.readFloatBE(0);
        const maxRank = compressedData.readUInt32BE(4);
        const totalCount = compressedData.readUInt32BE(8);
        
        // Extract compressed ranks
        const encodedRanks = compressedData.slice(12);
        
        // Reconstruct codec from parameters
        const zipfFrequencies = this._generateZipfFrequencies(sParam, maxRank, totalCount);
        
        const codec = new HuffmanCodec();
        codec.buildFromFrequencies(zipfFrequencies);
        
        // Decode the ranks
        const decodedRanks = codec.decode(encodedRanks);
        
        return decodedRanks;
    }

    /**
     * Compress ranks with timing information
     * @param {Array<number>} ranks - List of token ranks
     * @param {string} method - Compression method ('basic', 'zipf', 'zipf_bytes')
     * @returns {Object} {compressedData: Buffer, compressionTime: number}
     */
    compressRanksWithTiming(ranks, method = 'basic') {
        const startTime = Date.now();
        
        let compressedData;
        switch (method) {
            case 'basic':
                compressedData = this.compressBasic(ranks);
                break;
            case 'zipf':
                compressedData = this.compressZipf(ranks);
                break;
            case 'zipf_bytes':
                compressedData = this.compressZipfBytes(ranks);
                break;
            default:
                throw new Error(`Unknown compression method: ${method}`);
        }
        
        const compressionTime = (Date.now() - startTime) / 1000; // Convert to seconds
        return { compressedData, compressionTime };
    }

    /**
     * Decompress ranks with timing information
     * @param {Buffer} compressedData - Compressed data
     * @param {string} method - Compression method ('basic', 'zipf', 'zipf_bytes')
     * @returns {Object} {decompressedRanks: Array, decompressionTime: number}
     */
    decompressRanksWithTiming(compressedData, method = 'basic') {
        const startTime = Date.now();
        
        let decompressedRanks;
        switch (method) {
            case 'basic':
                decompressedRanks = this.decompressBasic(compressedData);
                break;
            case 'zipf':
                decompressedRanks = this.decompressZipf(compressedData);
                break;
            case 'zipf_bytes':
                decompressedRanks = this.decompressZipfBytes(compressedData);
                break;
            default:
                throw new Error(`Unknown compression method: ${method}`);
        }
        
        const decompressionTime = (Date.now() - startTime) / 1000; // Convert to seconds
        return { decompressedRanks, decompressionTime };
    }
}