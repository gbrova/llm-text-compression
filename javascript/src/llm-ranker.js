// Import transformers.js - this is now required for real functionality
const transformers = await import('@huggingface/transformers');
const { AutoModel, AutoTokenizer, env } = transformers;

// Configure transformers.js to allow remote model downloads
env.allowLocalModels = true;
env.allowRemoteModels = true;

/**
 * Enum for different processing modes
 */
export const ProcessingMode = {
    RANK: 'rank',
    GENERATE: 'generate'
};

/**
 * LLM-based token ranker that computes ranks for text compression
 */
export class LLMRanker {
    /**
     * Initialize the LLM ranker
     * @param {Object} options - Configuration options
     * @param {string} options.modelName - HuggingFace model name (default: "gpt2")
     * @param {number} options.maxContextLength - Maximum context length
     * @param {number} options.batchSize - Number of parallel batches (default: 1)
     * @param {boolean} options.useCache - Whether to use KV caching for efficiency (default: true)
     */
    constructor(options = {}) {
        this.modelName = options.modelName || 'Xenova/distilgpt2';
        this.maxContextLength = options.maxContextLength || 1024;
        this.batchSize = options.batchSize || 1;
        this.useCache = options.useCache !== undefined ? options.useCache : true;
        
        // Initialize model and tokenizer directly (not pipeline)
        this.model = null;
        this.tokenizer = null;
        this._initialized = false;
    }

    /**
     * Initialize the model and tokenizer
     * @private
     */
    async _initialize() {
        if (this._initialized) return;
        
        try {
            // Load model and tokenizer directly for logit access
            this.tokenizer = await AutoTokenizer.from_pretrained(this.modelName);
            this.model = await AutoModel.from_pretrained(this.modelName);
            this._initialized = true;
        } catch (error) {
            throw new Error(`Failed to initialize model ${this.modelName}: ${error.message}`);
        }
    }

    /**
     * Get the vocabulary size of the model
     * @returns {number} Vocabulary size
     */
    getVocabSize() {
        if (!this._initialized) {
            throw new Error('Model not initialized. Call initialize() first.');
        }
        return this.tokenizer.vocab_size || 50257;
    }

    /**
     * Get the maximum context length
     * @returns {number} Maximum context length
     */
    getContextLength() {
        return this.maxContextLength;
    }

    /**
     * Get a cache signature for this ranker configuration
     * @returns {Array} Cache signature
     */
    getCacheSignature() {
        return [
            this.modelName,
            this.maxContextLength,
            this.batchSize,
            this.useCache
        ];
    }

    /**
     * Tokenize text input
     * @param {string} text - Input text
     * @returns {Array} Token IDs
     */
    async tokenize(text) {
        await this._initialize();
        
        try {
            const encoded = await this.tokenizer(text, {
                return_tensors: false,
                padding: false,
                truncation: false
            });
            
            // Extract token IDs from the encoded result
            let tokenIds;
            
            if (encoded && encoded.input_ids) {
                // Handle tensor format - convert to regular numbers
                if (encoded.input_ids.data) {
                    // Tensor object with data array
                    tokenIds = Array.from(encoded.input_ids.data).map(id => Number(id));
                } else if (Array.isArray(encoded.input_ids)) {
                    // Regular array
                    tokenIds = encoded.input_ids.map(id => Number(id));
                } else {
                    // Handle BigInt arrays or other formats
                    tokenIds = Array.from(encoded.input_ids).map(id => Number(id));
                }
            } else if (Array.isArray(encoded)) {
                tokenIds = encoded.map(id => Number(id));
            } else {
                throw new Error('Unexpected encoded format');
            }
            
            // Validate token IDs
            if (!Array.isArray(tokenIds) || tokenIds.length === 0) {
                throw new Error('No valid token IDs found');
            }
            
            return tokenIds;
        } catch (error) {
            throw new Error(`Tokenization failed: ${error.message}`);
        }
    }

    /**
     * Decode token IDs to text
     * @param {Array} tokenIds - Token IDs
     * @returns {string} Decoded text
     */
    async decode(tokenIds) {
        await this._initialize();
        
        try {
            // Ensure tokenIds is a valid array of numbers
            if (!Array.isArray(tokenIds)) {
                throw new Error('tokenIds must be an array');
            }
            
            if (tokenIds.length === 0) {
                return '';
            }
            
            // Convert to regular numbers if needed
            const normalizedTokenIds = tokenIds.map(id => Number(id));
            
            // Validate all IDs are valid numbers
            if (normalizedTokenIds.some(id => isNaN(id) || id < 0)) {
                throw new Error('Invalid token IDs detected');
            }
            
            if (this.tokenizer && this.tokenizer.decode) {
                return this.tokenizer.decode(normalizedTokenIds, { skip_special_tokens: true });
            } else {
                throw new Error('Tokenizer decode method not available');
            }
        } catch (error) {
            throw new Error(`Decoding failed: ${error.message}`);
        }
    }

    /**
     * Truncate KV cache to respect max context length
     * @param {Object} pastKeyValues - The past key values from the model
     * @returns {Object} Truncated past key values or null if input is null
     * @private
     */
    _truncateKvCache(pastKeyValues) {
        if (!pastKeyValues || !this.useCache) {
            return pastKeyValues;
        }

        // For transformers.js, the cache structure may be different from PyTorch
        // We'll implement a simple approach: if the cache gets too large, 
        // we'll reset it to prevent memory issues
        // In a full implementation, this would do proper tensor truncation
        
        // Simple heuristic: for now, we'll let transformers.js handle cache management
        // internally, and only reset if we detect the cache is unusually large
        // This is a conservative approach to prevent memory issues
        try {
            // Check if the cache has a structure that indicates it's getting large
            if (pastKeyValues && typeof pastKeyValues === 'object') {
                // For now, we'll trust transformers.js to handle the cache properly
                // and only reset in extreme cases
                return pastKeyValues;
            }
        } catch (error) {
            // If there's any error accessing the cache, reset it
            return null;
        }
        
        return pastKeyValues;
    }

    /**
     * Get the rank of each token in the sequence given previous context
     * @param {string} text - Input text sequence
     * @returns {Array<number>} List of ranks for each token (1-indexed)
     */
    async getTokenRanks(text) {
        await this._initialize();
        
        if (!text || text.length === 0) {
            return [];
        }

        // Tokenize the input
        const tokens = await this.tokenize(text);
        
        if (tokens.length === 0) {
            return [];
        }

        // Use appropriate processing method based on batch size
        if (this.batchSize === 1) {
            return await this._getTokenRanksSequential(tokens);
        } else {
            return await this._getTokenRanksParallel(tokens);
        }
    }

    /**
     * Sequential implementation of token ranking with KV caching
     * @param {Array} tokens - Token IDs
     * @returns {Array<number>} List of ranks
     * @private
     */
    async _getTokenRanksSequential(tokens) {
        const ranks = [];
        let pastKeyValues = null;
        
        for (let i = 0; i < tokens.length; i++) {
            let predictions;
            
            if (this.useCache && i > 0) {
                // Use cached context with incremental token
                const previousToken = tokens[i - 1];
                predictions = await this._getModelPredictionsIncremental(previousToken, pastKeyValues);
                pastKeyValues = predictions.pastKeyValues;
            } else {
                // For first token or when not using cache, use full context
                const contextTokens = tokens.slice(0, i);
                const contextText = await this.decode(contextTokens);
                predictions = await this._getModelPredictions(contextText, true);
                if (this.useCache) {
                    pastKeyValues = predictions.pastKeyValues;
                }
            }
            
            // Find rank of actual token
            const actualToken = tokens[i];
            const rank = this._findTokenRank(actualToken, predictions.predictions || predictions);
            ranks.push(rank);
            
            // Truncate cache if needed
            if (this.useCache && pastKeyValues) {
                pastKeyValues = this._truncateKvCache(pastKeyValues);
            }
        }
        
        return ranks;
    }

    /**
     * Parallel implementation of token ranking (simplified for JS)
     * @param {Array} tokens - Token IDs
     * @returns {Array<number>} List of ranks
     * @private
     */
    async _getTokenRanksParallel(tokens) {
        // For now, fall back to sequential processing
        // In a full implementation, this would use worker threads or batch processing
        return await this._getTokenRanksSequential(tokens);
    }

    /**
     * Get model predictions for given context using direct model access
     * @param {string} context - Context text
     * @param {boolean} useCache - Whether to use cache (return past_key_values)
     * @returns {Array|Object} Sorted predictions with probabilities, optionally with pastKeyValues
     * @private
     */
    async _getModelPredictions(context, useCache = false) {
        try {
            // Handle empty context by using a minimal input
            if (!context || context.trim().length === 0) {
                // Use a minimal context - just use the BOS token or a single space
                context = ' ';
            }
            
            // Prepare model inputs in the correct format for transformers.js
            const inputs = await this.tokenizer(context, {
                return_tensors: "pt", // Use PyTorch-style tensors
                padding: false,
                truncation: false,
                add_special_tokens: true
            });
            
            // Debug: check if inputs are valid
            if (!inputs.input_ids || !inputs.input_ids.dims || inputs.input_ids.dims.length < 2 || inputs.input_ids.dims[1] === 0) {
                throw new Error('Invalid tokenizer inputs - empty token sequence');
            }
            
            // Get model output with logits and optionally past_key_values
            const output = await this.model(inputs, {
                use_cache: useCache && this.useCache
            });
            
            if (output && output.logits) {
                const predictions = this._processLogitsToRanking(output.logits);
                
                if (useCache && this.useCache) {
                    return {
                        predictions: predictions,
                        pastKeyValues: output.past_key_values
                    };
                } else {
                    return predictions;
                }
            } else {
                throw new Error('No logits found in model output');
            }
        } catch (error) {
            throw new Error(`Model prediction failed: ${error.message}`);
        }
    }

    /**
     * Get model predictions using incremental context with KV caching
     * @param {number} tokenId - Single token ID to add to cached context
     * @param {Object} pastKeyValues - Cached past key values
     * @returns {Object} Predictions and updated pastKeyValues
     * @private
     */
    async _getModelPredictionsIncremental(tokenId, pastKeyValues) {
        try {
            // Convert token ID to text and then tokenize to get proper inputs
            const tokenText = await this.tokenizer.decode([tokenId], { skip_special_tokens: false });
            
            // Create proper input format using tokenizer
            const inputs = await this.tokenizer(tokenText, {
                return_tensors: "pt",
                padding: false,
                truncation: false,
                add_special_tokens: false
            });
            
            // Get model output with cached context
            const output = await this.model(inputs, {
                past_key_values: pastKeyValues,
                use_cache: true
            });
            
            if (output && output.logits) {
                const predictions = this._processLogitsToRanking(output.logits);
                
                return {
                    predictions: predictions,
                    pastKeyValues: output.past_key_values
                };
            } else {
                throw new Error('No logits found in model output');
            }
        } catch (error) {
            throw new Error(`Incremental model prediction failed: ${error.message}`);
        }
    }

    /**
     * Process model logits to create token ranking
     * @param {Object} logits - Model logits tensor
     * @returns {Array} Sorted predictions with probabilities
     * @private
     */
    _processLogitsToRanking(logits) {
        try {
            // Get the last position logits (next token prediction)
            let lastLogits;
            
            // Handle different tensor formats
            if (logits.data) {
                // Tensor format: extract data array
                const shape = logits.dims || logits.shape;
                const data = logits.data;
                
                if (!shape || !data) {
                    throw new Error('Invalid logits tensor format');
                }
                
                if (shape.length === 3) {
                    // [batch_size, sequence_length, vocab_size]
                    const batchSize = shape[0];
                    const seqLength = shape[1];
                    const vocabSize = shape[2];
                    
                    // Get logits for the last position
                    const lastPosStart = (seqLength - 1) * vocabSize;
                    lastLogits = Array.from(data.slice(lastPosStart, lastPosStart + vocabSize));
                } else if (shape.length === 2) {
                    // [sequence_length, vocab_size] - already last position
                    lastLogits = Array.from(data);
                } else {
                    throw new Error(`Unexpected logits shape: ${shape}`);
                }
            } else if (Array.isArray(logits)) {
                lastLogits = logits;
            } else {
                throw new Error('Unknown logits format');
            }
            
            // Convert logits to probabilities using softmax
            const maxLogit = Math.max(...lastLogits);
            const expLogits = lastLogits.map(logit => Math.exp(logit - maxLogit));
            const sumExp = expLogits.reduce((sum, exp) => sum + exp, 0);
            const probabilities = expLogits.map(exp => exp / sumExp);
            
            // Create token predictions with probabilities
            const predictions = probabilities.map((prob, tokenId) => ({
                token_id: tokenId,
                probability: prob
            }));
            
            // Sort by probability (descending)
            predictions.sort((a, b) => b.probability - a.probability);
            
            return predictions;
        } catch (error) {
            throw new Error(`Failed to process logits: ${error.message}`);
        }
    }


    /**
     * Find the rank of a specific token in predictions
     * @param {number} tokenId - Token ID to find
     * @param {Array} predictions - Sorted predictions
     * @returns {number} Rank (1-indexed)
     * @private
     */
    _findTokenRank(tokenId, predictions) {
        for (let i = 0; i < predictions.length; i++) {
            if (predictions[i].token_id === tokenId) {
                return i + 1; // 1-indexed rank
            }
        }
        return predictions.length; // Fallback to worst rank
    }

    /**
     * Generate text by selecting tokens based on their ranks
     * @param {Array<number>} ranks - List of ranks (1-indexed)
     * @param {number} maxLength - Maximum length to generate
     * @returns {string} Generated text
     */
    async getStringFromTokenRanks(ranks, maxLength = null) {
        await this._initialize();
        
        if (!ranks || ranks.length === 0) {
            return '';
        }

        const targetLength = maxLength || ranks.length;
        const selectedRanks = ranks.slice(0, targetLength);
        
        // Use appropriate processing method based on batch size
        if (this.batchSize === 1) {
            return await this._getStringFromTokenRanksSequential(selectedRanks);
        } else {
            return await this._getStringFromTokenRanksParallel(selectedRanks);
        }
    }

    /**
     * Sequential implementation of string generation from ranks with KV caching
     * @param {Array<number>} ranks - List of ranks
     * @returns {string} Generated text
     * @private
     */
    async _getStringFromTokenRanksSequential(ranks) {
        const tokens = [];
        let pastKeyValues = null;
        
        for (let i = 0; i < ranks.length; i++) {
            let predictions;
            
            if (this.useCache && i > 0) {
                // Use cached context with incremental token
                const previousToken = tokens[i - 1];
                predictions = await this._getModelPredictionsIncremental(previousToken, pastKeyValues);
                pastKeyValues = predictions.pastKeyValues;
            } else {
                // For first token or when not using cache, use full context
                const contextText = await this.decode(tokens);
                predictions = await this._getModelPredictions(contextText, true);
                if (this.useCache) {
                    pastKeyValues = predictions.pastKeyValues;
                }
            }
            
            // Select token at specified rank
            const targetRank = ranks[i] - 1; // Convert to 0-indexed
            const predictionsList = predictions.predictions || predictions;
            const selectedToken = targetRank < predictionsList.length ? 
                predictionsList[targetRank].token_id : 
                predictionsList[predictionsList.length - 1].token_id;
            
            tokens.push(selectedToken);
            
            // Truncate cache if needed
            if (this.useCache && pastKeyValues) {
                pastKeyValues = this._truncateKvCache(pastKeyValues);
            }
        }
        
        return await this.decode(tokens);
    }

    /**
     * Parallel implementation of string generation from ranks (simplified)
     * @param {Array<number>} ranks - List of ranks
     * @returns {string} Generated text
     * @private
     */
    async _getStringFromTokenRanksParallel(ranks) {
        // For now, fall back to sequential processing
        return await this._getStringFromTokenRanksSequential(ranks);
    }

    /**
     * Create chunks for parallel processing
     * @param {Array} data - Data to chunk
     * @param {number} numChunks - Number of chunks
     * @returns {Array<Array>} Array of chunks
     * @private
     */
    _createChunks(data, numChunks) {
        if (data.length === 0 || numChunks === 1) {
            return [data];
        }

        const chunks = [];
        const baseSize = Math.floor(data.length / numChunks);
        const remainder = data.length % numChunks;
        
        let startIdx = 0;
        for (let i = 0; i < numChunks; i++) {
            const chunkSize = baseSize + (i < remainder ? 1 : 0);
            const endIdx = startIdx + chunkSize;
            
            if (startIdx >= data.length) break;
            
            chunks.push(data.slice(startIdx, endIdx));
            startIdx = endIdx;
        }
        
        return chunks;
    }
}

/**
 * Convenience function to get token ranks for a text string
 * @param {string} text - Input text
 * @param {string} modelName - HuggingFace model name
 * @param {number} batchSize - Batch size for processing
 * @param {boolean} useCache - Whether to use KV caching for efficiency
 * @returns {Array<number>} List of ranks
 */
export async function rankTokens(text, modelName = 'Xenova/distilgpt2', batchSize = 1, useCache = true) {
    const ranker = new LLMRanker({ modelName, batchSize, useCache });
    return await ranker.getTokenRanks(text);
}