import { createHash } from 'crypto';

// Try to import sqlite3, fall back to mock if not available
let Database;
try {
    const sqlite3 = await import('sqlite3');
    Database = sqlite3.Database;
} catch (error) {
    // Mock Database class for testing without sqlite3
    Database = class MockDatabase {
        constructor() {}
        run() {}
        get(query, params, callback) { callback(null, null); }
        all(query, params, callback) { callback(null, []); }
        close() {}
    };
}

/**
 * Cache for LLM ranker operations to avoid recomputing expensive operations
 */
export class RankerCache {
    /**
     * Initialize the ranker cache
     * @param {string} dbPath - Path to SQLite database
     */
    constructor(dbPath = 'compression_results.db') {
        this.dbPath = dbPath;
        this._setupCacheTables();
    }

    /**
     * Initialize cache tables in the database
     * @private
     */
    _setupCacheTables() {
        const db = new Database(this.dbPath);
        
        // Create token ranks cache table
        db.run(`
            CREATE TABLE IF NOT EXISTS token_ranks_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text_hash TEXT NOT NULL,
                ranker_signature TEXT NOT NULL,
                text_sample TEXT NOT NULL,
                ranks TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(text_hash, ranker_signature)
            )
        `);
        
        // Create string from ranks cache table
        db.run(`
            CREATE TABLE IF NOT EXISTS string_from_ranks_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ranks_hash TEXT NOT NULL,
                ranker_signature TEXT NOT NULL,
                ranks_sample TEXT NOT NULL,
                result_text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ranks_hash, ranker_signature)
            )
        `);
        
        db.close();
    }

    /**
     * Generate a hash for input text
     * @param {string} text - Input text
     * @returns {string} Hash string
     * @private
     */
    _getTextHash(text) {
        return createHash('sha256').update(text, 'utf8').digest('hex');
    }

    /**
     * Generate a hash for input ranks
     * @param {Array<number>} ranks - Input ranks
     * @returns {string} Hash string
     * @private
     */
    _getRanksHash(ranks) {
        const ranksStr = JSON.stringify(ranks);
        return createHash('sha256').update(ranksStr, 'utf8').digest('hex');
    }

    /**
     * Get string representation of ranker signature
     * @param {Object} ranker - Ranker instance
     * @returns {string} Signature string
     * @private
     */
    _getRankerSignatureStr(ranker) {
        const signature = ranker.getCacheSignature();
        return JSON.stringify(signature);
    }

    /**
     * Retrieve cached token ranks if available
     * @param {string} text - Input text
     * @param {Object} ranker - Ranker instance
     * @returns {Promise<Array<number>|null>} Cached ranks or null
     */
    async getCachedTokenRanks(text, ranker) {
        const textHash = this._getTextHash(text);
        const rankerSignature = this._getRankerSignatureStr(ranker);
        
        return new Promise((resolve, reject) => {
            const db = new Database(this.dbPath);
            
            const query = `
                SELECT ranks FROM token_ranks_cache 
                WHERE text_hash = ? AND ranker_signature = ?
            `;
            
            db.get(query, [textHash, rankerSignature], (err, row) => {
                db.close();
                if (err) {
                    reject(err);
                } else if (row) {
                    resolve(JSON.parse(row.ranks));
                } else {
                    resolve(null);
                }
            });
        });
    }

    /**
     * Cache computed token ranks
     * @param {string} text - Input text
     * @param {Object} ranker - Ranker instance
     * @param {Array<number>} ranks - Computed ranks
     * @returns {Promise<void>}
     */
    async cacheTokenRanks(text, ranker, ranks) {
        const textHash = this._getTextHash(text);
        const rankerSignature = this._getRankerSignatureStr(ranker);
        const ranksJson = JSON.stringify(ranks);
        const textSample = text.substring(0, 100);
        
        return new Promise((resolve, reject) => {
            const db = new Database(this.dbPath);
            
            const query = `
                INSERT OR REPLACE INTO token_ranks_cache 
                (text_hash, ranker_signature, text_sample, ranks)
                VALUES (?, ?, ?, ?)
            `;
            
            db.run(query, [textHash, rankerSignature, textSample, ranksJson], function(err) {
                db.close();
                if (err) {
                    reject(err);
                } else {
                    resolve();
                }
            });
        });
    }

    /**
     * Retrieve cached string from ranks if available
     * @param {Array<number>} ranks - Input ranks
     * @param {Object} ranker - Ranker instance
     * @returns {Promise<string|null>} Cached string or null
     */
    async getCachedStringFromRanks(ranks, ranker) {
        const ranksHash = this._getRanksHash(ranks);
        const rankerSignature = this._getRankerSignatureStr(ranker);
        
        return new Promise((resolve, reject) => {
            const db = new Database(this.dbPath);
            
            const query = `
                SELECT result_text FROM string_from_ranks_cache 
                WHERE ranks_hash = ? AND ranker_signature = ?
            `;
            
            db.get(query, [ranksHash, rankerSignature], (err, row) => {
                db.close();
                if (err) {
                    reject(err);
                } else if (row) {
                    resolve(row.result_text);
                } else {
                    resolve(null);
                }
            });
        });
    }

    /**
     * Cache computed string from ranks
     * @param {Array<number>} ranks - Input ranks
     * @param {Object} ranker - Ranker instance
     * @param {string} resultText - Computed result text
     * @returns {Promise<void>}
     */
    async cacheStringFromRanks(ranks, ranker, resultText) {
        const ranksHash = this._getRanksHash(ranks);
        const rankerSignature = this._getRankerSignatureStr(ranker);
        const ranksJson = JSON.stringify(ranks);
        const ranksSample = ranksJson.substring(0, 100);
        
        return new Promise((resolve, reject) => {
            const db = new Database(this.dbPath);
            
            const query = `
                INSERT OR REPLACE INTO string_from_ranks_cache 
                (ranks_hash, ranker_signature, ranks_sample, result_text)
                VALUES (?, ?, ?, ?)
            `;
            
            db.run(query, [ranksHash, rankerSignature, ranksSample, resultText], function(err) {
                db.close();
                if (err) {
                    reject(err);
                } else {
                    resolve();
                }
            });
        });
    }

    /**
     * Get token ranks with caching
     * @param {string} text - Input text
     * @param {Object} ranker - Ranker instance
     * @returns {Promise<Array<number>>} Token ranks
     */
    async getTokenRanksCached(text, ranker) {
        // Check cache first
        const cachedRanks = await this.getCachedTokenRanks(text, ranker);
        if (cachedRanks !== null) {
            return cachedRanks;
        }
        
        // Not in cache, compute using ranker
        const ranks = await ranker.getTokenRanks(text);
        
        // Cache the result
        await this.cacheTokenRanks(text, ranker, ranks);
        
        return ranks;
    }

    /**
     * Get string from token ranks with caching
     * @param {Array<number>} ranks - Input ranks
     * @param {Object} ranker - Ranker instance
     * @returns {Promise<string>} Generated text
     */
    async getStringFromTokenRanksCached(ranks, ranker) {
        // Check cache first
        const cachedResult = await this.getCachedStringFromRanks(ranks, ranker);
        if (cachedResult !== null) {
            return cachedResult;
        }
        
        // Not in cache, compute using ranker
        const resultText = await ranker.getStringFromTokenRanks(ranks);
        
        // Cache the result
        await this.cacheStringFromRanks(ranks, ranker, resultText);
        
        return resultText;
    }

    /**
     * Clear all cached data
     * @returns {Promise<void>}
     */
    async clearCache() {
        return new Promise((resolve, reject) => {
            const db = new Database(this.dbPath);
            
            db.serialize(() => {
                db.run('DELETE FROM token_ranks_cache');
                db.run('DELETE FROM string_from_ranks_cache', function(err) {
                    db.close();
                    if (err) {
                        reject(err);
                    } else {
                        resolve();
                    }
                });
            });
        });
    }

    /**
     * Get cache statistics
     * @returns {Promise<Object>} Cache statistics
     */
    async getCacheStats() {
        return new Promise((resolve, reject) => {
            const db = new Database(this.dbPath);
            
            const stats = {};
            
            db.serialize(() => {
                db.get('SELECT COUNT(*) as count FROM token_ranks_cache', (err, row) => {
                    if (err) {
                        db.close();
                        reject(err);
                        return;
                    }
                    stats.tokenRanksEntries = row.count;
                });
                
                db.get('SELECT COUNT(*) as count FROM string_from_ranks_cache', (err, row) => {
                    if (err) {
                        db.close();
                        reject(err);
                        return;
                    }
                    stats.stringFromRanksEntries = row.count;
                });
                
                db.all('SELECT DISTINCT ranker_signature FROM token_ranks_cache', (err, rows) => {
                    if (err) {
                        db.close();
                        reject(err);
                        return;
                    }
                    stats.uniqueRankerSignaturesToken = rows.length;
                });
                
                db.all('SELECT DISTINCT ranker_signature FROM string_from_ranks_cache', (err, rows) => {
                    db.close();
                    if (err) {
                        reject(err);
                        return;
                    }
                    stats.uniqueRankerSignaturesString = rows.length;
                    stats.totalEntries = stats.tokenRanksEntries + stats.stringFromRanksEntries;
                    resolve(stats);
                });
            });
        });
    }
}