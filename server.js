#!/usr/bin/env node
/**
 * Simple HTTP server for the LLM Rank Visualizer
 * Serves static files and handles ES6 module imports properly
 */

import http from 'http';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration
const PORT = process.env.PORT || 8000;
const HOST = process.env.HOST || 'localhost';

// MIME types for different file extensions
const mimeTypes = {
    '.html': 'text/html',
    '.js': 'application/javascript',
    '.css': 'text/css',
    '.json': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.gif': 'image/gif',
    '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon'
};

function getMimeType(filePath) {
    const ext = path.extname(filePath).toLowerCase();
    return mimeTypes[ext] || 'text/plain';
}

function serveFile(res, filePath) {
    try {
        const content = fs.readFileSync(filePath);
        const mimeType = getMimeType(filePath);
        
        res.writeHead(200, {
            'Content-Type': mimeType,
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization',
            'Cross-Origin-Embedder-Policy': 'cross-origin',
            'Cross-Origin-Opener-Policy': 'cross-origin'
        });
        
        res.end(content);
    } catch (error) {
        res.writeHead(404, { 'Content-Type': 'text/plain' });
        res.end('File not found');
    }
}

function serveNodeModules(res, modulePath) {
    // Serve files from node_modules
    const fullPath = path.join(__dirname, 'node_modules', modulePath);
    
    if (fs.existsSync(fullPath)) {
        serveFile(res, fullPath);
    } else {
        res.writeHead(404, { 'Content-Type': 'text/plain' });
        res.end('Module not found');
    }
}

const server = http.createServer((req, res) => {
    let url = req.url;
    
    // Handle root path
    if (url === '/') {
        url = '/visualizer-client.html';
    }
    
    // Handle node_modules requests
    if (url.startsWith('/@huggingface/transformers') || url.startsWith('/node_modules/@huggingface/transformers')) {
        const modulePath = url.replace('/node_modules/', '').replace('/@huggingface/transformers', '@huggingface/transformers');
        serveNodeModules(res, modulePath);
        return;
    }
    
    // Handle other node_modules requests
    if (url.startsWith('/node_modules/')) {
        const modulePath = url.replace('/node_modules/', '');
        serveNodeModules(res, modulePath);
        return;
    }
    
    // Construct file path
    const filePath = path.join(__dirname, url);
    
    // Security check - prevent directory traversal
    if (!filePath.startsWith(__dirname)) {
        res.writeHead(403, { 'Content-Type': 'text/plain' });
        res.end('Forbidden');
        return;
    }
    
    // Check if file exists
    if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
        serveFile(res, filePath);
    } else {
        res.writeHead(404, { 'Content-Type': 'text/plain' });
        res.end('File not found');
    }
});

server.listen(PORT, HOST, () => {
    console.log('🚀 LLM Rank Visualizer Server');
    console.log(`📁 Serving files from: ${__dirname}`);
    console.log(`🌐 Server running at: http://${HOST}:${PORT}`);
    console.log(`📄 Visualizer: http://${HOST}:${PORT}/visualizer-client.html`);
    console.log('💡 Press Ctrl+C to stop the server');
    
    // Try to open browser automatically (if available)
    if (process.platform === 'darwin') {
        import('child_process').then(({ exec }) => {
            exec(`open http://${HOST}:${PORT}`);
        });
    } else if (process.platform === 'win32') {
        import('child_process').then(({ exec }) => {
            exec(`start http://${HOST}:${PORT}`);
        });
    } else {
        import('child_process').then(({ exec }) => {
            exec(`xdg-open http://${HOST}:${PORT}`);
        });
    }
});

// Handle server shutdown gracefully
process.on('SIGINT', () => {
    console.log('\n👋 Server stopped');
    process.exit(0);
});

process.on('SIGTERM', () => {
    console.log('\n👋 Server stopped');
    process.exit(0);
});