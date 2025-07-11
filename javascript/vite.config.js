import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  // The root directory where index.html is located
  root: '.',
  
  // Build configuration
  build: {
    // Output directory for the built files
    outDir: 'dist',
    
    // Ensure the dist directory is cleaned before building
    emptyOutDir: true,
    
    // Configure the entry point - Vite will use index.html as the main file
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html')
      }
    },
    
    // Configure asset handling
    assetsDir: 'assets',
    
    // Enable source maps for debugging
    sourcemap: true,
    
    // Configure chunk size warnings
    chunkSizeWarningLimit: 1000
  },
  
  // Development server configuration
  server: {
    port: 8000,
    host: 'localhost',
    open: true
  },
  
  // Base URL for deployment 
  // Use repository name as base for GitHub Pages, or './' for local development
  base: process.env.NODE_ENV === 'production' ? './' : './',
  
  // Configure how modules are resolved
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src')
    }
  },
  
  // Configure handling of dependencies
  optimizeDeps: {
    include: ['@huggingface/transformers']
  }
});