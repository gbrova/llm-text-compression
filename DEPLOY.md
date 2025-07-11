# GitHub Pages Deployment

This repository is configured to automatically deploy the JavaScript compression demo to GitHub Pages.

## Setup

1. **Enable GitHub Pages**: Go to your repository settings > Pages
2. **Set Source**: Choose "GitHub Actions" as the source
3. **Push to main**: The deployment will trigger automatically when you push to the main branch

## Local Development

To run the application locally:

```bash
cd javascript
npm install
npm run dev
```

This will start the Vite development server at `http://localhost:8000`.

## Building for Production

To build the application for production:

```bash
cd javascript
npm run build
```

The built files will be in the `javascript/dist/` directory.

## How it Works

- The GitHub Actions workflow (`.github/workflows/deploy.yml`) automatically:
  1. Installs Node.js dependencies
  2. Builds the JavaScript application using Vite
  3. Deploys the built files to GitHub Pages

- The application is a static site that runs entirely in the browser
- It uses the `@huggingface/transformers` library for LLM inference
- All processing happens client-side, no backend server required

## Accessing the Demo

Once deployed, the demo will be available at:
`https://[your-username].github.io/[repository-name]/`

The demo provides a web interface for testing LLM-based text compression with various models and parameters.