/*
 * The browser entry point. index.html loads this file (through Vite), and this file puts the
 * React app into the page. It calls no backend routes itself; App.jsx and the panels do that.
 */
// StrictMode is a development helper: it runs some code twice on purpose to expose unsafe effects.
import { StrictMode } from 'react'
// createRoot is React 18's way to attach a React tree to a real DOM element.
import { createRoot } from 'react-dom/client'
// Importing a CSS file makes Vite bundle it and add it to the page.
import './index.css'
// The studio shell: header, health dot and the three tabs.
import App from './App.jsx'

// Find the empty <div id="root"> in index.html and render the whole app inside it.
createRoot(document.getElementById('root')).render(
  <StrictMode>
    {/* Everything below runs with StrictMode's extra checks, which are off in production builds. */}
    <App />
  </StrictMode>,
)
