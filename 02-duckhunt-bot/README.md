# DUCK HUNT JS v3.0

[Play the game](https://duckhuntjs.com)

This is an implementation of DuckHunt in Javascript and HTML5. It uses the PixiJS rendering engine, Green Sock Animations, Howler, and Bluebird Promises.

## Features

- **Rendering** - Supports WebGL and Canvas rendering via the PixiJS rendering engine
- **Audio** - Uses WebAudioAPI with HTML5 Audio fallback. Audio is loaded and controlled via HowlerJS
- **Tweening** - Animations combine PixiJS MovieClips built from sprite images and tweens using Green Sock
- **Game Logic** - Flow managed using Javascript. Business logic implemented as ES6 classes transpiled to ES5 using Babel
- **AI Bot** - YOLOv5n object detection running in a Web Worker via TensorFlow.js. Captures the game canvas every 200ms, detects ducks, and auto-clicks on them

## Getting Started

1. Install [nodejs](https://nodejs.org/)
2. Clone the repo
3. Run `npm install`
4. Start the dev server with `npm start`
5. Open http://localhost:8080/ in your browser

> **Note:** Cross origin errors prevent access via `file://` protocol. The dev server includes automatic builds and reloads when changes are detected in the `src` directory.

To manually build the application: `npm run build`

## Development

This repo ships with committed dist files for easy setup. To modify audio and visual assets:

- **Audio assets** - `npm run audio` (requires [ffmpeg](https://ffmpeg.org/download.html))
- **Visual assets** - `npm run images` (requires [texturepacker](https://www.codeandweb.com/texturepacker/download))
