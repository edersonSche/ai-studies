# Neural Network - TensorFlow.js

A simple neural network built with TensorFlow.js that classifies people into categories (premium, medium, basic) based on age, favorite color, and location.

## Prerequisites

- Node.js
- npm

## Installation

```bash
npm install
```

## Usage

```bash
npm start
```

## How It Works

The neural network uses the following input features (7 values):
- Normalized age
- One-hot encoded color (blue, red, green)
- One-hot encoded location (São Paulo, Rio, Curitiba)

It outputs 3 probability scores for each category:
- Premium
- Medium
- Basic
