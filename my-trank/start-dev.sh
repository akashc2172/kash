#!/bin/bash
# Start the development server for my-trank

cd "$(dirname "$0")"

echo "🚀 Starting my-trank development server..."
echo ""

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Start the dev server
echo "🌐 Starting Vite dev server..."
echo "   The site will be available at http://localhost:5173"
echo ""

npm run dev
